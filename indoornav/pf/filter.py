"""지도 제약 파티클 필터 (서버용 코어).

상태(파티클 1개): x, y [m, 지도 좌표], th [rad, 지도 기준 진행 방향, 반시계 +], k [보폭 배율]
입력: 걸음마다 (length, dheading) — PDR 출력 그대로 (앱 → 서버 메시지와 같은 형식)
관측: 위치 고정 (랜드마크/체크포인트), 가우시안 + 균등 바닥값 혼합
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .map import WallMap


@dataclass
class PFParams:
    n: int = 1000
    sigma_len: float = 0.10        # 걸음당 보폭 잡음 (비율)
    sigma_h0: float = np.radians(0.5)   # 걸음당 헤딩 잡음 [rad]
    sigma_h1: float = 0.01         # 회전량에 비례하는 헤딩 잡음 (|dheading| 배율)
    sigma_k: float = 0.005         # 걸음당 k 랜덤워크
    k_min: float = 0.7
    k_max: float = 1.4
    k0_lo: float = 0.80            # 초기 k 균등분포 범위 (평균 1.0)
    k0_hi: float = 1.20
    init_pos_sigma: float = 0.3    # 초기 위치 불확실도 [m]
    init_h_sigma: float = np.radians(5.0)   # 초기 헤딩 불확실도 [rad] (None 대신 큰 값 = 모름)
    init_h_unknown: bool = False   # True면 초기 헤딩 U(0, 2π)
    sigma_obs: float = 0.5         # 위치 관측 표준편차 [m]
    eps_obs: float = 0.05          # 관측 이상치 바닥값 (혼합 비율)
    ess_frac: float = 0.5          # ESS < n·ess_frac 이면 리샘플
    rough_xy: float = 0.05         # 리샘플 후 지터
    rough_h: float = np.radians(0.5)
    rough_k: float = 0.005
    use_map: bool = True
    # PDR 이 '제자리 움직임'으로 보고 보폭 0 을 준 걸음(moving=False)을 어떻게 볼지.
    # 0 이 확실하다고 믿으면 극단적으로 느린 걸음(묶음 규칙이 실제 걸음을 지움)에서 PF 가 과신한다.
    # p > 0 이면: 확률 p 로 실제 이동이 있었다고 보고 길이 ~ U(lo, hi) (파티클마다 따로 뽑음).
    nonmoving_p: float = 0.7          # 2026-10-29 극단 느림 세션으로 검증 (0.0 이면 예전 동작)
    nonmoving_lo: float = 0.10
    nonmoving_hi: float = 0.40
    # 최근 걸음 중 보폭 0(제자리 움직임) 비율이 이 이상이면 'PDR 신뢰 낮음' 으로 알림 (앱: "위치 불확실" 안내)
    suspect_window: int = 10
    suspect_frac: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class ParticleFilter:
    def __init__(self, params: PFParams, wall_map: WallMap | None, seed: int = 0):
        self.p = params
        self.map = wall_map if params.use_map else None
        self.rng = np.random.default_rng(seed)
        self.stats = {"resamples": 0, "all_dead": 0, "wall_kills": 0}
        self._recent_moving: list[bool] = []

    # ---------------------------------------------------------------- 초기화
    def init(self, xy0, th0: float) -> None:
        p, n, r = self.p, self.p.n, self.rng
        xy0 = np.asarray(xy0, float)
        self.xy = xy0 + r.normal(0, p.init_pos_sigma, (n, 2))
        if self.map is not None:   # 벽 너머에 뿌려진 초기 파티클은 출발점으로
            bad = self.map.crosses(np.repeat(xy0[None], n, 0), self.xy)
            self.xy[bad] = xy0
        self.th = (r.uniform(0, 2 * np.pi, n) if p.init_h_unknown
                   else th0 + r.normal(0, p.init_h_sigma, n))
        self.k = r.uniform(p.k0_lo, p.k0_hi, n)
        self.logw = np.full(n, -np.log(n))

    # ---------------------------------------------------------------- 예측
    def predict(self, length: float, dheading: float, moving: bool = True) -> None:
        p, n, r = self.p, self.p.n, self.rng
        self._recent_moving = (self._recent_moving + [bool(moving)])[-p.suspect_window:]
        self.k = np.clip(self.k + r.normal(0, p.sigma_k, n), p.k_min, p.k_max)
        self.th = self.th + dheading + r.normal(0, p.sigma_h0 + p.sigma_h1 * abs(dheading), n)
        if not moving and p.nonmoving_p > 0:
            # 보폭 0 이 맞는지 모르는 걸음: 일부 파티클만 짧게 이동
            go = r.random(n) < p.nonmoving_p
            L = np.where(go, r.uniform(p.nonmoving_lo, p.nonmoving_hi, n), 0.0)
            if not go.any():
                return
        elif length <= 0:
            return
        else:
            L = length * self.k * (1 + r.normal(0, p.sigma_len, n))
        new = self.xy + L[:, None] * np.column_stack([np.cos(self.th), np.sin(self.th)])
        if self.map is not None:
            hit = self.map.crosses(self.xy, new)
            if hit.all():
                new, hit = self._recover(L)
            self.stats["wall_kills"] += int(hit.sum())
            self.logw = np.where(hit, -np.inf, self.logw)
            new = np.where(hit[:, None], self.xy, new)   # 죽은 파티클은 벽 너머로 옮기지 않음
        self.xy = new
        self._normalize()
        # 죽은 파티클이 남아 있으면 바로 리샘플 (가중치 0 파티클이 나중에 되살아나는 버그 방지)
        self._maybe_resample(force=self.map is not None and bool(np.isneginf(self.logw).any()))

    # ---------------------------------------------------------------- 관측
    def observe_position(self, z, sigma: float | None = None, eps: float | None = None) -> None:
        p = self.p
        s = p.sigma_obs if sigma is None else sigma
        e = p.eps_obs if eps is None else eps
        d2 = np.sum((self.xy - np.asarray(z, float)) ** 2, axis=1)
        lik = (1 - e) * np.exp(-0.5 * d2 / s**2) + e * np.exp(-0.5 * 9.0)   # 바닥값 = 3σ 지점 수준
        self.logw = self.logw + np.log(lik)
        self._normalize()
        self._maybe_resample()

    # ---------------------------------------------------------------- 추정
    def estimate(self) -> dict:
        w = np.exp(self.logw)
        m = w @ self.xy
        cov = (self.xy - m).T @ ((self.xy - m) * w[:, None])
        c, s = w @ np.cos(self.th), w @ np.sin(self.th)
        return {"xy": m, "th": float(np.arctan2(s, c)), "k": float(w @ self.k),
                "k_std": float(np.sqrt(w @ (self.k - w @ self.k) ** 2)),
                "unc95": float(np.sqrt(5.991 * np.linalg.eigvalsh(cov).max())),
                "ess": self.ess(), "pdr_suspect": self.pdr_suspect()}

    def pdr_suspect(self) -> bool:
        """최근 걸음 대부분이 보폭 0 이면 True (극단적으로 느린 걸음 등 PDR 범위 밖)."""
        w = self._recent_moving
        return len(w) >= self.p.suspect_window and (1 - np.mean(w)) >= self.p.suspect_frac

    def ess(self) -> float:
        w = np.exp(self.logw)
        return float(1.0 / np.sum(w**2))

    # ---------------------------------------------------------------- 내부
    def _recover(self, L: np.ndarray):
        """모든 파티클이 벽에 막혔을 때: 위치는 그대로 두고 주변으로 퍼뜨린다 (지도 안에서만).

        보폭·헤딩 오차가 모델 범위를 넘었다는 뜻이므로, 이번 걸음 이동량만큼 아무 방향으로나
        흩어서 살아남는 파티클로 다시 시작한다 (헤딩도 이동 방향으로 바꾸지 않음 — 다음 회전에서 걸러짐).
        """
        self.stats["all_dead"] += 1
        n, r = self.p.n, self.rng
        for scale in (1.0, 2.0, 4.0):
            ang = r.uniform(0, 2 * np.pi, n)
            d = np.abs(L)[:, None] * scale * np.column_stack([np.cos(ang), np.sin(ang)])
            new = self.xy + d
            hit = self.map.crosses(self.xy, new)
            if not hit.all():
                return new, hit
        return self.xy.copy(), np.zeros(n, bool)

    def _normalize(self) -> None:
        m = np.max(self.logw)
        if not np.isfinite(m):
            self.logw = np.full(self.p.n, -np.log(self.p.n))
            self.stats["all_dead"] += 1
            return
        self.logw = self.logw - (m + np.log(np.sum(np.exp(self.logw - m))))

    def _maybe_resample(self, force: bool = False) -> None:
        if not force and self.ess() >= self.p.ess_frac * self.p.n:
            return
        p, n, r = self.p, self.p.n, self.rng
        w = np.exp(self.logw)
        idx = np.searchsorted(np.cumsum(w), (r.random() + np.arange(n)) / n)
        idx = np.minimum(idx, n - 1)
        base = self.xy[idx]
        jit = base + r.normal(0, p.rough_xy, (n, 2))
        if self.map is not None:   # 지터로 벽을 넘지 않게
            jit = np.where(self.map.crosses(base, jit)[:, None], base, jit)
        self.xy = jit
        self.th = self.th[idx] + r.normal(0, p.rough_h, n)
        self.k = np.clip(self.k[idx] + r.normal(0, p.rough_k, n), p.k_min, p.k_max)
        self.logw = np.full(n, -np.log(n))
        self.stats["resamples"] += 1
