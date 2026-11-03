"""VisionFilter: indoornav ParticleFilter(indoornav.pf.filter)에 비전 관측(obs_model.obs_likelihood)을 붙이는 어댑터 (연결안 3절 6).

indoornav 코드는 고치지 않고 불러 쓴다. 시뮬 PF 재생기의 필터 약속(predict·update·change_floor·estimate)과 같은 메서드.
    predict      : indoornav pf.predict(length, dheading, moving) 그대로. dheading은 indoornav 부호(반시계 +, rad)로 받음
    update       : 파티클 aligned → ENU·나침반 → obs_likelihood(관측 한 프레임) → pf.logw += log(w) → pf._normalize()
                   → pf._maybe_resample()  (indoornav PF에 외부 점수 넣는 공개 함수가 없어 비공개 메서드 호출, 연결안 2절·4절 P1)
    change_floor : 층 바뀜(기압). 파티클마다 가장 가까운 에스컬레이터 위 끝 → 그 에스컬레이터 아래 끝 ±1 m(축마다 σ 1 m,
                   시뮬 기준과 같음)를 **목적지 보행 마스크 안에서** 다시 뽑음(벽 밖 표본이 끝점 한 점에 몰리지 않게).
                   가중치는 그대로(균등화 안 함). 끝점은 그래프 좌표(역사 패키지 escalators, 키로 찾음)를 aligned로
    rotate       : 탑승 중 모아 둔 회전을 하차 뒤 한 번에(indoornav predict(0, dh) 한 번 = 방향 잡음·k 랜덤워크 한 번)
    estimate     : indoornav estimate(xy·th·k·unc95·ess·pdr_suspect) + RMS 불확실도 + 상위 두 모드 + 층
좌표: aligned = Rot(rot)·ENU (원점 그대로, en_to_st 안 씀), 나침반 = wrap360(90 + rot − deg(th)), rot = −52.73°
초기화(known·drift): pf.init() 뒤 xy·th를 덮어씀 — N(xy0, σ²I)에서 GridMap.free 만족 표본만 거절 샘플링, 방향 N(th0, σ_h²),
    k는 pf.init()의 U(k0_lo, k0_hi) 그대로, logw 균등 (indoornav init()은 벽 너머 파티클을 출발점 한 점에 모아서)
"""
import time
from dataclasses import replace

import numpy as np

import paths
from indoornav.pf.filter import ParticleFilter
from indoornav.pf.gridmap import GridMap
from obs_model import obs_likelihood, _cells
from station_pkg import aligned_to_enu, enu_to_aligned, th_to_compass

ESC_SNAP_SD = 1.0       # 하차 위치 퍼짐 [m] (시뮬 기준 change_floor와 같음)


def load_maps(meta=paths.MASK_META):
    return {fl: GridMap.load(meta, name) for fl, name in paths.FLOOR_NAME.items()}


def top_modes(xy, w, r=3.0, bin_m=1.0):
    """가중 파티클 → 상위 두 모드 [(중심 xy, 질량)]. 모드 = 가중 히스토그램 최대 칸에서 시작해 반경 r 평균 3번."""
    out, left = [], np.ones(len(w), bool)
    for _ in range(2):
        if w[left].sum() < 1e-3:
            break
        p, q = xy[left], w[left]
        lo = p.min(0)
        ij = np.floor((p - lo) / bin_m).astype(int)
        key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
        c = p[key == np.bincount(key, weights=q).argmax()].mean(0)
        for _ in range(3):
            sel = np.hypot(*(p - c).T) < r
            c = (p[sel] * q[sel, None]).sum(0) / q[sel].sum()
        sel_all = left & (np.hypot(*(xy - c).T) < r)
        out.append((c, float(w[sel_all].sum())))
        left &= ~sel_all
    return out


class VisionFilter:
    def __init__(self, P, maps, fmap, smeta, floor, xy0_enu, compass0, seed, obs_params=None,
                 init_sigma=2.0, init_h_sigma_deg=20.0):
        self.P, self.maps, self.fmap, self.floor = P, maps, fmap, floor
        self.rot = smeta["aligned_from_enu_rot_deg"]
        self.esc = smeta["escalators"]
        self.obs_params = obs_params
        self.pf = ParticleFilter(P, maps[floor], seed=seed)
        xy0 = enu_to_aligned(xy0_enu, self.rot)[0]
        th0 = np.radians(90.0 + self.rot - compass0)
        self.pf.init(xy0, th0)
        r, n = self.pf.rng, P.n
        xy = np.empty((0, 2))
        while len(xy) < n:                       # 보행 마스크로 자른 N(xy0, σ²I)
            c = xy0 + r.normal(0, init_sigma, (2 * n, 2))
            xy = np.vstack([xy, c[maps[floor].free(c)]])
        self.pf.xy = xy[:n]
        self.pf.th = th0 + r.normal(0, np.radians(init_h_sigma_deg), n)
        self.pf.logw = np.full(n, -np.log(n))
        self.diag = dict(t_predict=0.0, t_update=0.0, n_update=0, miss_sum=0.0, miss_max=0.0, snap_redraw=0)

    # ------------------------------------------------------------ 상태 변환
    def enu_compass(self):
        en = aligned_to_enu(self.pf.xy, self.rot)
        return en[:, 0], en[:, 1], th_to_compass(self.pf.th, self.rot)

    def miss_frac(self, x, y):
        """가시성 표를 계산하지 않은 칸 위 파티클 비율(가중). 그 파티클은 관측마다 바닥값만 받음(연결안 3절 4)."""
        fm = self.fmap[self.floor]
        i, j, inside = _cells(fm, x, y)
        comp = inside & fm["computed"][i, j]
        return float(np.exp(self.pf.logw) @ ~comp)

    # ------------------------------------------------------------ 필터 약속
    def predict(self, length, dheading_rad, moving=True):
        t = time.perf_counter()
        self.pf.predict(length, dheading_rad, moving)
        self.diag["t_predict"] += time.perf_counter() - t

    def rotate(self, dheading_rad):
        self.pf.predict(0.0, dheading_rad, True)

    def update(self, obs):
        if not obs:
            return
        t = time.perf_counter()
        x, y, c = self.enu_compass()
        m = self.miss_frac(x, y)
        self.diag["miss_sum"] += m
        self.diag["miss_max"] = max(self.diag["miss_max"], m)
        w = obs_likelihood(x, y, c, self.floor, obs, self.fmap, self.obs_params)
        self.pf.logw = self.pf.logw + np.log(np.maximum(w, 1e-300))
        self.pf._normalize()
        self.pf._maybe_resample()
        self.diag["t_update"] += time.perf_counter() - t
        self.diag["n_update"] += 1

    def change_floor(self, d):
        pf, r = self.pf, self.pf.rng
        src, dst = ("top_aligned", "bottom_aligned") if d < 0 else ("bottom_aligned", "top_aligned")
        A = np.array([e[src] for e in self.esc])
        B = np.array([e[dst] for e in self.esc])
        k = np.argmin(np.linalg.norm(pf.xy[:, None, :] - A[None], axis=2), axis=1)
        self.floor += d
        g = self.maps[self.floor]
        new = B[k] + r.normal(0, ESC_SNAP_SD, (len(k), 2))
        bad = ~g.free(new)
        for _ in range(50):                      # 목적지 보행 마스크 안에서 다시 뽑기 (가중치는 그대로)
            if not bad.any():
                break
            self.diag["snap_redraw"] += int(bad.sum())
            new[bad] = B[k[bad]] + r.normal(0, ESC_SNAP_SD, (int(bad.sum()), 2))
            bad = ~g.free(new)
        new[bad] = B[k[bad]]
        pf.xy = new
        pf.map = g

    def estimate(self):
        e = self.pf.estimate()
        w = np.exp(self.pf.logw)
        xy = self.pf.xy
        rms = float(np.sqrt(w @ ((xy - e["xy"]) ** 2).sum(1)))
        en = aligned_to_enu(e["xy"], self.rot)[0]
        modes = top_modes(xy, w)
        m = [(aligned_to_enu(c, self.rot)[0], q) for c, q in modes] + [(np.full(2, np.nan), 0.0)] * (2 - len(modes))
        return dict(x=float(en[0]), y=float(en[1]), compass=float(th_to_compass(e["th"], self.rot)), unc=rms,
                    unc95=e["unc95"], k=e["k"], k_std=e["k_std"], ess=e["ess"], pdr_suspect=e["pdr_suspect"],
                    floor=self.floor, m1=m[0][0], m1_mass=m[0][1], m2=m[1][0], m2_mass=m[1][1])


def pf_params(n, config=paths.PF_CONFIG, **over):
    import tomllib
    from indoornav.pf.filter import PFParams
    d = tomllib.loads(config.read_text(encoding="utf-8"))   # scripts/tune_pf.params_from_toml과 같음(스크립트라 직접 읽음)
    return replace(PFParams(**d), n=n, **over)
