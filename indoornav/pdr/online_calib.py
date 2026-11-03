"""보폭 계수 K 온라인 보정.

앱을 쓰는 동안, 랜드마크로 위치가 확정된 두 시점 사이의 '실제 이동 거리'(서버/파티클 필터가 알려줌)와
같은 구간의 'PDR 이동 거리'를 비교해 K 를 그 사람에게 맞게 조금씩 고쳐 나간다.
(상세설계서: "보폭은 Weinberg 모델로 추정하며, 수렴 후에는 서버가 추정한 이동거리로 계수를 보정한다")

    PDR 거리 = K · Σ feature     (feature = (a_max − a_min)^지수, 걸음마다)
    비율     = 실제 거리 / PDR 거리
    K_new    = (1 − α)·K + α·(K·비율)
    α        = max(alpha, 1 / (반영 횟수 + 2))   → 처음엔 크게(0.5, 0.33, …), 이후 alpha 로 고정

반영 조건 (하나라도 어기면 무시):
  - 구간 거리 ≥ min_dist_m            : 짧으면 위치 오차가 상대적으로 커서
  - 위치 불확실도 ≤ max_unc_m         : 서버가 확신할 때만
  - 구간 안 걸음 ≥ min_steps
  - 구간 안 방향 변화 ≤ max_turn_deg  : 서버가 주는 건 두 점 사이 직선거리라서, 꺾인 구간이면 실제 걸은 거리보다 짧다
  - 비율이 [1/max_ratio, max_ratio] 안 : 랜드마크 오인식 같은 이상값 차단
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CalibParams:
    alpha: float = 0.2            # 한 번에 반영하는 비율의 하한 (처음 몇 번은 더 크게)
    min_dist_m: float = 5.0
    max_unc_m: float = 1.0
    min_steps: int = 6
    max_turn_deg: float = 20.0
    max_ratio: float = 2.0        # 사람이 바뀌면 K 가 ±50% 다를 수 있어 여유를 둠
    k_min: float = 0.15
    k_max: float = 0.80
    history_s: float = 180.0      # 걸음 기록 보관 시간


@dataclass
class CalibResult:
    applied: bool
    reason: str
    k_before: float
    k_after: float
    ratio: float = float("nan")
    pdr_dist_m: float = float("nan")


@dataclass
class StrideCalibrator:
    k: float
    p: CalibParams = field(default_factory=CalibParams)
    _hist: deque = field(default_factory=deque, repr=False)   # (t, feature, heading)
    log: list = field(default_factory=list, repr=False)
    n_applied: int = 0

    def add_step(self, t: float, feature: float, heading: float) -> float:
        """걸음 하나를 기록하고, 현재 K 로 계산한 보폭을 돌려준다."""
        self._hist.append((t, feature, heading))
        while self._hist and t - self._hist[0][0] > self.p.history_s:
            self._hist.popleft()
        return self.k * feature

    def correct(self, t0: float, t1: float, distance_m: float, uncertainty_m: float) -> CalibResult:
        """서버가 '(t0, t1] 동안 distance_m 이동' 이라고 알려줄 때 호출."""
        seg = [(t, f, h) for t, f, h in self._hist if t0 < t <= t1]

        def reject(reason: str, **kw) -> CalibResult:
            r = CalibResult(False, reason, self.k, self.k, **kw)
            self.log.append(r)
            return r

        if uncertainty_m > self.p.max_unc_m:
            return reject("불확실도 큼")
        if distance_m < self.p.min_dist_m:
            return reject("구간 짧음")
        if len(seg) < self.p.min_steps:
            return reject("걸음 부족")
        h = np.unwrap([x[2] for x in seg])
        if np.degrees(h.max() - h.min()) > self.p.max_turn_deg:
            return reject("구간에 회전 있음")
        fsum = float(sum(x[1] for x in seg))
        pdr = self.k * fsum
        ratio = distance_m / pdr if pdr > 0 else float("inf")
        if not (1 / self.p.max_ratio <= ratio <= self.p.max_ratio):
            return reject("비율 이상", ratio=ratio, pdr_dist_m=pdr)
        k_old = self.k
        a = max(self.p.alpha, 1.0 / (self.n_applied + 2))
        self.k = float(np.clip((1 - a) * self.k + a * self.k * ratio, self.p.k_min, self.p.k_max))
        self.n_applied += 1
        r = CalibResult(True, "반영", k_old, self.k, ratio, pdr)
        self.log.append(r)
        return r
