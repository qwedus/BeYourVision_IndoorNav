"""걸음 검출: 필터된 수직 가속도의 봉우리 = 한 걸음.

기준 알고리즘 (교과서적 peak detection):
  1) 저역통과로 자잘한 떨림 제거
  2) 높이·돌출 기준을 넘는 봉우리만 채택
  3) 최소 간격보다 가까운 봉우리는 하나로 취급
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .params import StepParams


def detect_steps(sig: np.ndarray, fs: float, p: StepParams) -> np.ndarray:
    """걸음 순간의 샘플 인덱스 배열을 반환."""
    peaks, _ = find_peaks(
        sig,
        height=p.min_height,
        prominence=p.min_prominence,
        distance=max(1, int(round(p.min_interval_s * fs))),
    )
    return peaks


def step_windows(steps: np.ndarray, fs: float, p: StepParams, n: int) -> list[tuple[int, int]]:
    """각 걸음이 차지하는 [시작, 끝) 구간. 이전 걸음 ~ 현재 걸음.

    첫 걸음이거나 직전 걸음과 max_interval_s 이상 떨어지면 길이를 평균 간격으로 잡는다.
    """
    if len(steps) == 0:
        return []
    gaps = np.diff(steps)
    typical = int(np.median(gaps)) if len(gaps) else int(0.6 * fs)
    max_gap = int(p.max_interval_s * fs)
    win = []
    for i, s in enumerate(steps):
        if i == 0 or steps[i] - steps[i - 1] > max_gap:
            a = max(0, s - typical)
        else:
            a = steps[i - 1]
        win.append((a, min(n, s + 1)))
    return win
