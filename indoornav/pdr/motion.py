"""움직임 구간 판별: 걷기 시작·종료 시점.

가속도 크기의 이동 표준편차(흔들림 정도)가 기준값을 넘는 구간을 '움직임'으로 본다.
  - 서 있을 때(조끼 장착, 호흡·미세 흔들림): 약 0.03~0.05 m/s²
  - 매우 느린 걸음: 약 0.15~0.2 m/s²
  - 보통 걸음: 약 0.4~0.5 m/s²
(2026-10-11 측정 데이터 기준. 서 있는 구간이 짧아 정지 쪽 값은 더 측정이 필요)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class WalkBounds:
    start_s: float          # 걷기 시작 시각 [s]
    end_s: float            # 걷기 종료 시각 [s]
    lead_still_s: float     # 시작 전 정지 구간 길이 [s]
    tail_still_s: float     # 종료 후 정지 구간 길이 [s]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def motion_energy(acc_user: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """가속도 크기의 이동 표준편차 [m/s²]."""
    a = np.linalg.norm(acc_user, axis=1)
    w = max(3, int(window_s * fs))
    m = uniform_filter1d(a, w, mode="nearest")
    m2 = uniform_filter1d(a * a, w, mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def walking_bounds(session, threshold: float = 0.08, window_s: float = 1.0,
                   min_still_s: float = 1.0) -> WalkBounds:
    """처음과 마지막의 정지 구간(min_still_s 이상 연속)을 잘라낸 걷기 구간.

    정지 구간이 없으면 녹화 전체를 걷기로 본다.
    """
    e = motion_energy(session.acc_user, session.fs, window_s)
    t = session.t
    still = e < threshold
    n_min = int(min_still_s * session.fs)

    i = 0
    while i < len(still) and still[i]:
        i += 1
    start = i if i >= n_min else 0

    j = len(still)
    while j > 0 and still[j - 1]:
        j -= 1
    end = j if len(still) - j >= n_min else len(still)

    if end <= start:            # 전부 정지
        start, end = 0, len(t)
    return WalkBounds(
        start_s=float(t[start]),
        end_s=float(t[end - 1]),
        lead_still_s=float(t[start] - t[0]),
        tail_still_s=float(t[-1] - t[end - 1]),
    )
