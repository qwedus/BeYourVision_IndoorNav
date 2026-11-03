"""헤딩(진행 방향) 추정.

조끼 가슴 고정 → 폰 방향 = 몸 방향 = 진행 방향 으로 가정.
출발 방향을 0 rad 으로 두는 상대 헤딩. 반시계(좌회전) +.

source="gyro"       : 수직축 회전 속도(gyro · up)를 직접 적분. 자력계 영향 없음.
source="orientation": iPhone 자세 yaw. 앱의 기준 좌표계 설정에 따라 자력계가 섞일 수 있음.
zaru=True           : 정지 구간의 평균 회전 속도를 바이어스로 보고 빼서 드리프트 감소.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .params import HeadingParams
from .preprocess import yaw_rate


def still_mask(acc_user: np.ndarray, gyro: np.ndarray, fs: float, p: HeadingParams) -> np.ndarray:
    """정지 샘플 True. 창 내 가속도·자이로 크기의 표준편차가 모두 작으면 정지."""
    w = max(3, int(p.still_window_s * fs))

    def rolling_std(x):
        m = uniform_filter1d(x, w, mode="nearest")
        m2 = uniform_filter1d(x * x, w, mode="nearest")
        return np.sqrt(np.maximum(m2 - m * m, 0.0))

    a = rolling_std(np.linalg.norm(acc_user, axis=1))
    g = rolling_std(np.linalg.norm(gyro, axis=1))
    return (a < p.still_acc_std) & (g < p.still_gyro_std)


def gyro_bias_track(rate: np.ndarray, still: np.ndarray) -> np.ndarray:
    """정지 구간마다 평균 회전 속도를 바이어스로 측정하고, 구간 사이는 직전 값을 유지."""
    bias = np.zeros_like(rate)
    cur = 0.0
    i, n = 0, len(rate)
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            cur = float(rate[i:j].mean())
            bias[i:j] = cur
            i = j
        else:
            bias[i] = cur
            i += 1
    return bias


def heading_series(session, p: HeadingParams, fs: float) -> np.ndarray:
    """샘플별 헤딩 [rad], 출발 시 0."""
    if p.source == "orientation":
        if session.yaw is None:
            raise ValueError("Orientation.csv 의 yaw 가 없음")
        h = session.yaw - session.yaw[0]
        return h
    if p.source != "gyro":
        raise ValueError(f"알 수 없는 헤딩 소스: {p.source}")

    rate = yaw_rate(session.gyro, session.gravity)
    if p.zaru:
        rate = rate - gyro_bias_track(rate, still_mask(session.acc_user, session.gyro, fs, p))
    h = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) / fs)])
    return h


def step_headings(h: np.ndarray, steps: np.ndarray, windows: list[tuple[int, int]],
                  p: HeadingParams) -> np.ndarray:
    """걸음별 헤딩. step_average면 걸음 구간 원형 평균, 아니면 걸음 순간 값."""
    if not p.step_average:
        return h[steps]
    out = np.empty(len(windows))
    for i, (a, b) in enumerate(windows):
        seg = h[a:b]
        out[i] = np.arctan2(np.sin(seg).mean(), np.cos(seg).mean())
    # 원형 평균은 (-π, π] 로 접히므로 누적 회전을 보존하도록 원래 값 근처로 펼친다
    ref = h[steps]
    return ref + (out - ref + np.pi) % (2 * np.pi) - np.pi
