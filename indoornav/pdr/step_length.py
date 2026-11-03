"""보폭 추정.

Weinberg 모델 (Weinberg, 2002, Analog Devices AN-602):
    L = K * (a_max - a_min) ** 지수
  a_max, a_min: 한 걸음 구간의 수직 가속도 최대·최소.
  K: 사람(·장착 위치)마다 다른 계수 → calibrate_k()로 직선 데이터에서 보정.

Weinberg + 걸음 빈도 (model="weinberg_freq"):
    L = K * (a_max - a_min) ** 지수 + k_freq * f + offset
  f: 걸음 빈도 [걸음/s] = 1 / (직전 걸음과의 시간 간격).
  빨리 걸을수록 보폭이 커지는 경향을 출렁임 크기와 별도로 반영한다 → calibrate_freq()로 보정.
"""

from __future__ import annotations

import numpy as np

from .params import LengthParams


def weinberg_feature(sig: np.ndarray, windows: list[tuple[int, int]], exponent: float) -> np.ndarray:
    """걸음별 (a_max - a_min)^exponent."""
    out = np.empty(len(windows))
    for i, (a, b) in enumerate(windows):
        seg = sig[a:b]
        out[i] = (seg.max() - seg.min()) ** exponent if len(seg) else 0.0
    return out


def step_frequency(steps: np.ndarray, fs: float, max_interval_s: float = 2.0) -> np.ndarray:
    """걸음별 빈도 [걸음/s]. 첫 걸음과 긴 공백 뒤 걸음은 전체 중앙값으로 채운다."""
    if len(steps) == 0:
        return np.array([])
    dt = np.diff(steps) / fs
    ok = dt <= max_interval_s
    typical = float(np.median(dt[ok])) if ok.any() else 0.6
    full = np.concatenate([[typical], np.where(ok, dt, typical)])
    return 1.0 / full


def step_lengths(sig: np.ndarray, windows: list[tuple[int, int]], p: LengthParams,
                 freq: np.ndarray | None = None, slow: np.ndarray | None = None) -> np.ndarray:
    """slow: 걸음별 느린 걸음 모드 여부 (step.method="adaptive"). k_slow > 0 이면 그 걸음은 k_slow 사용."""
    k = np.full(len(windows), p.k)
    if slow is not None and p.k_slow > 0 and len(slow) == len(windows):
        k = np.where(slow, p.k_slow, p.k)
    if p.model == "constant":
        L = np.full(len(windows), p.constant_m)
    elif p.model == "weinberg":
        L = k * weinberg_feature(sig, windows, p.exponent)
    elif p.model == "weinberg_freq":
        if freq is None:
            raise ValueError("weinberg_freq 모델에는 걸음 빈도(freq)가 필요")
        L = k * weinberg_feature(sig, windows, p.exponent) + p.k_freq * freq + p.offset_m
    else:
        raise ValueError(f"알 수 없는 보폭 모델: {p.model}")
    return np.clip(L, p.min_m, p.max_m)


def calibrate_k(features_per_session: list[np.ndarray], distances_m: list[float]) -> float:
    """여러 세션의 (걸음별 feature, 실제 거리)로 K를 최소제곱 추정.

    거리_i ≈ K * sum(feature_i)  →  K = Σ(d_i·s_i) / Σ(s_i²)
    """
    s = np.array([f.sum() for f in features_per_session])
    d = np.array(distances_m, float)
    return float((d @ s) / (s @ s))


def calibrate_freq(features_per_session: list[np.ndarray], freqs_per_session: list[np.ndarray],
                   distances_m: list[float]) -> tuple[float, float, float]:
    """거리_i ≈ K·Σfeature + k_freq·Σf + offset·n 을 최소제곱으로 풀어 (K, k_freq, offset) 반환.

    세션이 3개 미만이면 풀 수 없으므로 Weinberg 만 맞춘다.
    """
    A = np.array([[f.sum(), q.sum(), len(f)] for f, q in zip(features_per_session, freqs_per_session)])
    d = np.array(distances_m, float)
    if len(d) < 3:
        return calibrate_k(features_per_session, distances_m), 0.0, 0.0
    k, kf, c = np.linalg.lstsq(A, d, rcond=None)[0]
    return float(k), float(kf), float(c)


def calibrate_k_mode(features_per_session: list[np.ndarray], slow_per_session: list[np.ndarray],
                     distances_m: list[float]) -> tuple[float, float]:
    """보통 모드 K 와 느린 걸음 모드 K_slow 를 함께 최소제곱으로 추정.

    거리_i ≈ K·Σ(보통 걸음 feature) + K_slow·Σ(느린 걸음 feature)
    느린 걸음이 하나도 없으면 K_slow = 0 (사용 안 함).
    """
    A = np.array([[f[~s].sum(), f[s].sum()] for f, s in zip(features_per_session, slow_per_session)])
    d = np.array(distances_m, float)
    if A[:, 1].sum() == 0 or A[:, 0].sum() == 0:
        return calibrate_k(features_per_session, distances_m), 0.0
    k, ks = np.linalg.lstsq(A, d, rcond=None)[0]
    return float(k), float(ks)


def bout_mask(steps: np.ndarray, fs: float, min_bout_steps: int, bout_gap_s: float) -> np.ndarray:
    """걸음별로 '이어서 걷는 묶음'에 속하면 True.

    걸음 간격이 bout_gap_s 이하인 걸음끼리 한 묶음으로 보고, 묶음 크기가 min_bout_steps 미만이면 False.
    제자리 회전(발을 뺐다 되돌리는 동작 2번 후 몇 초 정지), 자리 고쳐 서기 같은 움직임은
    걸음으로는 잡히지만 앞으로 이동하지 않으므로 보폭을 0 으로 만들 때 쓴다.
    """
    n = len(steps)
    keep = np.ones(n, bool)
    if min_bout_steps <= 1 or n == 0:
        return keep
    gap = np.diff(steps) / fs
    start = 0
    for i in range(1, n + 1):
        if i == n or gap[i - 1] > bout_gap_s:
            if i - start < min_bout_steps:
                keep[start:i] = False
            start = i
    return keep
