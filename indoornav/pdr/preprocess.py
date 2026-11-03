"""전처리: 기기 좌표계 가속도·자이로 → 수직 성분 + 저역통과 필터."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfilt

from .params import PreprocessParams


def up_unit(gravity: np.ndarray) -> np.ndarray:
    """위쪽 단위벡터 [N,3]. iOS 중력 벡터는 지구 방향이므로 부호를 뒤집는다."""
    g = np.asarray(gravity, float)
    return -g / np.linalg.norm(g, axis=1, keepdims=True)


def vertical_acc(acc_user: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """중력 제거 가속도의 수직(위쪽 +) 성분 [m/s²]. 폰 기울기와 무관."""
    return np.einsum("ij,ij->i", acc_user, up_unit(gravity))


def yaw_rate(gyro: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """수직축 회전 속도 [rad/s]. 위에서 내려다볼 때 반시계(좌회전) +."""
    return np.einsum("ij,ij->i", gyro, up_unit(gravity))


def lowpass(x: np.ndarray, fs: float, p: PreprocessParams) -> np.ndarray:
    if p.mode == "causal":
        sos = butter(p.lpf_order, p.lpf_cutoff_hz, btype="low", fs=fs, output="sos")
        return sosfilt(sos, x)
    b, a = butter(p.lpf_order, p.lpf_cutoff_hz, btype="low", fs=fs)
    if p.mode == "offline":
        return filtfilt(b, a, x)
    if p.mode == "delayed":
        return delayed_filtfilt(x, fs, b, a, p.delay_s, p.window_s, p.hop_s)
    raise ValueError(f"알 수 없는 필터 방식: {p.mode}")


def delayed_filtfilt(x: np.ndarray, fs: float, b, a, delay_s: float, window_s: float,
                     hop_s: float) -> np.ndarray:
    """앱에서 실시간으로 돌릴 때와 같은 결과를 오프라인으로 재현한다.

    hop_s 마다 '지금'이 한 칸씩 이동하고, 그때마다
      1) 지금까지 들어온 최근 window_s 구간만 꺼내서 앞뒤 필터(filtfilt)를 적용하고
      2) 그중 [지금 − delay_s − hop_s, 지금 − delay_s) 구간만 확정 출력한다.
    미래 데이터는 delay_s 만큼만 보므로, 출력은 항상 delay_s 늦게 나온다.
    """
    n = len(x)
    d, w, h = int(round(delay_s * fs)), int(round(window_s * fs)), max(1, int(round(hop_s * fs)))
    pad = 3 * max(len(a), len(b))                      # filtfilt 가 요구하는 최소 길이
    y = np.empty(n)
    for now in range(h, n + d + h, h):                 # now: 지금까지 들어온 샘플 수
        end = min(now, n)
        out0, out1 = max(0, now - d - h), min(n, now - d)
        if out1 <= out0:
            continue
        start = max(0, end - w)
        seg = x[start:end]
        if len(seg) <= pad:
            y[out0:out1] = seg[out0 - start:out1 - start]
            continue
        f = filtfilt(b, a, seg)
        y[out0:out1] = f[out0 - start:out1 - start]
    return y
