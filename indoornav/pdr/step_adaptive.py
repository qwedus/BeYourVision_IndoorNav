"""속도 적응형 걸음 검출 (step.method = "adaptive").

배경 (docs/tuning_log.md 2026-10-16 분석):
  - 보통·빠른 걸음: 몸이 걸음마다 위아래로 출렁인다 → 수직 가속도의 리듬 = 걸음.
  - 매우 느린 걸음: 위아래 출렁임이 거의 없고, 수직 가속도에는 한 걸음에 봉우리가 두 개 생긴다.
    대신 가속도 크기(전체 흔들림)의 리듬이 걸음과 맞는다.

동작:
  1) win_s 길이 창을 hop_s 마다 옮기며, 창마다 수직 흔들림 크기(표준편차)를 잰다.
     slow_energy 보다 작으면 '느린 걸음 모드', 크면 '보통 모드'. 이웃 창 다수결로 모드를 다듬는다.
  2) 모드에 맞는 신호에서 자기상관으로 걸음 주기 T 를 구한다. 주기성이 약하거나(min_quality 미만)
     흔들림 자체가 작으면(min_energy 미만) 걷지 않는 구간으로 보고 걸음을 내지 않는다
     → 서 있을 때 가짜 걸음도 막는 효과.
  3) 모드별 신호를 최근 흔들림 크기로 나눠 단위를 맞춘 뒤 이어 붙이고(합성 신호), 봉우리를 찾는다.
  4) 직전 걸음과 0.6·T 보다 가까운 봉우리는 같은 걸음으로 보고 큰 쪽만 남긴다.

제약: 현재는 오프라인(녹화 전체) 필터만 지원한다. 앱용 0.5초 지연 방식은 이후 구현.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

from .params import StepParams
from .preprocess import vertical_acc


@dataclass
class AdaptiveTrace:
    """그래프·분석용 중간 결과."""
    signal: np.ndarray      # 봉우리를 찾은 합성 신호 (단위 없음)
    slow_mode: np.ndarray   # 샘플별 느린 걸음 모드 여부
    period_s: np.ndarray    # 샘플별 추정 걸음 주기 [s]
    walking: np.ndarray     # 샘플별 주기성 있음(걷는 중) 여부


def _band(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    b, a = butter(4, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, x)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    m = uniform_filter1d(x, w, mode="nearest")
    m2 = uniform_filter1d(x * x, w, mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 1e-12))


def _acf_period(seg: np.ndarray, fs: float, fmin: float = 0.4, fmax: float = 2.6) -> tuple[float, float]:
    seg = seg - seg.mean()
    n = len(seg)
    ac = np.correlate(seg, seg, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fs / fmax), min(int(fs / fmin), n - 1)
    r = ac[lo:hi]
    k = int(np.argmax(r))
    return (k + lo) / fs, float(r[k])


def detect_steps_adaptive(session, fs: float, p: StepParams) -> tuple[np.ndarray, AdaptiveTrace]:
    n = len(session.t)
    v = _band(vertical_acc(session.acc_user, session.gravity), fs, 0.3, 3.0)
    a = _band(np.linalg.norm(session.acc_user, axis=1), fs, 0.3, 3.0)
    a_s = _band(a, fs, 0.3, 1.5)                         # 느린 모드용: 더 부드럽게

    W, H = int(p.win_s * fs), max(1, int(p.hop_s * fs))
    starts = list(range(0, max(1, n - W + 1), H))
    slow, per, qual, moving = [], [], [], []
    for st in starts:
        sl = slice(st, min(n, st + W))
        e_v, e_a = float(v[sl].std()), float(a[sl].std())
        s_mode = e_v < p.slow_energy
        T, q = _acf_period((a if s_mode else v)[sl], fs)
        slow.append(s_mode); per.append(T); qual.append(q)
        moving.append(max(e_v, e_a) >= p.min_energy)
    slow = np.array(slow, float)
    k = p.mode_smooth                                   # 이웃 창 다수결
    slow = np.array([round(slow[max(0, i - k):i + k + 1].mean()) for i in range(len(slow))], bool)

    # 창 → 샘플: 각 샘플은 중심이 가장 가까운 창의 값을 쓴다
    centers = np.array(starts) + min(W, n) // 2
    nearest = np.clip(np.searchsorted(centers, np.arange(n)), 0, len(centers) - 1)
    left = np.clip(nearest - 1, 0, len(centers) - 1)
    nearest = np.where(np.abs(np.arange(n) - centers[left]) < np.abs(np.arange(n) - centers[nearest]),
                       left, nearest)
    slow_s = slow[nearest]
    per_s = np.array(per)[nearest]
    walk_s = (np.array(qual)[nearest] >= p.min_quality) & np.array(moving)[nearest]

    w3 = int(3 * fs)
    zv = v / np.maximum(_rolling_std(v, w3), p.min_energy)      # 하한: 정지 중 미세 떨림을 키우지 않도록
    za = a_s / np.maximum(_rolling_std(a_s, w3), p.min_energy)
    c = np.where(slow_s, za, zv)

    cand, props = find_peaks(c, prominence=p.adaptive_prom, distance=max(1, int(0.25 * fs)))
    prom = props["prominences"]
    steps: list[int] = []
    best: list[float] = []
    for i, pr in zip(cand, prom):
        if not walk_s[i]:
            continue
        if steps and i - steps[-1] < 0.6 * per_s[i] * fs:
            if pr > best[-1]:                           # 같은 걸음 → 더 뚜렷한 봉우리로 교체
                steps[-1], best[-1] = i, pr
            continue
        steps.append(i); best.append(pr)
    return np.array(steps, int), AdaptiveTrace(c, slow_s, per_s, walk_s)
