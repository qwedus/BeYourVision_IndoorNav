"""PDR 전체 파이프라인: 걸음 검출 → 보폭 → 헤딩 → 위치 누적."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .heading import heading_series, step_headings
from .params import PDRParams
from .preprocess import lowpass, vertical_acc
from .step_adaptive import detect_steps_adaptive
from .step_detect import detect_steps, step_windows
from .step_length import bout_mask, step_frequency, step_lengths, weinberg_feature


@dataclass
class PDRResult:
    steps: pd.DataFrame        # 걸음별: t, idx, length, heading, dheading, x, y, feature, freq, slow(느린 걸음 모드), moving(이동하는 걸음)
    signal: np.ndarray         # 걸음 검출에 쓴 필터된 신호 (그래프용)
    heading: np.ndarray        # 샘플별 헤딩 [rad]
    params: PDRParams

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def distance(self) -> float:
        return float(self.steps["length"].sum())

    @property
    def end_xy(self) -> np.ndarray:
        if self.steps.empty:
            return np.zeros(2)
        return self.steps[["x", "y"]].iloc[-1].to_numpy()

    @property
    def net_turn_deg(self) -> float:
        """출발~종료 전체 헤딩 변화 [deg]."""
        return float(np.degrees(self.heading[-1] - self.heading[0]))

    def events(self) -> pd.DataFrame:
        """파티클 필터로 넘길 형식(초안): 걸음마다 시각·보폭·헤딩 변화량."""
        return self.steps[["t", "length", "dheading"]].copy()


def walk_signal(session, params: PDRParams) -> np.ndarray:
    sp = params.step
    if sp.signal == "vertical":
        raw = vertical_acc(session.acc_user, session.gravity)
    elif sp.signal == "norm":
        raw = np.linalg.norm(session.acc_user, axis=1)
        raw = raw - raw.mean()
    else:
        raise ValueError(f"알 수 없는 걸음 신호: {sp.signal}")
    return lowpass(raw, params.fs, params.preprocess)


def run_pdr(session, params: PDRParams | None = None) -> PDRResult:
    params = params or PDRParams()
    fs = params.fs
    sig = walk_signal(session, params)

    if params.step.method == "adaptive":
        steps, trace = detect_steps_adaptive(session, fs, params.step)
        slow = trace.slow_mode[steps] if len(steps) else np.array([], bool)
    elif params.step.method == "peak":
        steps = detect_steps(sig, fs, params.step)
        slow = np.zeros(len(steps), bool)
    else:
        raise ValueError(f"알 수 없는 걸음 검출 방식: {params.step.method}")
    win = step_windows(steps, fs, params.step, len(sig))
    freq = step_frequency(steps, fs, params.step.max_interval_s)
    L = step_lengths(sig, win, params.length, freq, slow)
    moving = bout_mask(steps, fs, params.length.min_bout_steps, params.length.bout_gap_s)
    L = np.where(moving, L, 0.0)
    feat = weinberg_feature(sig, win, params.length.exponent) if len(win) else np.array([])

    h = heading_series(session, params.heading, fs)
    hs = step_headings(h, steps, win, params.heading) if len(steps) else np.array([])

    x = np.cumsum(L * np.cos(hs)) if len(L) else np.array([])
    y = np.cumsum(L * np.sin(hs)) if len(L) else np.array([])
    dh = np.diff(np.concatenate([[0.0], hs])) if len(hs) else np.array([])

    df = pd.DataFrame({
        "t": session.t[steps] if len(steps) else [],
        "idx": steps, "length": L, "heading": hs, "dheading": dh,
        "x": x, "y": y, "feature": feat, "freq": freq, "slow": slow, "moving": moving,
    })
    return PDRResult(steps=df, signal=sig, heading=h, params=params)
