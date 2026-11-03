"""확인용 그래프."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Windows 한글 폰트 (없으면 기본 폰트 유지)
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_steps(session, result, ax=None, title: str | None = None):
    """필터된 걸음 신호 + 검출된 걸음 + Annotation 시각."""
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    ax.plot(session.t, result.signal, lw=0.8, label="수직 가속도 (LPF)")
    idx = result.steps["idx"].to_numpy(int)
    ax.plot(session.t[idx], result.signal[idx], "rv", ms=4, label=f"걸음 {len(idx)}")
    ax.axhline(result.params.step.min_height, color="gray", ls="--", lw=0.7, label="min_height")
    if not session.annotations.empty:
        for t in session.annotations["seconds_elapsed"]:
            ax.axvline(t, color="green", lw=0.8, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("m/s²")
    ax.set_title(title or session.name); ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_heading(session, result, ax=None):
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    ax.plot(session.t, np.degrees(result.heading), lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("heading [deg]")
    ax.set_title(f"{session.name}  net {result.net_turn_deg:+.1f}°")
    return ax


def plot_trajectory(result, ax=None, title: str | None = None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    x = np.concatenate([[0], result.steps["x"]])
    y = np.concatenate([[0], result.steps["y"]])
    ax.plot(x, y, "-o", ms=2, lw=1)
    ax.plot(0, 0, "gs", label="출발"); ax.plot(x[-1], y[-1], "r^", label="도착")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title or f"{result.distance:.1f} m, {result.n_steps} steps")
    return ax
