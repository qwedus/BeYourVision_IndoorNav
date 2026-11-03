"""PF 결과 그림: 지도 위 PF·PDR 궤적과 체크포인트 오차."""

from __future__ import annotations

import numpy as np


def plot_replay(ax, wm, res, cps=None, title: str | None = None):
    for a, b in zip(wm.a, wm.b):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.5)
    if res.pdr_track is not None:
        ax.plot(res.pdr_track[:, 0], res.pdr_track[:, 1], color="tab:gray", lw=1, alpha=0.8, label="PDR 단독")
    if res.track is not None:
        ax.plot(res.track[:, 0], res.track[:, 1], color="tab:red", lw=1.2, label="PF 추정")
    for r in res.rows:
        name = {"SE_END": "SE", "START_END": "START", "NE_back": "NE"}.get(r["checkpoint"], r["checkpoint"])
        if r["checkpoint"] == "START_END" and r.get("exp") == "L4":
            name = "START_W"
        xy = wm.checkpoints[name]
        mk = "s" if r["observed"] else "*"
        ax.plot(*xy, mk, color="tab:blue" if r["observed"] else "goldenrod", ms=9, mec="black", mew=0.5)
        ax.annotate(f"{r['checkpoint']} {r['err_pf']:.1f}m", xy, fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_aspect("equal")
    ax.set_title(title or res.session_id, fontsize=9)
    ax.grid(alpha=0.2)


def set_korean_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            return
