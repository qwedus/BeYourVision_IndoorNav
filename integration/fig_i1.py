"""I1 indoornav PF 재생 그림: 시간별 위치 오차(비전 끔/켬, PDR만)와 B1·B2 경로 (seed 0).

    .venv/Scripts/python integration/fig_i1.py [--name inav_pdr_pf] [--set 중간] [--obs tf]
→ <녹화 설정 out>/<name>/fig_error.png, fig_map.png, fig_paths.png
"""
import argparse
import json

import matplotlib
import numpy as np
from PIL import Image

import paths
import replay as rp
from indoornav.pf.plots import set_korean_font
from station_pkg import enu_to_aligned
from vision_filter import load_maps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="inav_pdr_pf")
    ap.add_argument("--set", default="중간")
    ap.add_argument("--obs", default="tf")
    a = ap.parse_args()
    set_korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    out = paths.REC_OUT / a.name
    z = np.load(out / "runs.npz")
    truth = rp.load_truth(paths.I1 / "truth.csv")
    rot = json.loads(paths.MASK_META.read_text(encoding="utf-8"))["aligned_from_enu_rot_deg"]
    n = sum(k.startswith("known_none_") for k in z.files)
    runs = {sc: dict(off=[z[f"{sc}_none_{i}"] for i in range(n)], on=[z[f"{sc}_{a.obs}_{a.set}_{i}"] for i in range(n)],
                     dr=z[f"{sc}_pdr_only_0"]) for sc in ("known", "drift")}
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, sc in zip(axs, runs):
        R = runs[sc]
        t = R["on"][0]["t"]
        for lab, rs, c in (("PF 비전 끔", R["off"], "tab:gray"), (f"PF + 비전 ({a.obs}·{a.set})", R["on"], "tab:blue")):
            e = np.array([r["err"] for r in rs])
            ax.fill_between(t, np.percentile(e, 10, 0), np.percentile(e, 90, 0), color=c, alpha=0.2)
            ax.plot(t, np.median(e, 0), c=c, label=lab + " (seed 10개 중앙, 10~90%)")
        dr = R["dr"]
        ex, ey = np.interp(truth["t"], dr[:, 0], dr[:, 1]), np.interp(truth["t"], dr[:, 0], dr[:, 2])
        ax.plot(truth["t"], np.hypot(ex - truth["east"], ey - truth["north"]), "--", c="k", lw=1, label="PDR만")
        for ia, (ea, eb) in enumerate(paths.RC["exclude"]):
            ax.axvspan(ea, eb, color="orange", alpha=0.12, label="층 이동(지표 제외)" if ia == 0 else None)
        ax.set_ylabel("위치 오차 (m)")
        ax.set_title({"known": "출발점 앎", "drift": "출발점 앎 · PDR 틀어짐(보폭 ×1.15, 0.4°/걸음)"}[sc] + f" — {a.name}")
        ax.set_ylim(0, 12)
        ax.legend(fontsize=8, loc="upper left")
    axs[-1].set_xlabel("녹화 시작부터 (s)")
    fig.tight_layout()
    fig.savefig(out / "fig_error.png", dpi=110)
    plt.close(fig)

    maps = load_maps()
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, (fl, name) in zip(axs, paths.FLOOR_NAME.items()):
        g = maps[fl]
        h, w = g.walk.shape
        x0, y1 = g.to_world(0, 0)
        x1, y0 = g.to_world(w, h)
        ax.imshow(np.array(Image.fromarray((g.walk * 255).astype(np.uint8))), extent=[x0, x1, y0, y1], cmap="gray",
                  alpha=0.35)
        sel = (truth["floor"] == fl) & (truth["on_esc"] == 0)
        if not sel.any():
            ax.set_axis_off()
            continue
        ta = enu_to_aligned(np.c_[truth["east"][sel], truth["north"][sel]], rot)
        ax.plot(ta[:, 0], ta[:, 1], c="red", lw=2, label="정답")
        for sc, c in (("known", "tab:blue"), ("drift", "tab:purple")):
            r = runs[sc]["on"][0]
            s = (r["efloor"] == fl) & ~r["riding"]
            p = enu_to_aligned(np.c_[r["ex"][s], r["ey"][s]], rot)
            ax.plot(p[:, 0], p[:, 1], ".", ms=2, c=c, label=f"{sc} PF + 비전 (seed 0)")
            dr = runs[sc]["dr"]
            q = enu_to_aligned(dr[dr[:, 3] == fl][:, 1:3], rot)
            ax.plot(q[:, 0], q[:, 1], "--", c=c, lw=1, alpha=0.7, label=f"{sc} PDR만")
        lo, hi = ta.min(0) - 12, ta.max(0) + 12
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(f"{name} (지도 aligned 좌표, 방 뺀 마스크)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_map.png", dpi=100)
    plt.close(fig)
    # 경로 비교: 조건마다 정답 · PDR만 · PF 비전 끔 · PF + 비전 (seed 0)
    fls = [fl for fl in paths.FLOOR_NAME if ((truth["floor"] == fl) & (truth["on_esc"] == 0)).any()]
    fig, axs = plt.subplots(2, len(fls), figsize=(8 * len(fls), 12), squeeze=False)
    for i, sc in enumerate(("known", "drift")):
        for j, fl in enumerate(fls):
            ax, g = axs[i, j], maps[fl]
            h, w = g.walk.shape
            x0, y1 = g.to_world(0, 0)
            x1, y0 = g.to_world(w, h)
            ax.imshow((g.walk * 255).astype(np.uint8), extent=[x0, x1, y0, y1], cmap="gray", alpha=0.35)
            sel = (truth["floor"] == fl) & (truth["on_esc"] == 0)
            ta = enu_to_aligned(np.c_[truth["east"][sel], truth["north"][sel]], rot)
            ax.plot(ta[:, 0], ta[:, 1], c="red", lw=2.5, label="정답")
            q = enu_to_aligned(runs[sc]["dr"][runs[sc]["dr"][:, 3] == fl][:, 1:3], rot)
            ax.plot(q[:, 0], q[:, 1], "--", c="k", lw=1, label="PDR만")
            for key_, lab, c in (("off", "PF 비전 끔", "tab:gray"), ("on", f"PF + 비전 ({a.obs}·{a.set})", "tab:blue")):
                r = runs[sc][key_][0]
                s = (r["efloor"] == fl) & ~r["riding"]
                p = enu_to_aligned(np.c_[r["ex"][s], r["ey"][s]], rot)
                ax.plot(p[:, 0], p[:, 1], ".", ms=2.5, c=c, label=lab)
            lo, hi = ta.min(0) - 12, ta.max(0) + 12
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal")
            ax.set_title(f"{paths.FLOOR_NAME[fl]} · " + {"known": "출발점 앎", "drift": "PDR 틀어짐"}[sc] + " (seed 0)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_paths.png", dpi=90)
    plt.close(fig)
    print(out / "fig_error.png", out / "fig_map.png", out / "fig_paths.png")


if __name__ == "__main__":
    main()
