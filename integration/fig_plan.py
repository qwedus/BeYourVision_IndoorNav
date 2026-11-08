"""재생 결과 경로 그림: 정식 도면(벽선·점자블록) 위에 방 뺀 보행 마스크를 옅게 깔고 정답·PDR만·PF·PF + 비전 경로를 그림.

    I1_RECORDING=<이름> .venv/Scripts/python integration/fig_plan.py [--name replay] [--set 기본] [--obs tf] [--seed 0] [--plan <plan_lines.npz|plan.gpkg>]
→ <녹화 설정 out>/<name>/fig_plan.png

PF 입력은 방 뺀 보행 마스크 그대로. 도면은 사람이 보는 바탕이라 결과 숫자와 상관없음. 좌표는 역 좌표.
도면 벽선·점자블록은 기본으로 maps/plan_lines.npz(make_plan_lines.py가 정식 도면에서 옮긴 것)를 읽음.
--plan에 .gpkg를 주면 그 파일을 geopandas로 읽음(그때만 geopandas 필요).
"""
import argparse
import json
import sys

import matplotlib
import numpy as np
from PIL import Image

import paths
from plan_lines import load_lines
from station_pkg import enu_to_aligned, load_station

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
ST = {"확실": "#1a9850", "추정": "#f28e2b", "모름": "#d62728"}


def lines_st(path, layer, R, T):
    """[(도형 종류, 역 좌표), ...]. path 없음 → maps/plan_lines.npz, .gpkg → geopandas로 원본 도면, 그 밖 → 그 npz"""
    if path is None or not str(path).lower().endswith(".gpkg"):
        return load_lines(layer, R, T, path)
    import geopandas as gpd
    out = []
    for g in gpd.read_file(path, layer=layer).geometry:
        for part in getattr(g, "geoms", [g]):
            xy = np.asarray(part.exterior.coords if part.geom_type == "Polygon" else part.coords)[:, :2]
            out.append((part.geom_type, (xy - T) @ R))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", help="도면: 기본 maps/plan_lines.npz, .gpkg면 geopandas로 원본")
    ap.add_argument("--name", default="replay")
    ap.add_argument("--set", default="기본")
    ap.add_argument("--obs", default="tf")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    fmap = load_station(paths.INPUTS)
    R, T = fmap[-1]["R"], fmap[-1]["T"]
    meta = json.loads(paths.MASK_META.read_text(encoding="utf-8"))
    rot = meta["aligned_from_enu_rot_deg"]
    shift = (enu_to_aligned(T[None], rot))[0]     # 역 좌표 = aligned - shift (회전이 같음)
    st = lambda en: (np.atleast_2d(en) - T) @ R
    out = paths.REC_OUT / a.name
    z = np.load(out / "runs.npz")
    tr = np.genfromtxt(paths.I1 / "truth.csv", delimiter=",", names=True)
    wps = paths.RC["waypoints"]
    run = lambda k: z[f"known_{k}_{a.seed}"]
    on, off, dr = run(f"{a.obs}_{a.set}"), run("none"), z["known_pdr_only_0"]
    fls = [fl for fl in (-1, -2) if ((tr["floor"] == fl) & (tr["on_esc"] == 0)).any()]
    fig, axs = plt.subplots(1, len(fls), figsize=(11 * len(fls), 10), squeeze=False)
    for ax, fl in zip(axs[0], fls):
        name = paths.FLOOR_NAME[fl]
        f = meta["floors"][name]
        walk = np.array(Image.open(paths.MASK_DIR / f["walk_png"])) > 127
        H, W = walk.shape
        m, o = f["m_per_px"], np.array(f["offset_aligned"]) - shift
        ax.imshow(np.ma.masked_where(~walk, walk), cmap="Blues", vmin=0, vmax=4, alpha=.45, zorder=0,
                  extent=[o[0], o[0] + W * m, o[1] - H * m, o[1]], origin="upper", interpolation="nearest")
        for kind, xy in lines_st(a.plan, f"{name}_walls", R, T):
            ax.plot(xy[:, 0], xy[:, 1], c="#444", lw=.6, zorder=1)
        for kind, xy in lines_st(a.plan, f"{name}_tactile", R, T):
            ax.fill(xy[:, 0], xy[:, 1], c="#e6b800", alpha=.55, lw=0, zorder=1)
        s = (tr["floor"] == fl) & (tr["on_esc"] == 0)
        q = st(np.c_[tr["east"][s], tr["north"][s]])
        ax.plot(q[:, 0], q[:, 1], c="#d7191c", lw=2.5, zorder=4, label="정답")
        d = dr[dr[:, 3] == fl]
        if len(d):
            p = st(d[:, 1:3])
            ax.plot(p[:, 0], p[:, 1], "--", c="k", lw=1.3, zorder=3, label="PDR만")
        for r, c, lab in ((off, "#7f7f7f", "PF 비전 끔"), (on, "#2c7bb6", f"PF + 비전 ({'시간 필터' if a.obs == 'tf' else '필터 없음'}·{a.set})")):
            k = (r["efloor"] == fl) & ~r["riding"]
            if k.any():
                p = st(np.c_[r["ex"][k], r["ey"][k]])
                ax.plot(p[:, 0], p[:, 1], ".", ms=3, c=c, zorder=3, label=lab)
        for i, w in enumerate(wps):
            if w["floor"] == fl:
                ax.plot(*w["st"], "o", ms=8, c=ST.get(w.get("status"), "w"), mec="k", zorder=5)
                ax.annotate(f"{i}", w["st"], xytext=(4, 4), textcoords="offset points", fontsize=9, fontweight="bold", zorder=6)
        allp = [q]
        if len(d):
            allp.append(st(d[:, 1:3]))
        for r in (off, on):
            k = (r["efloor"] == fl) & ~r["riding"]
            if k.any():
                allp.append(st(np.c_[r["ex"][k], r["ey"][k]]))
        P = np.vstack(allp)
        lo, hi = q.min(0), q.max(0)
        lo, hi = np.minimum(lo - 8, np.percentile(P, 1, axis=0) - 3), np.maximum(hi + 8, np.percentile(P, 99, axis=0) + 3)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        ax.set_title(f"{name}  ·  출발점 앎, seed {a.seed}  ·  숫자 = 기준점(초록 확실, 주황 추정)", fontsize=11)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"{paths.RC['note'] or paths.RECORDING}  ({paths.RECORDING})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_plan.png", dpi=85)
    print(out / "fig_plan.png")


if __name__ == "__main__":
    main()
