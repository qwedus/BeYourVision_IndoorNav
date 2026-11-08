"""실시간 권장 조합(rec) 경로 그림: fig_plan.py와 같은 바탕(정식 도면 + 방 뺀 보행 마스크)에 정답·PDR만·PF 비전 끔·PF + 비전을 그림.

두 단계:
    I1_RECORDING=<이름> .venv/Scripts/python integration/fig_plan_rt.py data [--seed 0]
        → <녹화 out>/rt_fig/rec_paths.npz (PF 비전 끔을 rec 조건으로 새로 돌림, PDR만은 rec 걸음으로 쌓음)
    I1_RECORDING=<이름> PYTHONPATH=integration .venv/Scripts/python integration/fig_plan_rt.py draw [--seed 0] [--plan <plan_lines.npz|plan.gpkg>]
        → <녹화 out>/rt_fig/fig_plan_rec.png, fig_plan_compare.png(위: 지금 방식, 아래: rec)

PF + 비전 rec는 run_rt.py 결과(<녹화 out>/rt_compare/runs.npz `rec_<seed>`)를 읽음. 지금 방식은 기준 exp_d58_r20 runs.npz.
점은 프레임마다 PF가 낸 위치. 실시간에서 한 번 낸 위치는 고치지 않으므로 화면에 보인 점과 같고, 나온 시각만 늦음(rec 중앙 약 1.06 s).
도면은 fig_plan.py와 같음(기본 maps/plan_lines.npz, --plan .gpkg면 geopandas). 기존 결과 파일은 읽기만 함. 기본 설정·기존 그림(fig_plan.png)은 안 건드림.
"""
import argparse
import json
import sys

import numpy as np

import paths

REF = "exp_d58_r20"
OUT = "rt_fig"
STAGE = "rec"


# ---------------------------------------------------------------- data (브랜치 .venv)
def rt_dead_reckon(steps, cands, samples, truth, t0, escalators):
    """PDR만(지도·필터 없음), rec 조건: rec 걸음(확정된 걸음 전부 이동, 봉우리 뒤 0.5 s)을 PF에 넣는 것과 같은 지식으로 쌓음.
    탑승(3 m 확정된 후보)은 확정을 받은 뒤 처리된 걸음만 회전만(의심 무시와 같음), 층 전환은 탑승 끝 표본 시각에 가장 가까운 에스컬레이터 끝으로.
    출발점·방향은 정답. 좌표 ENU → 행 (t, east, north, floor)"""
    r0 = truth[0]
    x, y, h, fl = r0["east"], r0["north"], r0["heading_deg"], int(r0["floor"])
    rides = []
    for c in cands:
        if c["k_confirm"] is not None and (c["ride"] or c["k_end"] is None):     # 끝 표본을 아직 못 받은 확정 탑승도 탑승(OnlineBaro와 같음)
            rides.append((c["t_first"], c["t_end"] if c["k_end"] is not None else float("inf"), c["k_confirm"],
                          c["delta"] if c["k_end"] is not None else None))      # k_confirm = 3 m 표본 수신 시각
    ev = [(c[1], 0, c[3]) for c in rides if c[3] is not None] + [(s["t"] - t0, 1, s) for s in steps]
    ev.sort(key=lambda e: (e[0], e[1]))
    out = [(0.0, x, y, fl)]
    for t, kind, v in ev:
        if kind == 1:
            h += v["dheading_deg"]
            known = v["t_known"] - t0
            if not any(a <= t <= b and known >= rc for a, b, rc, _ in rides):
                x += v["length_m"] * np.sin(np.radians(h))
                y += v["length_m"] * np.cos(np.radians(h))
        else:
            src, dst = ("top", "bottom") if v < 0 else ("bottom", "top")
            k = int(np.argmin([np.hypot(x - e[src][0], y - e[src][1]) for e in escalators]))
            x, y = escalators[k][dst]
            fl += v
        out.append((t, x, y, fl))
    return np.array(out)


def data(a):
    import time
    import rt_pdr
    import rt_replay as rr
    import run_i1
    import run_rt
    from time_filter import TrackLimiter
    from vision_filter import VisionFilter
    src = paths.REC_OUT / "rt_compare"
    out = paths.REC_OUT / OUT
    out.mkdir(exist_ok=True)
    n_rt = json.loads((src / "summary.json").read_text(encoding="utf-8"))["info"]["n"]   # PF + 비전 rec 결과와 같은 입자 수
    if a.n is not None and a.n != n_rt:
        raise SystemExit(f"--n {a.n}이 실시간 재생 결과의 입자 수 {n_rt}와 다름")
    a.n = n_rt
    run_rt._init(str(src), a.n)
    G = run_i1._G
    sg = run_rt.stage(STAGE)
    r0 = G["truth"][0]
    st = run_i1.SETTINGS["기본"]
    F = VisionFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                     float(r0["heading_deg"]), a.seed, st["obs_params"])
    lim = TrackLimiter(G["opt"]["track_limit"]) if G["opt"].get("track_limit") else None
    sk, bk = run_rt.knowledge(sg)
    t = time.perf_counter()
    off = rr.run_rt(F, G["obs"]["tf"], sk, bk, G["truth"], G["t0"], G["rides"], vision=False, obs_hz=st["obs_hz"], limiter=lim,
                    heading=run_rt.heading_for(sg) if run_rt._R["bearing"] else None, max_wait=sg["max_wait"], log={})
    tag = f"l{int(round(sg['look'] * 100)):03d}"
    steps = run_rt._R["steps"][f"steps_{sg['steps']}_{tag}"]
    cands, samples = run_rt._R["baro"]
    dr = rt_dead_reckon(steps, cands, samples, G["truth"], G["t0"], G["smeta"]["escalators"])
    on = np.load(src / "runs.npz")[f"{STAGE}_{a.seed}"]
    np.savez(out / "rec_paths.npz", on=on, off=off, dr=dr)
    (out / "rec_paths.json").write_text(json.dumps(dict(stage=STAGE, stage_values={k: v for k, v in sg.items() if k != "note"},
                                                        seed=a.seed, n=a.n, ref=REF, s_off=round(time.perf_counter() - t, 1),
                                                        n_steps=len(steps), n_rides=sum(c["ride"] for c in cands)),
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / "rec_paths.npz", f"PF 비전 끔 {len(off)}행, PDR만 {len(dr)}점")


# ---------------------------------------------------------------- draw (브랜치 .venv)
def draw(a):
    import matplotlib
    from PIL import Image
    from fig_plan import ST, lines_st
    from station_pkg import enu_to_aligned, load_station
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fmap = load_station(paths.INPUTS)
    R, T = fmap[-1]["R"], fmap[-1]["T"]
    meta = json.loads(paths.MASK_META.read_text(encoding="utf-8"))
    shift = (enu_to_aligned(T[None], meta["aligned_from_enu_rot_deg"]))[0]
    stf = lambda en: (np.atleast_2d(en) - T) @ R
    out = paths.REC_OUT / OUT
    z = np.load(paths.REC_OUT / REF / "runs.npz")
    zr = np.load(out / "rec_paths.npz")
    seed_d = json.loads((out / "rec_paths.json").read_text(encoding="utf-8"))["seed"]
    if seed_d != a.seed:
        raise SystemExit(f"draw --seed {a.seed}이 data의 seed {seed_d}와 다름 → data --seed {a.seed}를 먼저")
    sets = {"now": (z[f"known_tf_기본_{a.seed}"], z[f"known_none_{a.seed}"], z["known_pdr_only_0"]),
            "rec": (zr["on"], zr["off"], zr["dr"])}
    lab = {"now": "지금 방식", "rec": "실시간 권장 조합"}
    tr = np.genfromtxt(paths.I1 / "truth.csv", delimiter=",", names=True)
    wps = paths.RC["waypoints"]
    fls = [fl for fl in (-1, -2) if ((tr["floor"] == fl) & (tr["on_esc"] == 0)).any()]

    def pts(r, fl):
        k = (r["efloor"] == fl) & ~r["riding"]
        return stf(np.c_[r["ex"][k], r["ey"][k]]) if k.any() else None

    def limits(fl, keys):
        s = (tr["floor"] == fl) & (tr["on_esc"] == 0)
        q = stf(np.c_[tr["east"][s], tr["north"][s]])
        allp = [q]
        for key in keys:
            on, off, dr = sets[key]
            d = dr[dr[:, 3] == fl]
            if len(d):
                allp.append(stf(d[:, 1:3]))
            allp += [p for p in (pts(off, fl), pts(on, fl)) if p is not None]
        P = np.vstack(allp)
        lo, hi = q.min(0), q.max(0)
        return np.minimum(lo - 8, np.percentile(P, 1, axis=0) - 3), np.maximum(hi + 8, np.percentile(P, 99, axis=0) + 3), q

    def panel(ax, fl, key, lim):
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
        lo, hi, q = lim
        ax.plot(q[:, 0], q[:, 1], c="#d7191c", lw=2.5, zorder=4, label="정답")
        on, off, dr = sets[key]
        d = dr[dr[:, 3] == fl]
        if len(d):
            p = stf(d[:, 1:3])
            ax.plot(p[:, 0], p[:, 1], "--", c="k", lw=1.3, zorder=3, label="PDR만")
        for r, c, lb in ((off, "#7f7f7f", "PF 비전 끔"), (on, "#2c7bb6", "PF + 비전 (시간 필터·기본)")):
            p = pts(r, fl)
            if p is not None:
                ax.plot(p[:, 0], p[:, 1], ".", ms=3, c=c, zorder=3, label=lb)
        for i, w in enumerate(wps):
            if w["floor"] == fl:
                ax.plot(*w["st"], "o", ms=8, c=ST.get(w.get("status"), "w"), mec="k", zorder=5)
                ax.annotate(f"{i}", w["st"], xytext=(4, 4), textcoords="offset points", fontsize=9, fontweight="bold", zorder=6)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        ax.set_title(f"{name}  ·  {lab[key]}", fontsize=11)
        ax.legend(loc="best", fontsize=9)

    import textwrap
    notes = ["지금 방식 = 녹화 전체를 보고 계산  ·  실시간 권장 조합 = 그 시각까지 받은 값만 씀(위치는 약 1.1 s 늦게 나옴)",
             "점 = 프레임마다 PF가 낸 위치(실시간은 낸 뒤 안 고치므로 화면에 보인 점과 같음, 나온 시각만 늦음)",
             f"출발점 앎, seed {a.seed}  ·  숫자 = 기준점(초록 확실, 주황 추정)"]
    desc = (paths.RC["note"] or "").split(". ")[0]       # 녹화 설명 첫 문장만(뒤 문장은 입력 파일 안내라 그림에 안 씀)
    title = f"{desc}  ({paths.RECORDING})" if desc else paths.RECORDING
    files = []
    for rows, fname in ((("rec",), "fig_plan_rec.png"), (("now", "rec"), "fig_plan_compare.png")):
        lims = [limits(fl, rows) for fl in fls]          # 비교 그림은 위아래 같은 범위
        size = []
        for lo, hi, _ in lims:                           # 패널 크기를 도면 범위 비율에 맞춤(긴 변 11 in)
            r = (hi[1] - lo[1]) / (hi[0] - lo[0])
            size.append((11.0, max(11 * r, 3.5)) if r <= 1 else (max(11 / r, 4.5), 11.0))
        W = max(sum(w for w, _ in size) + 1.0, 10.0)
        lines = [l for n in [title] + notes for l in textwrap.wrap(n, int(W / .14))]
        head = .3 + .25 * len(lines)
        ph = max(h for _, h in size) + .6
        H = ph * len(rows) + head
        fig, axs = plt.subplots(len(rows), len(fls), figsize=(W, H), squeeze=False,
                                gridspec_kw=dict(width_ratios=[w for w, _ in size]))
        for j, fl in enumerate(fls):
            for i, key in enumerate(rows):
                panel(axs[i][j], fl, key, lims[j])
        ntitle = len(textwrap.wrap(title, int(W / .14)))
        for k, l in enumerate(lines):
            fig.text(.5, 1 - (.2 + .25 * k) / H, l, ha="center", va="top", fontsize=12 if k < ntitle else 9.5,
                     fontweight="bold" if k < ntitle else "normal")
        fig.tight_layout(rect=(0, 0, 1, 1 - head / H))
        fig.savefig(out / fname, dpi=85, bbox_inches="tight", pad_inches=.2)
        plt.close(fig)
        files.append(out / fname)
    print(*files, sep="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["data", "draw"])
    ap.add_argument("--plan", help="draw 도면: 기본 maps/plan_lines.npz, .gpkg면 geopandas로 원본")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, help="입자 수 (기본: 실시간 재생 결과와 같게)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.step == "data":
        data(a)
    else:
        draw(a)


if __name__ == "__main__":
    main()
