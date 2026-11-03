"""동춘역 세션 재생: PDR → PF(B01 대합실) → 에스컬레이터(기압) → PF(B02 승강장).

정답: 사용자가 평면도에 그린 경로(maps/dongchun/*_drawn_path.png), 출발점, B01 끝(에스컬레이터 위 끝), B02 끝(화살표 끝).
채점: (1) 추정 궤적이 그린 경로에서 떨어진 평균·최대 거리, (2) B01 끝·최종 위치 오차.
출발 방향은 두 가설(H1 이미지 오른쪽, H2 이미지 아래)로 모두 돌려 그린 경로와 더 비슷한 쪽을 고른다.

에스컬레이터 처리 (지도 + 기압 = 동작 랜드마크):
  기압이 30 s 동안 가장 크게 오르는 구간(≈ 한 층 하강)을 탑승으로 보고,
  탑승 시작에 '에스컬레이터 위 끝 근처'로 가중치(σ 2.5 m, 위·아래 승강장 에스컬레이터 둘 다 후보),
  탑승 중 걸음은 위치에 반영하지 않고(서 있거나 걸어도 수평 이동은 에스컬레이터 길이로 고정),
  탑승 끝에 각 파티클을 가장 가까운 에스컬레이터의 아래 끝으로 옮기고 B02 지도로 바꾼다.

사용: uv run python scripts/run_dongchun.py data/dongchun/<세션1> data/dongchun/<세션2>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from indoornav.io import load_session  # noqa: E402
from indoornav.pdr import PDRParams, run_pdr  # noqa: E402
from indoornav.pf.filter import ParticleFilter  # noqa: E402
from indoornav.pf.gridmap import GridMap  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from tune_pf import params_from_toml  # noqa: E402

META = ROOT / "maps/dongchun/map_meta.json"


def escalator_window(s, ride_s=30.0):
    p = uniform_filter1d(s.pressure, 300)
    w = int(ride_s * s.fs)
    i = int(np.argmax(p[w:] - p[:-w]))
    return float(s.t[i]), float(s.t[i] + ride_s), float(p[i + w] - p[i])


def drawn_path_world(meta, floor):
    f = meta["floors"][floor]
    m = np.array(Image.open(META.parent / f["drawn_path_png"])) > 127
    ys, xs = np.where(m)
    g = GridMap.load(META, floor)
    return g.to_world(xs, ys)


def run_one(s, pdr, P, meta, th0_deg, seed, use_esc_landmark=True):
    maps = {fl: GridMap.load(META, fl) for fl in ("B01", "B02")}
    esc = meta["escalators"]
    tops = np.array([e["top"] for e in esc]); bots = np.array([e["bottom"] for e in esc])
    ta, tb, dp = escalator_window(s)
    st = pdr.steps
    hrel = st.heading.to_numpy() - pdr.heading[0]
    dh = np.diff(np.r_[0.0, hrel])
    pf = ParticleFilter(P, maps["B01"], seed=seed)
    th0 = np.radians(th0_deg)
    pf.init(np.array(meta["route_truth"]["start_B01"]), th0)
    track = {"B01": [], "B02": []}
    floor, rode, b01_end, res_esc = "B01", False, None, None
    for i in range(len(st)):
        t = float(st.t.iloc[i])
        if floor == "B01" and t >= ta and not rode:
            b01_end = pf.estimate()["xy"].copy()     # 에스컬레이터 정보 넣기 전 PF 자체 추정 (채점용)
            rode = True
        if rode and floor == "B01" and t >= tb:
            # 에스컬레이터 = 위치가 알려진 동작 랜드마크: 각 파티클을 위·아래 승강장 에스컬레이터 중 하나의
            # 아래 끝으로 (가까울수록 높은 확률, σ 10 m). 잘못 고른 승강장은 B02 벽이 걸러낸다.
            d = np.linalg.norm(pf.xy[:, None, :] - tops[None], axis=2)
            pr = np.exp(-0.5 * (d / 10.0) ** 2) + 1e-6; pr /= pr.sum(1, keepdims=True)
            rng = pf.rng
            k = (rng.random(len(pr))[:, None] > np.cumsum(pr, 1)).sum(1)
            if not use_esc_landmark:   # 비교용: 가장 가까운 에스컬레이터로 상대 위치 유지
                k = d.argmin(1); pf.xy = pf.xy + (bots[k] - tops[k])
            else:
                pf.xy = bots[k] + rng.normal(0, 0.7, (len(k), 2))
            bad = ~maps["B02"].free(pf.xy)
            pf.xy[bad] = bots[k[bad]]
            pf.logw = np.full(P.n, -np.log(P.n))
            res_esc = np.bincount(k, minlength=len(tops)) / len(k)
            pf.map = maps["B02"]; floor = "B02"
        if floor == "B01" and rode:          # 탑승 중: 방향만 반영
            pf.predict(0.0, float(dh[i]), True)
        else:
            pf.predict(float(st.length.iloc[i]), float(dh[i]), bool(st.moving.iloc[i]))
        if not (floor == "B01" and rode):
            track[floor].append(pf.estimate()["xy"].copy())
    est = pf.estimate()
    return {"track": {k: np.array(v) for k, v in track.items()}, "b01_end": b01_end, "final": est["xy"],
            "esc_share": res_esc.tolist() if rode and floor == "B02" else None,
            "unc95": est["unc95"], "k": est["k"], "esc": (ta, tb, dp), "stats": dict(pf.stats)}


def pdr_only(pdr, meta, th0_deg):
    xy = pdr.steps[["x", "y"]].to_numpy()
    c, s_ = np.cos(np.radians(th0_deg)), np.sin(np.radians(th0_deg))
    return np.array(meta["route_truth"]["start_B01"]) + xy @ np.array([[c, -s_], [s_, c]]).T


def score(res, paths, meta):
    out = {}
    for fl in ("B01", "B02"):
        tr = res["track"][fl]
        if len(tr):
            d, _ = cKDTree(paths[fl]).query(tr)
            out[f"{fl}_path_mean"], out[f"{fl}_path_max"] = float(d.mean()), float(d.max())
    esc_top = np.array(meta["escalators"][0]["top"])
    out["B01_end_err"] = float(np.linalg.norm(res["b01_end"] - esc_top)) if res["b01_end"] is not None else np.nan
    out["final_err"] = float(np.linalg.norm(res["final"] - np.array(meta["route_truth"]["end_B02"])))
    return out


def plot(name, s_res, pdr_xy, paths, meta, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from indoornav.pf.plots import set_korean_font
    set_korean_font()
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 1.6]})
    srcs = {"B01": "B01_src.jpg", "B02": "B02_src.jpg"}
    for ax, fl in zip(axs, ("B01", "B02")):
        g = GridMap.load(META, fl)
        src = META.parent / srcs[fl]
        h, w = g.walk.shape
        x0, y1 = g.to_world(0, 0); x1, y0 = g.to_world(w, h)
        if src.exists():
            ax.imshow(np.array(Image.open(src)), extent=[x0, x1, y0, y1], alpha=0.55)
        ax.scatter(paths[fl][::7, 0], paths[fl][::7, 1], s=1, c="red", alpha=0.35, label="그린 경로(정답)")
        tr = s_res["track"][fl]
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], "-", c="tab:blue", lw=2, label="PDR+PF 추정")
        if fl == "B01":
            ax.plot(pdr_xy[:, 0], pdr_xy[:, 1], "--", c="gray", lw=1.2, label="PDR 단독")
            ax.plot(*meta["route_truth"]["start_B01"], "go", ms=9, label="출발")
            ax.plot(*s_res["b01_end"], "bs", ms=8, label="에스컬레이터 탑승 시 추정")
        else:
            ax.plot(*meta["route_truth"]["end_B02"], "r*", ms=14, label="도착(정답)")
            ax.plot(*s_res["final"], "b*", ms=12, label="도착(추정)")
            ax.set_xlim(paths["B02"][:, 0].min() - 25, paths["B02"][:, 0].max() + 25)
            ax.set_ylim(paths["B02"][:, 1].min() - 8, paths["B02"][:, 1].max() + 8)
        ax.set_aspect("equal"); ax.set_title(f"{name} · {fl}"); ax.legend(fontsize=8, loc="lower left")
        if fl == "B01":
            ax.set_xlim(paths["B01"][:, 0].min() - 15, paths["B01"][:, 0].max() + 15)
            ax.set_ylim(paths["B01"][:, 1].min() - 12, paths["B01"][:, 1].max() + 12)
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--pdr-config", default=ROOT / "configs/pdr_tuned_v4.toml")
    ap.add_argument("--pf-config", default=ROOT / "configs/pf_home_v3.toml")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=ROOT / "reports/dongchun")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    paths = {fl: drawn_path_world(meta, fl) for fl in ("B01", "B02")}
    P = params_from_toml(a.pf_config); P.n = a.n; P.init_pos_sigma = 0.5; P.init_h_sigma = np.radians(10)
    pp = PDRParams.from_toml(a.pdr_config)
    rows = []
    for sp in a.sessions:
        s = load_session(sp); pdr = run_pdr(s, pp); name = Path(sp).name
        best = None
        for hyp, th in meta["route_truth"]["start_heading_deg_hyp"].items():
            res = [run_one(s, pdr, P, meta, th, sd) for sd in range(a.seeds)]
            sc = [score(r, paths, meta) for r in res]
            m = {k: float(np.nanmean([x[k] for x in sc])) for k in sc[0]}
            pd_xy = pdr_only(pdr, meta, th)
            d_pdr = cKDTree(paths["B01"]).query(pd_xy[pdr.steps.t.to_numpy() < res[0]["esc"][0]])[0]
            m["PDR_B01_path_mean"] = float(d_pdr.mean())
            m["lower_share"] = float(np.mean([r["esc_share"][0] for r in res]))
            row = {"session": name, "hyp": hyp, "steps": pdr.n_steps, "pdr_dist": round(pdr.distance, 1),
                   "esc_s": [round(x, 1) for x in res[0]["esc"]], **{k: round(v, 2) for k, v in m.items()}}
            rows.append(row); print(json.dumps(row, ensure_ascii=False))
            obj = m["B01_path_mean"] + m.get("B02_path_mean", 0)
            per = [x["B01_path_mean"] + x.get("B02_path_mean", 0) + x["final_err"] for x in sc]
            med = int(np.argsort(per)[len(per) // 2])          # 그림은 '중간' 시드 (가장 좋은 시드 아님)
            if best is None or obj < best[0]:
                best = (obj, hyp, res[med], pd_xy)
        plot(f"{name} ({best[1]})", best[2], best[3], paths, meta, out / f"{name}_{best[1]}.png")
        print(f"{name}: 그린 경로와 가장 비슷한 출발 방향 = {best[1]}")
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
