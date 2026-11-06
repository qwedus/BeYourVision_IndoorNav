"""원인 분리 실험 집계 (exp_cause.py 결과 → 표 재료 JSON). 지표만 계산, 재생은 안 함.

    I1_RECORDING=<이름> .venv/Scripts/python integration/exp_report.py [--segments 시작-끝,시작-끝]
→ <녹화 out>/<name, 기본 exp_base>/cause.json

A k: 시간별 k 평균·분위, k 하한 근처 비율, PF 이동 거리 비율, 평균 k 변화의 걸음 단계·관측 단계 몫
D 구간: 관측 수·종류·공백, 그 자리에서 보일 수 있는 랜드마크 수, PDR 보폭 비율·방향 오차, PF 앞섬(+)/뒤짐(-)·옆 벗어남(+ 오른쪽), k
E 과신: seed별로 '불확실도 < 2 m, 오차 > 3 m'인 구간, 오차 ≤ 2×불확실도 비율
판 간격 거리: 두 '나가는 곳' 판(5·6번, 지도 간격 4.0 m, y 89.1)이 한 화면에 잡힌 프레임 → y = 89.1 - fx×4.0/간격 (북쪽으로 걸을 때만)
"""
import argparse
import collections
import json
import sys

import numpy as np

import paths
import replay as rp
from analyze_recording import camera_offset
from edge_faces import frame_obs
from obs_model import _cells
from pdr_steps import read_steps
from station_pkg import load_station
from time_filter import load_frames

KEYS = ("known_none", "known_tf_기본", "known_tf_중간")
PLATE_DX, PLATE_Y = 4.0, 89.1
HALF_FOV = 35.0


def seeds(z, key):
    return [z[k] for k in sorted(z.files, key=lambda s: int(s.rsplit("_", 1)[1])) if k.rsplit("_", 1)[0] == key]


def k_table(z, key, t_end):
    a, x = seeds(z, key), seeds(z, "x_" + key)
    t = a[0]["t"]
    grid = np.linspace(0, 1, 11)                    # 기록 길이의 10% 간격
    out = []
    for g in grid:
        i = min(np.searchsorted(t, g * t_end), len(t) - 1)
        out.append(dict(t=float(t[i]), k=float(np.median([r["k"][i] for r in a])), k_std=float(np.median([r["k_std"][i] for r in a])),
                        p10=float(np.median([r["k_p10"][i] for r in x])), p90=float(np.median([r["k_p90"][i] for r in x])),
                        near=float(np.median([r["k_near"][i] for r in x]))))
    w = ~a[0]["esc"] & ~a[0]["riding"] & ~paths.excluded(t)
    return dict(series=out, k_mean=float(np.median([r["k"][w].mean() for r in a])), near_mean=float(np.median([r["k_near"][w].mean() for r in x])),
                near_max=float(np.median([r["k_near"][w].max() for r in x])),
                dk_pred=float(np.median([r["dk_pred"][-1] for r in x])), dk_upd=float(np.median([r["dk_upd"][-1] for r in x])),
                walls=float(np.median([r["walls"][-1] for r in x])), walls_per_step=None)


def overconfident(z, key, wps):
    a = seeds(z, key)
    t = a[0]["t"]
    w = ~a[0]["esc"] & ~a[0]["riding"] & ~paths.excluded(t)
    bad = np.array([(r["unc"] < 2.0) & (r["err"] > 3.0) & w for r in a])
    cnt = bad.sum(0)
    out, i = [], 0
    idx = np.where(cnt > 0)[0]
    if len(idx):
        groups, s, p = [], idx[0], idx[0]
        for j in idx[1:]:
            if t[j] - t[p] > 2.0:
                groups.append((s, p))
                s = j
            p = j
        groups.append((s, p))
        for s, e in groups:
            sd = np.where(bad[:, s:e + 1].any(1))[0]
            out.append(dict(t0=float(t[s]), t1=float(t[e]), seeds=len(sd), err_max=float(np.median([a[q]["err"][s:e + 1].max() for q in sd])),
                            unc=float(np.median([np.median(a[q]["unc"][s:e + 1]) for q in sd]))))
    return dict(seeds_wrong=int(bad.any(1).sum()), intervals=out, cover2=float(np.median([np.mean((r["err"] <= 2 * r["unc"])[w & (t > 2)]) for r in a])))


def segment(z, t0, t1, truth, steps, tstep, frames, fm_all, off, ty, yaw, t_zero, R, T):
    tt = (truth["t"] >= t0) & (truth["t"] <= t1)
    tr = truth[tt]
    L = float(np.hypot(np.diff(tr["east"]), np.diff(tr["north"])).sum())
    sel = (tstep >= t0) & (tstep <= t1)
    ln = np.array([s["length_m"] for s in steps])
    mv = np.array([s["moving"] for s in steps])
    hs = np.array([s["heading_deg"] for s in steps])
    h_true = np.array([rp.truth_at(truth, q)["heading_deg"] for q in tstep[sel & mv]]) if (sel & mv).any() else np.array([])
    herr = ((hs[sel & mv] + off - h_true + 180) % 360) - 180
    # 관측
    cnt, cnt_tf, used_t = collections.Counter(), 0, []
    for t, obs in frames:
        if t0 <= t <= t1 and obs:
            cnt_tf += len(obs)
            used_t.append(t)
            cnt.update(o["label"] for o in obs)
    edges = [t0] + used_t + [t1]
    gap = float(np.max(np.diff(edges)))
    # 지도에서 보일 수 있는 판(앞면·가림 표 + 20 m + 화각)
    uniq, per = collections.Counter(), []
    for r in tr[::10]:
        fm = fm_all[int(r["floor"])]
        i, j, ins = _cells(fm, np.array([r["east"]]), np.array([r["north"]]))
        d = fm["dist"][:, i[0], j[0]].astype(float)
        dx, dy = fm["lm_xy"][:, 0] - r["east"], fm["lm_xy"][:, 1] - r["north"]
        rel = (np.degrees(np.arctan2(dx, dy)) - r["heading_deg"] + 180) % 360 - 180
        vis = np.isfinite(d) & (d <= 20) & (np.abs(rel) <= HALF_FOV) & ins[0]
        per.append(int(vis.sum()))
        uniq.update(map(str, fm["lm_id"][vis]))
    res = {}
    for key in KEYS:
        a = seeds(z, key)
        x = seeds(z, "x_" + key)
        t = a[0]["t"]
        q = (t >= t0) & (t <= t1)
        tq = t[q]
        along, cross = [], []
        for r in a:
            al, cr = [], []
            for ti, ex, ey in zip(tq, r["ex"][q], r["ey"][q]):
                g = rp.truth_at(truth, ti)
                h = np.radians(g["heading_deg"])
                v = np.array([ex - g["east"], ey - g["north"]])
                al.append(v @ np.array([np.sin(h), np.cos(h)]))
                cr.append(v @ np.array([np.cos(h), -np.sin(h)]))
            along.append(al)
            cross.append(cr)
        along, cross = np.median(along, 0), np.median(cross, 0)
        e = np.median([r["err"][q] for r in a], 0)
        kp = int(np.argmax(e))
        res[key] = dict(err_med=float(np.median(e)), err_peak=float(e[kp]), t_peak=float(tq[kp]), along_med=float(np.median(along)), cross_med=float(np.median(cross)),
                        along_peak=float(along[kp]), cross_peak=float(cross[kp]), unc_med=float(np.median([r["unc"][q] for r in a])), k_start=float(np.median([r["k"][q][0] for r in a])),
                        k_end=float(np.median([r["k"][q][-1] for r in a])), near=float(np.median([r["k_near"][q].mean() for r in x])),
                        walls=float(np.median([r["walls"][q][-1] - r["walls"][q][0] for r in x])), err_end=float(e[-1]))
    return dict(t0=t0, t1=t1, truth_m=L, steps=int((sel & mv).sum()), steps_all=int(sel.sum()), pdr_m=float(ln[sel & mv].sum()), stride=float(ln[sel & mv].mean()) if (sel & mv).any() else None,
                ratio=float(ln[sel & mv].sum() / max(L, 1e-6)), herr_med=float(np.median(herr)) if len(herr) else None, herr_p10=float(np.percentile(herr, 10)) if len(herr) else None,
                herr_p90=float(np.percentile(herr, 90)) if len(herr) else None, obs_tf=cnt_tf, frames_with_obs=len(used_t), obs_by_label=dict(cnt), gap_max=gap,
                lm_visible_any=len(uniq), lm_visible_med=float(np.median(per)), lm_visible_zero=float(np.mean(np.array(per) == 0)), lm_list=[k.replace("B01_", "") for k, _ in uniq.most_common(8)], pf=res,
                start_st=[float(v) for v in (np.array([tr["east"][0], tr["north"][0]]) - T) @ R], end_st=[float(v) for v in (np.array([tr["east"][-1], tr["north"][-1]]) - T) @ R])


def plate_range(raw_frames, truth, t_zero, R, T):
    rows = []
    for f in raw_frames:
        t = f["frame_info"]["capture_time_s"] - t_zero
        if t < 0 or t > truth["t"][-1]:
            continue
        r = rp.truth_at(truth, t)
        if int(r["floor"]) != -1:
            continue
        p = (np.array([r["east"], r["north"]]) - T) @ R
        if not (45 < p[1] < 86 and 136 < p[0] < 144):
            continue
        ex = [d["box_xyxy"] for d in f["detections"] if d["label"] == "exit" and d["score"] >= 0.5]
        if len(ex) != 2:
            continue
        a, b = sorted(ex, key=lambda v: v[0])
        ha, hb = a[3] - a[1], b[3] - b[1]
        if not (0.8 < ha / hb < 1.25) or abs((a[1] + a[3]) - (b[1] + b[3])) / 2 > 0.3 * ha or a[0] < 5 or b[2] > 1915:
            continue
        sep = (b[0] + b[2] - a[0] - a[2]) / 2
        rows.append((t, PLATE_Y - f["frame_info"]["intrinsics"]["fx"] * PLATE_DX / sep, p[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="")
    ap.add_argument("--name", default="exp_base")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = paths.REC_OUT / a.name
    z = np.load(out / "runs.npz")
    truth = rp.load_truth(paths.I1 / "truth.csv")
    t_zero = paths.rec_t0()
    fm_all = load_station(paths.INPUTS)
    R, T = fm_all[-1]["R"], fm_all[-1]["T"]
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - t_zero, -np.degrees(np.unwrap(h["heading"]))
    off = camera_offset(truth, paths.RC["waypoints"], ty, yaw)
    steps = read_steps(paths.REC_OUT / "steps_indoornav.csv")
    tstep = np.array([s["t"] - t_zero for s in steps])
    raw = load_frames(paths.I1 / "obs_timefilter.jsonl")[1]
    frames = sorted((frame_obs(f, t_zero) for f in raw), key=lambda v: v[0])
    res = dict(k={k: k_table(z, k, float(truth["t"][-1])) for k in KEYS}, over={k: overconfident(z, k, paths.RC["waypoints"]) for k in KEYS[1:]})
    res["segments"] = []
    for s in filter(None, a.segments.split(",")):
        p, q = map(float, s.split("-"))
        res["segments"].append(segment(z, p, q, truth, steps, tstep, frames, fm_all, off, ty, yaw, t_zero, R, T))
    pr = plate_range(load_frames(paths.I1 / "obs_nofilter.jsonl")[1], truth, t_zero, R, T)
    if pr:
        v = np.array(pr)
        b = {}
        for t, y, yt in pr:
            b.setdefault(int(t // 5) * 5, []).append((y, yt))
        res["plate"] = [dict(t=k, y_plate=float(np.median([x[0] for x in w])), y_truth=float(np.median([x[1] for x in w])), n=len(w)) for k, w in sorted(b.items())]
    (out / "cause.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(paths.RECORDING, "ok", len(res["segments"]), len(res.get("plate", [])), flush=True)


if __name__ == "__main__":
    main()
