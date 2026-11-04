"""진단: 끝 k가 왜 줄어드는가, 끝(3-1 문 앞) 오차가 왜 시뮬보다 큰가 (작업기록 3번). indoornav 코드는 고치지 않고 조건만 바꿔 재생.

    .venv/Scripts/python integration/diag_k_end.py   → integration/out/i1/diag_k_end.json·diag_k_end.md

하나씩 빼 보는 조건
    걸음  : indoornav PDR / 이상적 걸음(같은 걸음 시각, 보폭·방향은 정답 궤적에서 계산 → PDR 오차 없음)
    벽    : 있음 / 없음 (pf.map = None)
    비전  : 끔 / 필터 없음 / 시간 필터 × 기본·중간
    k     : 그대로 / 하차 때 k = 1로 되돌림 / 처음부터 1로 고정
    + PDR 틀어짐에서 k 그대로 / 하차 때 1 (되돌림이 틀어진 보폭을 해치는지)
같이 보는 것: 구간별 PDR 보폭 합 vs 정답 거리, 정답 기준점이 의심되는 두 구간(8~10 s, 77~78 s)을 뺀 과신 지표
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import paths
import replay as rp
from run_i1 import DRIFT, SETTINGS, STEPS

SUSPECT = ((8.0, 10.0), (77.0, 78.0))
_G = {}


def ideal_steps(steps, truth, t0, rot):
    """같은 걸음 시각에서 보폭·방향을 정답 궤적으로 계산. 에스컬레이터에 걸친 걸음은 보폭 0."""
    out, prev, prev_th = [], truth[0], np.radians(90.0 + rot - truth[0]["heading_deg"])
    for s in steps:
        tr = rp.truth_at(truth, s["t"] - t0)
        de, dn = tr["east"] - prev["east"], tr["north"] - prev["north"]
        ride = bool(tr["on_esc"] or prev["on_esc"] or tr["floor"] != prev["floor"])
        L = 0.0 if ride else float(np.hypot(de, dn))
        dh = 0.0
        if L > 1e-6:
            th = np.radians(90.0 + rot - np.degrees(np.arctan2(de, dn)))
            dh = float((th - prev_th + np.pi) % (2 * np.pi) - np.pi)
            prev_th = th
        out.append(dict(s, length_m=L, dheading_rad=dh, dheading_deg=-np.degrees(dh), moving=True))
        prev = tr
    return out


def _init():
    from edge_faces import add_face_points
    from pdr_steps import read_steps
    from station_pkg import load_station
    from vision_filter import load_maps, pf_params
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")
    steps = rp.indoornav_sign(read_steps(STEPS["indoornav"]))
    _G.update(fmap=add_face_points(fmap), smeta=smeta, maps=load_maps(), t0=t0, truth=truth,
              obs={"raw": rp.load_obs(paths.I1 / "obs_nofilter.jsonl", t0),
                   "tf": rp.load_obs(paths.I1 / "obs_timefilter.jsonl", t0)},
              steps=dict(pdr=steps, ideal=ideal_steps(steps, truth, t0, smeta["aligned_from_enu_rot_deg"])),
              pf_params=pf_params, drift=rp.drift_steps(steps, **DRIFT))
    _G["rides"], _G["floors"] = rp.baro_events(rp.read_baro(paths.REC / "baro.csv", t0))


def _job(c):
    from vision_filter import VisionFilter

    class F(VisionFilter):
        def change_floor(self, d):
            super().change_floor(d)
            if not c["wall"]:
                self.pf.map = None
            if c["k"] == "reset":
                self.pf.k = np.ones_like(self.pf.k)

    over = dict(k_min=1.0, k_max=1.0, sigma_k=0.0, rough_k=0.0) if c["k"] == "fixed" else {}
    P = _G["pf_params"](3000, use_map=c["wall"], **over)
    st = SETTINGS[c["set"]]
    r0 = _G["truth"][0]
    f = F(P, _G["maps"], _G["fmap"], _G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]], float(r0["heading_deg"]),
          c["seed"], st["obs_params"])
    if c["k"] == "fixed":
        f.pf.k = np.ones_like(f.pf.k)
    obs = _G["obs"]["raw" if c["obs"] == "none" else c["obs"]]
    r = rp.run_one(f, obs, _G["steps"][c["steps"]] if c.get("sc") != "drift" else _G["drift"], _G["rides"], _G["floors"], _G["truth"], _G["t0"],
                   vision=c["obs"] != "none", obs_hz=st["obs_hz"])
    m = rp.metrics(r)
    t = r["t"]
    tr = _G["truth"]
    stop = t >= rp.T_STOP
    # 끝 오차를 마지막 구간 진행 방향(앞 +)과 옆으로 나눔. 정답 마지막 구간: 왼쪽 회전 뒤 → 3-1 문 앞
    a, b = tr[np.searchsorted(tr["t"], 108.5)], tr[np.searchsorted(tr["t"], 114.4)]
    u = np.array([b["east"] - a["east"], b["north"] - a["north"]])
    u = u / np.hypot(*u)
    de = np.c_[r["ex"][stop] - b["east"], r["ey"][stop] - b["north"]]
    seg_along = {}
    for g in (1, 2, 3, 4, 5, 7, 8):                # 구간별 오차를 정답 진행 방향으로 (뒤처짐 −). 구간 앞뒤 끝 1 s는 뺌
        q = tr[tr["segment"] == g]
        v = np.array([q["east"][-1] - q["east"][0], q["north"][-1] - q["north"][0]])
        v = v / np.hypot(*v)
        sel = (t >= q["t"][0] + 1) & (t <= q["t"][-1] - 1)
        ti = np.searchsorted(tr["t"], t[sel]).clip(0, len(tr) - 1)
        seg_along[g] = float(np.mean(np.c_[r["ex"][sel] - tr["east"][ti], r["ey"][sel] - tr["north"][ti]] @ v))
    return c, dict(seg_along=seg_along, k_78=float(r["k"][np.searchsorted(t, 78.0)]), k_end=m["k_end"], med=m["med_all"], p90=m["p90_all"],
                   b1=m["b1_med"], b2=m["b2_med"], end=m["end"], along=float(np.mean(de @ u)),
                   across=float(np.mean(de @ np.array([-u[1], u[0]]))), kills=f.pf.stats["wall_kills"],
                   k_t=[float(r["k"][np.searchsorted(t, x)]) for x in (5, 20, 40, 60, 78, 104, 118)])


def segments(steps, truth, t0):
    """정답 구간별 거리 vs PDR 보폭 합·걸음 수."""
    out = []
    for g in np.unique(truth["segment"]):
        tr = truth[truth["segment"] == g]
        d = float(np.hypot(np.diff(tr["east"]), np.diff(tr["north"])).sum())
        L = [s["length_m"] for s in steps if tr["t"][0] <= s["t"] - t0 <= tr["t"][-1]]
        out.append(dict(seg=int(g), t0=float(tr["t"][0]), t1=float(tr["t"][-1]), truth_m=round(d, 2), pdr_m=round(sum(L), 2),
                        n=len(L), esc=bool(tr["on_esc"].any())))
    return out


def suspect_table():
    """③(새 기준)에서 의심 구간을 빼면 과신 지표가 어떻게 되는가."""
    z = np.load(paths.OUT / "i1/inav_pdr_pf/runs.npz")
    out = {}
    for k in ("known_raw_기본", "known_raw_중간", "known_tf_기본", "known_tf_중간"):
        rows = []
        for i in range(10):
            r = z[f"{k}_{i}"]
            t = r["t"]
            walk = ~r["esc"] & ~r["riding"] & ~((t > 78.3) & (t < 103.0)) & (t > 2)
            bad = np.zeros(len(t), bool)
            for a, b in SUSPECT:
                bad |= (t >= a) & (t <= b)
            s = lambda sel: dict(ratio=float(np.median(r["err"][sel] / np.maximum(r["unc"][sel], 1e-3))),
                                 cover2=float(np.mean(r["err"][sel] <= 2 * r["unc"][sel])),
                                 in95=float(np.mean(r["err"][sel] <= r["unc95"][sel])),
                                 med=float(np.median(r["err"][sel])), p90=float(np.percentile(r["err"][sel], 90)),
                                 wrong=bool(((r["unc"][sel] < 2) & (r["err"][sel] > 3)).any()))
            wr = (r["unc"] < 2) & (r["err"] > 3) & walk
            rows.append(dict(all=s(walk), clean=s(walk & ~bad), in_bad=float(np.median(r["err"][walk & bad])),
                             wrong_frames=int(wr.sum()), wrong_in_bad=int((wr & bad).sum())))
        med = lambda a, q: float(np.median([x[a][q] for x in rows]))
        out[k] = dict(all={q: med("all", q) for q in ("ratio", "cover2", "in95", "med", "p90")},
                      clean={q: med("clean", q) for q in ("ratio", "cover2", "in95", "med", "p90")},
                      wrong_all=sum(x["all"]["wrong"] for x in rows), wrong_clean=sum(x["clean"]["wrong"] for x in rows),
                      err_in_bad=float(np.median([x["in_bad"] for x in rows])),
                      wrong_frames=int(np.median([x["wrong_frames"] for x in rows])),
                      wrong_frames_in_bad=int(np.median([x["wrong_in_bad"] for x in rows])))
        hist = {}
        for i in range(10):
            r = z[f"{k}_{i}"]
            t = r["t"]
            walk = ~r["esc"] & ~r["riding"] & ~((t > 78.3) & (t < 103.0))
            wr = t[(r["unc"] < 2) & (r["err"] > 3) & walk]
            for b, n in zip(*np.unique((wr // 5 * 5).astype(int), return_counts=True)):
                hist[int(b)] = hist.get(int(b), 0) + int(n)
        out[k]["wrong_hist_5s"] = dict(sorted(hist.items()))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    obs_sets = [("none", "기본"), ("raw", "기본"), ("raw", "중간"), ("tf", "기본"), ("tf", "중간")]
    conds = [dict(steps=st, wall=w, obs=o, set=s, k=k, seed=sd)
             for st in ("pdr", "ideal") for w in (True, False) for o, s in obs_sets for k in ("free", "reset", "fixed")
             for sd in range(10)]
    conds_d = [dict(steps="pdr", wall=True, obs=o, set=s, k=k, seed=sd, sc="drift")
               for o, s in obs_sets for k in ("free", "reset") for sd in range(10)]
    with Pool(10, _init) as pool:
        res = pool.map(_job, conds, chunksize=5)
        res_d = pool.map(_job, conds_d, chunksize=5)
    _init()
    agg, agg_d = {}, {}
    for c, m in res_d:
        agg_d.setdefault((c["obs"], c["set"], c["k"]), []).append(m)
    for c, m in res:
        agg.setdefault((c["steps"], c["wall"], c["obs"], c["set"], c["k"]), []).append(m)
    rows = []
    for (st, w, o, s, k), ms in agg.items():
        d = {q: float(np.median([m[q] for m in ms])) for q in ("k_78", "k_end", "med", "p90", "b1", "b2", "end", "along", "across", "kills")}
        d["k_t"] = np.median([m["k_t"] for m in ms], axis=0).round(3).tolist()
        d["seg_along"] = {g: round(float(np.median([m["seg_along"][g] for m in ms])), 2) for g in ms[0]["seg_along"]}
        rows.append(dict(steps=st, wall=w, obs=o, set=s, k=k, **d))
    drift_rows = [dict(obs=o, set=s, k=k, **{q: float(np.median([m[q] for m in ms])) for q in ("k_end", "med", "p90", "b2", "end")})
                  for (o, s, k), ms in agg_d.items()]
    segs = dict(pdr=segments(_G["steps"]["pdr"], _G["truth"], _G["t0"]),
                ideal=segments(_G["steps"]["ideal"], _G["truth"], _G["t0"]))
    sus = suspect_table()
    out = paths.OUT / "i1"
    (out / "diag_k_end.json").write_text(json.dumps(dict(rows=rows, drift=drift_rows, segments=segs, suspect=sus), ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    OB = {("none", "기본"): "끔", ("raw", "기본"): "없음·기본", ("raw", "중간"): "없음·중간", ("tf", "기본"): "시간·기본", ("tf", "중간"): "시간·중간"}
    L = ["# 끝 k·끝 오차 진단", "", "seed 10개 중앙, N 3000, 출발점 앎. 녹화 1회(손에 든 촬영). 끝 오차 앞뒤 = 마지막 구간 진행 방향(앞 +, 뒤 −).", "",
         "## k와 위치 (k 그대로)", "", "| 걸음 | 벽 | 비전 | k 78 s | 끝 k | 오차 중앙 / 90% | B2 중앙 | 끝 오차 | 끝 앞뒤 / 옆 |", "|---|---|---|---|---|---|---|---|---|"]
    def row(r):
        return (f"| {'indoornav PDR' if r['steps'] == 'pdr' else '이상적'} | {'있음' if r['wall'] else '없음'} | {OB[(r['obs'], r['set'])]} | {r['k_78']:.2f} | {r['k_end']:.2f} | "
                f"{r['med']:.1f} / {r['p90']:.1f} | {r['b2']:.1f} | {r['end']:.1f} | {r['along']:+.1f} / {r['across']:+.1f} |")
    for r in rows:
        if r["k"] == "free":
            L.append(row(r))
    L += ["", "## k 처리 바꿈 (indoornav PDR, 벽 있음)", "", "| 비전 | k | 끝 k | 오차 중앙 / 90% | B2 중앙 | 끝 오차 | 끝 앞뒤 / 옆 |", "|---|---|---|---|---|---|---|"]
    KN = {"free": "그대로", "reset": "하차 때 1", "fixed": "1로 고정"}
    for r in rows:
        if r["steps"] == "pdr" and r["wall"]:
            L.append(f"| {OB[(r['obs'], r['set'])]} | {KN[r['k']]} | {r['k_end']:.2f} | {r['med']:.1f} / {r['p90']:.1f} | {r['b2']:.1f} | {r['end']:.1f} | {r['along']:+.1f} / {r['across']:+.1f} |")
    L += ["", "## PDR 틀어짐에서 하차 때 k = 1 (indoornav PDR, 벽 있음)", "", "| 비전 | k | 끝 k | 오차 중앙 / 90% | B2 중앙 | 끝 오차 |", "|---|---|---|---|---|---|"]
    for r in drift_rows:
        L.append(f"| {OB[(r['obs'], r['set'])]} | {KN[r['k']]} | {r['k_end']:.2f} | {r['med']:.1f} / {r['p90']:.1f} | {r['b2']:.1f} | {r['end']:.1f} |")
    L += ["", "## 구간별 진행 방향 오차 (m, 뒤처짐 −, k 그대로)", "", "| 걸음 | 벽 | 비전 | 구간 1 | 2 | 3 | 4 | 5 | 7 | 8 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["k"] == "free":
            L.append(f"| {'indoornav PDR' if r['steps'] == 'pdr' else '이상적'} | {'있음' if r['wall'] else '없음'} | {OB[(r['obs'], r['set'])]} | " + " | ".join(f"{r['seg_along'][g]:+.1f}" for g in (1, 2, 3, 4, 5, 7, 8)) + " |")
    L += ["", "## 구간별 거리", "", "| 구간 | 시각 (s) | 정답 (m) | PDR 보폭 합 (m) | 걸음 |", "|---|---|---|---|---|"]
    for g in segs["pdr"]:
        L.append(f"| {g['seg']}{' 에스컬레이터' if g['esc'] else ''} | {g['t0']:.1f}~{g['t1']:.1f} | {g['truth_m']} | {g['pdr_m']} | {g['n']} |")
    L += ["", "## 의심 구간(8~10 s, 77~78 s)을 뺀 과신 지표 (③ 새 기준)", "",
          "| 조건 | 구간 안 오차 | 자신 있게 틀림 전→후 | 오차÷불확실도 전→후 | 오차≤2×불확실도 전→후 | unc95 전→후 | 틀림 프레임 중 구간 안 |", "|---|---|---|---|---|---|---|"]
    for k, v in sus.items():
        a, b = v["all"], v["clean"]
        L.append(f"| {k} | {v['err_in_bad']:.1f} | {v['wrong_all']}/10 → {v['wrong_clean']}/10 | {a['ratio']:.2f} → {b['ratio']:.2f} | {100 * a['cover2']:.0f}% → {100 * b['cover2']:.0f}% | "
                 f"{100 * a['in95']:.0f}% → {100 * b['in95']:.0f}% | {v['wrong_frames_in_bad']} / {v['wrong_frames']} |")
    L += ["", "## 자신 있게 틀림 프레임이 나온 시각 (seed 10개 합, 5 s 묶음 시작 시각: 프레임 수)", ""]
    for k, v in sus.items():
        L.append(f"- {k}: " + ", ".join(f"{b} s: {n}" for b, n in v["wrong_hist_5s"].items()))
    (out / "diag_k_end.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
