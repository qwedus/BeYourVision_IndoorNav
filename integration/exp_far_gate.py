"""조건부 먼 후보(exp_cause --far-gate)를 정답 자리에서 잰다. PF는 안 돌림.

    I1_RECORDING=<이름> .venv/Scripts/python integration/exp_far_gate.py
→ <녹화 out>/exp_d58_g30/far_gate.json

시간 필터 관측, 화면 안 검출. 정답 위치 + 카메라 방향.
먼 후보 = 같은 종류, 그 자리에서 보이는 면, 20~30 m, 화각 ±35° 안 (파티클이 정답에 있을 때 후보가 되는 면).
통과 = 크기 조건(±20·30·50%) 통과. 맞는 판 = 통과 후보 중 관측 방위와 10° 안.
자리 비교: 정답, 진행 방향 ±5·±10 m(보행 칸일 때만)에서 후보와 10° 안 비율을 20 m / 30 m / 조건부로.
"""
import json
import sys

import numpy as np

import paths
import replay as rp
from analyze_recording import camera_offset
from edge_faces import frame_obs
from exp_cause import EDGE_PX, LONG_W
from obs_model import _cells, kind_types
from station_pkg import load_station
from time_filter import load_frames

TOL = 10.0
TOLS = (0.2, 0.3, 0.5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")
    fmap = load_station(paths.INPUTS)
    size = {fl: (np.array([d["size"][1] for d in f["lm_json"]], float), np.array([d["size"][0] for d in f["lm_json"]], float)) for fl, f in fmap.items()}
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
    off = camera_offset(truth, paths.RC["waypoints"], ty, yaw)
    far = dict(cand=0, cand_ok_bearing=0, no_size=0, long=0)
    passed = {str(q): [0, 0] for q in TOLS}
    keys = ["r20", "r30"] + [f"g{int(q * 100)}" for q in TOLS]
    place = {k: dict(true=[0, 0], d5=[0, 0], d10=[0, 0]) for k in keys}
    by_label = {}

    def candidates(fm, fl, kinds, e, n, az):
        i, j, ins = _cells(fm, np.array([e]), np.array([n]))
        if not ins[0]:
            return None
        d = fm["dist"][:, i[0], j[0]].astype(float)
        k = np.where(np.isin(fm["lm_type"], kinds) & np.isfinite(d) & (d <= 30))[0]
        dx, dy = fm["lm_xy"][k, 0] - e, fm["lm_xy"][k, 1] - n
        return k, d[k], (az - np.degrees(np.arctan2(dx, dy)) + 180) % 360 - 180

    def hits(c, fl, f_over_h):
        k, d, df = c
        near = bool(((d <= 20) & (np.abs(df) <= TOL)).any())
        m = (d > 20) & (np.abs(df) <= TOL)
        out = dict(r20=near, r30=near or bool(m.any()))
        for q in TOLS:
            ok = False
            if not near and m.any() and f_over_h is not None:
                H, W = size[fl]
                s = f_over_h * H[k[m]]
                ok = bool(((W[k[m]] < LONG_W) & (np.abs(s / d[m] - 1) <= q)).any())
            out[f"g{int(q * 100)}"] = near or ok
        return out

    for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1]:
        t, obs = frame_obs(f, t0)
        if paths.excluded(np.array([t]))[0] or t < 0 or t > truth["t"][-1]:
            continue
        r = rp.truth_at(truth, t)
        if r["on_esc"]:
            continue
        fl = int(r["floor"])
        fm = fmap[fl]
        cam = (np.interp(t, ty, yaw) + off) % 360
        box = {d["detection_id"]: d["box_xyxy"] for d in f["detections"]}
        fx, IH = f["frame_info"]["intrinsics"]["fx"], f["frame_info"]["image_height"]
        for o in obs:
            if o["edge"]:
                continue
            b = box[o["detection_id"]]
            fh = None if b[1] < EDGE_PX or b[3] > IH - EDGE_PX else fx / max(b[3] - b[1], 1.0)
            kinds = kind_types(o)
            az = (cam + o["bearing"]) % 360
            c = candidates(fm, fl, kinds, r["east"], r["north"], az)
            if c is None:
                continue
            k, d, df = c
            H, W = size[fl]
            q0 = (d > 20) & (np.abs(((cam - az + df + 180) % 360) - 180) <= 35)     # 파티클 방향 기준 화각 안
            g = by_label.setdefault(o["label"], dict(cand=0, ok=0, **{f"pass{int(q * 100)}": 0 for q in TOLS}, **{f"right{int(q * 100)}": 0 for q in TOLS}))
            for kk, dd, ee in zip(k[q0], d[q0], df[q0]):
                far["cand"] += 1
                g["cand"] += 1
                right = abs(ee) <= TOL
                far["cand_ok_bearing"] += right
                g["ok"] += right
                if W[kk] >= LONG_W:
                    far["long"] += 1
                    continue
                if fh is None:
                    far["no_size"] += 1
                    continue
                s = fh * H[kk]
                for q in TOLS:
                    if abs(s / dd - 1) <= q:
                        passed[str(q)][0] += 1
                        passed[str(q)][1] += right
                        g[f"pass{int(q * 100)}"] += 1
                        g[f"right{int(q * 100)}"] += right
            for nm, v in hits(c, fl, fh).items():
                place[nm]["true"][0] += v
                place[nm]["true"][1] += 1
            hd = np.radians(r["heading_deg"])
            for sh, tag in ((5, "d5"), (-5, "d5"), (10, "d10"), (-10, "d10")):
                e2, n2 = r["east"] + sh * np.sin(hd), r["north"] + sh * np.cos(hd)
                i, j, ins = _cells(fm, np.array([e2]), np.array([n2]))
                if not (ins[0] and fm["walk_pf"][i[0], j[0]]):
                    continue
                c2 = candidates(fm, fl, kinds, e2, n2, az)
                for nm, v in hits(c2, fl, fh).items():
                    place[nm][tag][0] += v
                    place[nm][tag][1] += 1
    (paths.REC_OUT / "exp_d58_g30/far_gate.json").write_text(json.dumps(dict(far=far, passed=passed, place=place, by_label=by_label), ensure_ascii=False, indent=1, default=int), encoding="utf-8")
    print(paths.RECORDING, far, passed, flush=True)


if __name__ == "__main__":
    main()
