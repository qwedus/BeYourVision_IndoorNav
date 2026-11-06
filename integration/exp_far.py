"""20 m 넘는 관측이 정답 위치에서 지도와 맞는지 잰다 (관측 거리 한계 비교용). PF는 안 돌림.

    I1_RECORDING=<이름> .venv/Scripts/python integration/exp_far.py --station <40 m 표 패키지 폴더>
→ <녹화 out>/exp_d58_r40/far.json

시간 필터 관측, 정답 위치 + 카메라 방향(자이로 + 곧은 구간 중앙값). 화면 끝 검출은 뺌(박스 크기로 거리를 못 재서).
후보 = 같은 종류, 그 자리 가시성 표에서 보이는 면. 거리대: 20 m 안 / 20~30 m / 30~40 m.
관측 하나를 이렇게 나눔
    near      20 m 안 후보와 10° 안 (지금 한계로도 비교됨)
    far       20 m 안에는 없고 20~40 m 후보와 10° 안 (한계를 늘려야 비교됨)
    none      40 m 안 어떤 후보와도 10° 밖
far 관측을 박스 크기로 다시 나눔: 판 세로 크기로 낸 거리 ÷ 맞은 판까지 거리
    맞음      0.67~1.5배 (맞는 판을 본 것으로 봄)
    다른 판   그 밖 (같은 종류 다른 판이 우연히 그 방위에 있었을 가능성. 박스가 판 일부만 잡아도 여기 들어감)
우연 비율: 방위를 ±30·±45·±60° 돌렸을 때 far 조건에 드는 비율 (이 값만큼은 우연히도 맞음)
틀린 자리 비율: 같은 관측을 정답에서 진행 방향 ±5·±10 m(보행 칸일 때만) 옮긴 자리에서 비교. 거리 한계 20·30·40 m마다
    후보 중 하나와 10° 안이면 맞음으로 셈. 정답 자리 비율과 벌어질수록 위치를 가리는 힘이 있음. 판 위치·카메라 방향 오차는 그대로 들어감
"""
import argparse
import json
import sys

import numpy as np

import paths
import replay as rp
from analyze_recording import camera_offset
from edge_faces import frame_obs
from obs_model import _cells, kind_types
from station_pkg import load_station
from time_filter import load_frames

TOL = 10.0
SHIFTS = (-60, -45, -30, 30, 45, 60)


def diffs(fm, kinds, r, az):
    """→ 거리, 방위 차, 행 번호 (같은 종류·그 칸에서 보이는 면)"""
    i, j, ins = _cells(fm, np.array([r["east"]]), np.array([r["north"]]))
    if not ins[0]:
        return np.array([]), np.array([]), np.array([], int)
    d = fm["dist"][:, i[0], j[0]].astype(float)
    k = np.where(np.isin(fm["lm_type"], kinds) & np.isfinite(d))[0]
    dx, dy = fm["lm_xy"][k, 0] - r["east"], fm["lm_xy"][k, 1] - r["north"]
    return d[k], (az - np.degrees(np.arctan2(dx, dy)) + 180) % 360 - 180, k


def in_range(d, df, rng):
    return bool(len(d) and (np.abs(df[d <= rng]) <= TOL).any())


def walkable(fm, e, n):
    i, j, ins = _cells(fm, np.array([e]), np.array([n]))
    return bool(ins[0] and fm["walk_pf"][i[0], j[0]])


def classify(d, df):
    if len(d) and (np.abs(df[d <= 20]) <= TOL).any():
        return "near"
    m = (d > 20) & (np.abs(df) <= TOL)
    return "far" if m.any() else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")
    fmap = load_station(a.station)
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
    off = camera_offset(truth, paths.RC["waypoints"], ty, yaw)
    rows, chance, far = {}, {}, []
    disc = {str(q): dict(true=[0, 0], d5=[0, 0], d10=[0, 0]) for q in (20, 30, 40)}
    for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1]:
        t, obs = frame_obs(f, t0)
        if paths.excluded(np.array([t]))[0] or t < 0 or t > truth["t"][-1]:
            continue
        r = rp.truth_at(truth, t)
        if r["on_esc"]:
            continue
        cam = (np.interp(t, ty, yaw) + off) % 360
        fm = fmap[int(r["floor"])]
        box = {d["detection_id"]: d["box_xyxy"] for d in f["detections"]}
        fx = f["frame_info"]["intrinsics"]["fx"]
        for o in obs:
            if o["edge"]:
                continue
            kinds = kind_types(o)
            az = (cam + o["bearing"]) % 360
            d, df, k = diffs(fm, kinds, r, az)
            c = classify(d, df)
            for q in (20, 30, 40):
                disc[str(q)]["true"][0] += in_range(d, df, q)
                disc[str(q)]["true"][1] += 1
            hd = np.radians(r["heading_deg"])
            for sh, nm in ((5, "d5"), (-5, "d5"), (10, "d10"), (-10, "d10")):
                e2, n2 = r["east"] + sh * np.sin(hd), r["north"] + sh * np.cos(hd)
                if walkable(fm, e2, n2):
                    d3, df3, _ = diffs(fm, kinds, dict(east=e2, north=n2), az)
                    for q in (20, 30, 40):
                        disc[str(q)][nm][0] += in_range(d3, df3, q)
                        disc[str(q)][nm][1] += 1
            g = rows.setdefault(o["label"], dict(near=0, far=0, none=0, far_ok=0, far_other=0, far_multi=0, far_20_30=0, far_30_40=0))
            g[c] += 1
            for s in SHIFTS:
                ch = chance.setdefault(o["label"], [0, 0])
                ch[1] += 1
                d2, df2, _ = diffs(fm, kinds, r, (az + s) % 360)
                ch[0] += classify(d2, df2) == "far"
            if c == "far":
                m = np.where((d > 20) & (np.abs(df) <= TOL))[0]
                j = m[int(np.argmin(np.abs(df[m])))]
                b = box[o["detection_id"]]
                hgt = fm["lm_json"][int(k[j])]["size"][1]
                ratio = fx * hgt / max(b[3] - b[1], 1.0) / d[j]
                ok = 0.67 <= ratio <= 1.5
                g["far_ok" if ok else "far_other"] += 1
                g["far_multi"] += len(m) > 1
                g["far_20_30" if d[j] <= 30 else "far_30_40"] += 1
                far.append(dict(t=float(t), label=o["label"], face=str(fm["lm_id"][int(k[j])]), d=float(d[j]), res=float(df[j]), ratio=float(ratio)))
    res = dict(disc=disc, by_label=rows, chance={k: v[0] / max(v[1], 1) for k, v in chance.items()}, far=far)
    (paths.REC_OUT / "exp_d58_r40/far.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {k: sum(v[k] for v in rows.values()) for k in ("near", "far", "none", "far_ok", "far_other")}
    print(paths.RECORDING, n, flush=True)


if __name__ == "__main__":
    main()
