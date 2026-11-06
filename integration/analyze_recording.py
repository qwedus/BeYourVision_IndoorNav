"""재생 결과 추가 집계: 관측이 지도와 맞는 비율, 확정 경로 대비 PDR 거리, 크게 틀린 구간.

    I1_RECORDING=<이름> .venv/Scripts/python integration/analyze_recording.py [--name replay]
→ <녹화 설정 out>/<name>/analysis.json

- 관측 맞음: 정답 위치 + 카메라 방향에서 같은 종류 랜드마크 20 m 안 중 가장 가까운 방위와의 차이가 10° 안(에스컬레이터 구간 제외,
  화면 끝 검출은 판 끝점과 비교). 카메라 방향 = 자이로 방향 + 곧게 걷는 구간(기준점 앞뒤 3 s 뺌)에서 '정답 방향 - 자이로 방향'의 중앙값.
  사원수 방향은 자기장 보정으로 긴 녹화에서 흘러 쓰지 않음.
- 판별 잔차: 화면 안 관측을 같은 조건에서 가장 가까운 판(15° 안)에 붙여, 정답 위치에서 본 '관측 방위 - 지도 방위'와 옆 오차(거리 × tan).
  옆 오차 벡터는 역 좌표. 카메라 방향 기준값이 3° 틀리면 15 m에서 약 0.8 m 틀어짐.
- PDR 거리: indoornav 이동 판정 걸음의 보폭 합(에스컬레이터 구간 제외) ÷ 정답 궤적 길이. 구간별로도 냄.
- 크게 틀린 구간: seed 10개 중앙 오차가 문턱(3 m)을 넘는 구간(2 s 안 끊김은 이음). 가까운 기준점(시각 5 s 안 또는 위치 3 m 안)이 추정인지 표시.
PF·관측 모델·비전 설정은 안 바꿈. 지표만 계산함.
"""
import argparse
import json
import sys

import numpy as np

import paths
import replay as rp
from edge_faces import add_face_points, frame_obs
from obs_model import kind_types
from pdr_steps import read_steps
from station_pkg import load_station
from time_filter import load_frames

TOL, RANGE, FACE_TOL = 10.0, 20.0, 15.0
BIG, GAP = 3.0, 2.0
COND = ("tf_기본", "tf_중간", "raw_기본", "raw_중간", "none")


def camera_offset(truth, wps, ty, yaw):
    """곧게 걷는 구간(기준점 앞뒤 3 s 뺌, 에스컬레이터 뺌)에서 정답 진행 방향 - 카메라 방향의 중앙값"""
    t = truth["t"]
    tw = np.array([w["t"] for w in wps])
    ok = (np.min(np.abs(t[:, None] - tw[None]), 1) > 3) & (truth["on_esc"] == 0) & (t > 1)
    h = np.interp(t[ok], ty, yaw)
    return float(np.degrees(np.median(np.angle(np.exp(1j * np.radians(truth["heading_deg"][ok] - h))))))


def obs_match(path, fmap, truth, t0, ty, yaw, off, R=None, T=None):
    rows, faces = {}, {}
    for f in load_frames(path)[1]:
        t, obs = frame_obs(f, t0)
        if paths.excluded(np.array([t]))[0] or t < 0 or t > truth["t"][-1]:
            continue
        r = rp.truth_at(truth, t)
        if r["on_esc"]:
            continue
        cam = (np.interp(t, ty, yaw) + off) % 360
        fm = fmap[int(r["floor"])]
        for o in obs:
            az = (cam + o["bearing"]) % 360
            dx, dy = fm["lm_xy"][:, 0] - r["east"], fm["lm_xy"][:, 1] - r["north"]
            d = np.hypot(dx, dy)
            k = np.where(np.isin(fm["lm_type"], kind_types(o)) & (d < RANGE))[0]
            diff = (az - np.degrees(np.arctan2(dx[k], dy[k])) + 180) % 360 - 180
            ok = len(k) > 0 and np.abs(diff).min() <= TOL
            rows.setdefault(o["label"], []).append(bool(ok))
            if len(k) and not o["edge"] and R is not None and np.abs(diff).min() <= FACE_TOL:
                j = int(np.argmin(np.abs(diff)))
                res, dist = float(diff[j]), float(d[k[j]])
                sight = np.array([dx[k[j]], dy[k[j]]]) / max(dist, 1e-6)
                # 나침반 시계 + 방위 차 → 시선의 오른쪽(동·북에서 (n, -e))으로 옆 오차
                side = np.array([sight[1], -sight[0]]) * dist * np.tan(np.radians(res))
                faces.setdefault(str(fm["lm_id"][k[j]]), []).append(
                    (res, dist, *(side @ R), *(sight @ R), t))
    n = sum(len(v) for v in rows.values())
    fc = {}
    for i, v in faces.items():
        a = np.array(v)
        fc[i] = dict(n=len(v), res_med=float(np.median(a[:, 0])), dist_med=float(np.median(a[:, 1])), side_st=[float(np.median(a[:, 2])), float(np.median(a[:, 3]))],
                     sight_st=[float(np.mean(a[:, 4])), float(np.mean(a[:, 5]))], t=[float(a[:, 6].min()), float(a[:, 6].max())])
    return dict(n=n, match=float(np.mean(sum(rows.values(), []))) if n else None,
                by_label={k: dict(n=len(v), match=float(np.mean(v))) for k, v in sorted(rows.items())}, faces=fc)


def pdr_distance(truth, wps, t0):
    steps = read_steps(paths.REC_OUT / "steps_indoornav.csv")
    st = np.array([s["t"] - t0 for s in steps])
    ln = np.array([s["length_m"] for s in steps])
    mv = np.array([s["moving"] for s in steps])
    segs = []
    for a, b in zip(wps[:-1], wps[1:]):
        L = float(np.hypot(*(np.subtract(b["st"], a["st"]))))
        sel = (st >= a["t"]) & (st < b["t"])
        segs.append(dict(a=a["name"], b=b["name"], t0=a["t"], t1=b["t"], esc=bool(a.get("esc")), truth_m=L,
                         pdr_m=float(ln[sel & mv].sum()), steps=int((sel & mv).sum())))
    walk = [s for s in segs if not s["esc"]]
    tl, pl = sum(s["truth_m"] for s in walk), sum(s["pdr_m"] for s in walk)
    return dict(truth_m=tl, pdr_m=pl, ratio=pl / tl if tl else None, pdr_all_m=float(ln[mv].sum()),
                esc_steps=int(sum(s["steps"] for s in segs if s["esc"])), n_moving=int(mv.sum()), n_all=len(steps), segments=segs)


def big_errors(z, key, truth, wps, st_of):
    arr = [z[k] for k in sorted(z.files) if k.rsplit("_", 1)[0] == key]
    if not arr:
        return []
    t = arr[0]["t"]
    med = np.median([a["err"] for a in arr], 0)
    walk = ~arr[0]["esc"] & ~arr[0]["riding"] & ~paths.excluded(t)
    idx = np.where((med > BIG) & walk)[0]
    if not len(idx):
        return []
    groups, a, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[p] > GAP:
            groups.append((a, p))
            a = i
        p = i
    groups.append((a, p))
    out = []
    for a, b in groups:
        k = a + int(np.argmax(med[a:b + 1]))
        tr = rp.truth_at(truth, t[k])
        p = st_of(tr["east"], tr["north"])
        near = []
        for i, w in enumerate(wps):
            if (t[a] - 5 <= w["t"] <= t[b] + 5 or (w["floor"] == int(tr["floor"]) and np.hypot(*(np.subtract(w["st"], p))) <= 3)):
                near.append(dict(i=i, name=w["name"], t=w["t"], status=w.get("status")))
        ex = np.median([x["ex"][k] for x in arr])
        ey = np.median([x["ey"][k] for x in arr])
        out.append(dict(t0=float(t[a]), t1=float(t[b]), peak=float(med[k]), t_peak=float(t[k]), st=[float(p[0]), float(p[1])],
                        floor=int(tr["floor"]), est_st=[float(v) for v in st_of(ex, ey)], unc_at_peak=float(np.median([x["unc"][k] for x in arr])),
                        near=near, near_estimated=any(n["status"] == "추정" for n in near)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="replay")
    ap.add_argument("--station", help="역사 패키지 폴더(기본 vision/inputs). 옛 지도로 집계할 때")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")
    wps = paths.RC["waypoints"]
    fmap0 = load_station(a.station or paths.INPUTS)
    R, T = fmap0[-1]["R"], fmap0[-1]["T"]
    st_of = lambda e, n: (np.array([e, n]) - T) @ R
    fmap = add_face_points(fmap0)
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    gy_t, gy = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
    ty, yaw = gy_t, gy
    off = camera_offset(truth, wps, ty, yaw)
    start_gyro = off % 360                                       # 시작 0° 자이로라 이 값이 곧 자이로로 본 첫 방향
    seg_off = []                                                 # 5 m 넘는 구간별 기준값: 긴 녹화에서 방향이 흐르는지 봄
    for w0, w1 in zip(wps[:-1], wps[1:]):
        if w0.get("esc") or np.hypot(*np.subtract(w1["st"], w0["st"])) < 5 or w1["t"] - w0["t"] < 8:
            continue
        k = (truth["t"] > w0["t"] + 3) & (truth["t"] < w1["t"] - 3)
        if k.any():
            seg_off.append(float(np.degrees(np.angle(np.exp(1j * np.radians(np.median(((truth["heading_deg"][k] - np.interp(truth["t"][k], ty, yaw) - off + 180) % 360) - 180)))))))
    out = paths.REC_OUT / a.name
    z = np.load(out / "runs.npz")
    res = dict(cam_offset=off, start_gyro=start_gyro, seg_off=seg_off, obs_tf=obs_match(paths.I1 / "obs_timefilter.jsonl", fmap, truth, t0, ty, yaw, off, R, T),
               obs_raw=obs_match(paths.I1 / "obs_nofilter.jsonl", fmap, truth, t0, ty, yaw, off),
               pdr=pdr_distance(truth, wps, t0),
               first=dict(heading=float(truth["heading_deg"][0]), name=wps[0]["name"], face=bool(wps[0].get("face")), status=wps[0].get("status")),
               big={k: big_errors(z, f"known_{k}", truth, wps, st_of) for k in COND})
    (out / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(paths.RECORDING, "관측 10° 안 tf %.2f raw %.2f | PDR/정답 %.2f | 큰 오차 구간 %d" % (
        res["obs_tf"]["match"], res["obs_raw"]["match"], res["pdr"]["ratio"], len(res["big"]["tf_기본"])))


if __name__ == "__main__":
    main()
