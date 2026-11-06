"""판 위치 확인: 시간 필터 관측을 시선(반직선)으로 모아 판별 시선 각도 분포·교차 위치 추정을 낸다. 지도는 안 고침.

    .venv/Scripts/python integration/exp_signs.py dump     기록마다 시선 → integration/out/rec/_exp_cause/rays.json
    .venv/Scripts/python integration/exp_signs.py locate   판별 분포·추정 → integration/out/rec/_exp_cause/signs.json

시선 = 정답 위치 + 카메라 방향(자이로 + 곧은 구간 중앙값, analyze_recording과 같음) + 관측 방위. 화면 끝 검출은 뺌.
판에 붙이는 규칙은 판별 잔차와 같음: 같은 종류 20 m 안, 방위 차가 가장 작은 판, 15° 안.
불확실 범위: 기록마다 카메라 기준값에 N(0, 2°), 정답 위치에 N(0, 0.7 m)를 더해 다시 푼 300번의 표준편차.
판 위치에 붙이는 조건에 지도 위치가 들어가므로, 지도와 15° 넘게 다른 위치는 이 방법으로 찾지 못한다.
"""
import json
import os
import subprocess
import sys

import numpy as np

import paths

OUT = paths.OUT / "rec/_exp_cause"          # 저장소에 안 넣음
RECS = ["route2_1030", "s2_b_ex2_31", "s2_b_pillar_map", "s2_b_elev_pillar", "s2_b_emart_ex5", "s2_a_71_ex2", "s2_a_platform",
        "s2_a_ex3_ex6", "s2_a_ex6_ex3"]
FACES = ["B01_ticket_sign_85", "B01_ticket_sign_87", "B01_exit_sign_14", "B01_exit_sign_15", "B01_track_sign_28"]
SD_OFF, SD_POS, N_MC = 2.0, 0.7, 300
TOL, RANGE = 15.0, 20.0


def dump_one():
    import analyze_recording as ar
    import replay as rp
    from edge_faces import add_face_points, frame_obs
    from obs_model import kind_types
    from station_pkg import load_station
    from time_filter import load_frames
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")
    fmap = load_station(paths.INPUTS)
    R, T = fmap[-1]["R"], fmap[-1]["T"]
    fp = add_face_points(fmap)
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
    off = ar.camera_offset(truth, paths.RC["waypoints"], ty, yaw)
    rays = []
    for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1]:
        t, obs = frame_obs(f, t0)
        if paths.excluded(np.array([t]))[0] or t < 0 or t > truth["t"][-1]:
            continue
        r = rp.truth_at(truth, t)
        if r["on_esc"]:
            continue
        cam = (np.interp(t, ty, yaw) + off) % 360
        fm = fp[int(r["floor"])]
        by_id = {o["detection_id"]: o for o in f["observations"]}
        boxes = {d["detection_id"]: d["box_xyxy"] for d in f.get("detections", [])}
        for o in obs:
            if o["edge"]:
                continue
            az = (cam + o["bearing"]) % 360
            dx, dy = fm["lm_xy"][:, 0] - r["east"], fm["lm_xy"][:, 1] - r["north"]
            d = np.hypot(dx, dy)
            k = np.where(np.isin(fm["lm_type"], kind_types(o)) & (d < RANGE))[0]
            if not len(k):
                continue
            diff = (az - np.degrees(np.arctan2(dx[k], dy[k])) + 180) % 360 - 180
            if np.abs(diff).min() > TOL:
                continue
            j = k[int(np.argmin(np.abs(diff)))]
            p = (np.array([r["east"], r["north"]]) - T) @ R
            v = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))]) @ R
            rays.append(dict(rec=paths.RECORDING, t=float(t), face=str(fm["lm_id"][j]), label=o["label"], floor=int(r["floor"]),
                             p=[float(p[0]), float(p[1])], az_st=float(np.degrees(np.arctan2(v[0], v[1]))), res=float(diff[np.argmin(np.abs(diff))]),
                             dist=float(d[j]), box=boxes.get(o.get("detection_id")), fx=f["frame_info"]["intrinsics"]["fx"], bearing=float(o["bearing"])))
    return rays


def solve(p, az, x0):
    """방위 차 제곱합이 최소인 점 (가우스-뉴턴). p (n,2) 관측 위치, az 역 좌표 나침반 방위 [도]"""
    x = np.array(x0, float)
    a = np.radians(az)
    for _ in range(25):
        d = x - p
        r2 = np.maximum((d ** 2).sum(1), .25)
        e = (np.arctan2(d[:, 0], d[:, 1]) - a + np.pi) % (2 * np.pi) - np.pi
        J = np.c_[d[:, 1] / r2, -d[:, 0] / r2]
        A = J.T @ J + 1e-9 * np.eye(2)
        s = np.linalg.solve(A, -J.T @ e)
        x = x + np.clip(s, -2, 2)
        if np.abs(s).max() < 1e-3:
            break
    return x


def one_per_sec(rays):
    """같은 기록에서 1 s 안의 시선은 거의 같은 정보라 1 s당 1개로 줄임(중앙값)"""
    out = {}
    for r in rays:
        out.setdefault((r["rec"], int(r["t"])), []).append(r)
    res = []
    for (rec, _), v in out.items():
        res.append(dict(rec=rec, p=np.median([x["p"] for x in v], 0), az=float(np.median([x["az_st"] for x in v])), dist=float(np.median([x["dist"] for x in v]))))
    return res


def recs_all(rec):
    return sorted(set(rec.tolist()))


def locate():
    from station_pkg import load_station
    fm = load_station(paths.INPUTS)[-1]
    ids = list(map(str, fm["lm_id"]))
    all_rays = json.loads((OUT / "rays.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    res = {}
    for fid in FACES:
        rs = [r for r in all_rays if r["face"] == fid]
        i = ids.index(fid)
        m = fm["lm_st"][i]
        js = fm["lm_json"][i]
        if not rs:
            res[fid] = dict(n=0)
            continue
        q = one_per_sec(rs)
        P = np.array([x["p"] for x in q])
        A = np.array([x["az"] for x in q])
        rec = np.array([x["rec"] for x in q])
        sight = (np.degrees(np.arctan2(*(m - P).T)) + 360) % 360       # 관측 위치에서 지도 판을 보는 방향
        c = np.degrees(np.angle(np.exp(1j * np.radians(sight)).mean()))
        hist, edges = np.histogram(sight, bins=np.arange(0, 361, 30))    # 나침반식 역 좌표 방향, 30° 칸
        by_rec = {r: [float(np.percentile(sight[rec == r], 10)), float(np.percentile(sight[rec == r], 90)), np.round(np.median(P[rec == r], 0), 1).tolist()] for r in recs_all(rec)}
        est = solve(P, A, m)
        recs = sorted(set(rec))
        mc = []
        for _ in range(N_MC):
            o = {r: rng.normal(0, SD_OFF) for r in recs}
            s = {r: rng.normal(0, SD_POS, 2) for r in recs}
            mc.append(solve(P + np.array([s[r] for r in rec]), A + np.array([o[r] for r in rec]), m))
        mc = np.array(mc)
        cov = np.cov(mc.T)
        ev, U = np.linalg.eigh(cov)
        lo = {}                            # 기록 하나 뺀 값
        for r in recs:
            if (rec != r).sum() >= 3:
                lo[r] = solve(P[rec != r], A[rec != r], m).round(2).tolist()
        res[fid] = dict(n_raw=len(rs), n_sec=len(q), recs={r: int((rec == r).sum()) for r in recs}, map_st=np.round(m, 2).tolist(), normal=js.get("normal"), width=js.get("size", [None])[0],
                        sight_center=float(c), sight_hist=dict(edges=edges.tolist(), n=hist.tolist()), sight_by_rec=by_rec,
                        dist_med=float(np.median([x["dist"] for x in q])),
                        est=np.round(est, 2).tolist(), est_minus_map=np.round(est - m, 2).tolist(), sd=np.round(np.sqrt(np.diag(cov)), 2).tolist(),
                        sd_long=float(np.sqrt(ev[1])), sd_short=float(np.sqrt(ev[0])), long_dir=np.round(U[:, 1], 2).tolist(), leave_one_out=lo,
                        rms_res=float(np.sqrt(np.mean((((np.degrees(np.arctan2(*(est - P).T)) - A + 180) % 360 - 180)) ** 2))))
    (OUT / "signs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False)[:900])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "one":
        print(json.dumps(dump_one(), ensure_ascii=False))
    elif cmd == "dump":
        rays = []
        for r in RECS:
            p = subprocess.run([sys.executable, __file__, "one"], env={**os.environ, "I1_RECORDING": r, "PYTHONIOENCODING": "utf-8"}, capture_output=True, text=True, encoding="utf-8")
            if p.returncode:
                raise SystemExit(p.stderr[-1500:])
            rays += json.loads(p.stdout)
        (OUT / "rays.json").write_text(json.dumps(rays, ensure_ascii=False), encoding="utf-8")
        print(len(rays))
    else:
        locate()


if __name__ == "__main__":
    main()
