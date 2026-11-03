"""연결안 7절 1·2단계 검산: 입력(좌표·각도·단위·시각·첫 회전·moving·끝점)과 지도·초기화.

    .venv/Scripts/python integration/check_inputs.py   → integration/out/i1/check_inputs.json (+ 화면 출력)
"""
import json
import sys

import numpy as np

import paths
import replay as rp
from app_session import load_app_session
from edge_faces import add_face_points, SUF_L, SUF_R
from indoornav.pf.filter import ParticleFilter
from station_pkg import load_station, enu_to_aligned, aligned_to_enu, th_to_compass, compass_to_th
from team_filter import TeamFilter, load_maps, team_params
from team_pdr_steps import read_steps


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    rot = smeta["aligned_from_enu_rot_deg"]
    team_meta = json.loads(paths.TEAM_META.read_text(encoding="utf-8"))
    t0 = paths.rec_t0()
    truth = rp.load_truth(paths.I1 / "truth.csv")

    # 1-1 좌표 왕복·알려진 점
    rng = np.random.default_rng(0)
    p = rng.uniform(-100, 100, (1000, 2))
    esc2 = next(e for e in smeta["escalators"] if e["top_key"] == "ESC_P2_B1")
    out["coords"] = dict(
        rot_same_as_team=rot == team_meta["aligned_from_enu_rot_deg"],
        roundtrip_max_m=float(np.abs(aligned_to_enu(enu_to_aligned(p, rot), rot) - p).max()),
        esc_P2_top_aligned_ours=esc2["top_aligned"], esc_P2_top_aligned_team=team_meta["escalators"][0]["top"],
        esc_P2_top_diff_m=float(np.hypot(*(np.array(esc2["top_aligned"]) - team_meta["escalators"][0]["top"]))),
        esc_P2_bottom_diff_m=float(np.hypot(*(np.array(esc2["bottom_aligned"]) - team_meta["escalators"][0]["bottom"]))),
        note="아래 끝은 약 1.5 m 다름(연결안 0절) — 어댑터는 우리 그래프 좌표 사용")
    # 1-2 각도: 나침반 ↔ aligned 수학각
    c = rng.uniform(0, 360, 1000)
    out["angles"] = dict(compass_of_th0=float(th_to_compass(0.0, rot)), compass_of_th90=float(th_to_compass(np.pi / 2, rot)),
                         roundtrip_max_deg=float(np.abs((th_to_compass(compass_to_th(c, rot), rot) - c + 180) % 360 - 180).max()))
    # 1-3 단위·시계
    s, info = load_app_session()
    gn = np.linalg.norm(s.gravity, axis=1)
    out["session"] = dict(fs=s.fs, n=len(s.t), duration_s=float(s.t[-1]), t0_sensor=info["t0_sensor"], t0_video=info["t0_video"],
                          sensor_minus_video_s=info["t0_sensor"] - info["t0_video"], gravity_norm_med=float(np.median(gn)),
                          acc_user_norm_med=float(np.median(np.linalg.norm(s.acc_user, axis=1))),
                          gyro_norm_p99=float(np.percentile(np.linalg.norm(s.gyro, axis=1), 99)))
    # 1-4 걸음·관측·기압 시각
    steps = read_steps(paths.OUT / "i1/steps_team.csv")
    st_t = np.array([x["t"] - t0 for x in steps])
    rides, floors = rp.baro_events(rp.read_baro(paths.REC / "baro.csv", t0))
    base = json.loads((paths.I1 / "baseline_summary.json").read_text(encoding="utf-8"))["info"]
    obs = rp.load_obs(paths.I1 / "obs_timefilter.jsonl", t0)
    ride = lambda t: any(a <= t <= b for a, b in rides)
    mv = np.array([x["moving"] for x in steps])
    out["time"] = dict(steps_t=[float(st_t.min()), float(st_t.max())], obs_t=[obs[0][0], obs[-1][0]],
                       truth_t=[float(truth["t"][0]), float(truth["t"][-1])], rides=rides, floors=floors,
                       baro_same_as_baseline=bool(np.allclose(rides, base["rides"], atol=1e-6)
                                                  and np.allclose(np.array(floors, float), base["floors"], atol=1e-6)))
    # 1-5 첫 회전·moving·방향 부호(정답 진행 방향과 비교)
    tr = np.array([rp.truth_at(truth, t)["heading_deg"] for t in st_t])
    walk = np.array([not rp.truth_at(truth, t)["on_esc"] for t in st_t]) & (st_t > 10)
    comp = truth[0]["heading_deg"] + np.array([x["heading_deg"] for x in steps])     # 나침반 부호 상대 방향 + 출발 방향
    d = (comp - tr + 180) % 360 - 180
    comp_wrong = truth[0]["heading_deg"] - np.array([x["heading_deg"] for x in steps])
    dw = (comp_wrong - tr + 180) % 360 - 180
    out["steps"] = dict(n=len(steps), moving=int(mv.sum()), not_moving=int((~mv).sum()),
                        before_ride=int((st_t < rides[0][0]).sum()), in_ride=int(sum(ride(t) for t in st_t)),
                        after_ride=int((st_t > rides[0][1]).sum()),
                        in_ride_len_m=float(sum(x["length_m"] for x, t in zip(steps, st_t) if ride(t))),
                        first_t=float(st_t[0]), first_dheading_deg=steps[0]["dheading_deg"],
                        dist_m=float(sum(x["length_m"] for x in steps)),
                        heading_vs_truth_med_abs_deg=float(np.median(np.abs(d[walk]))),
                        heading_vs_truth_if_sign_flipped=float(np.median(np.abs(dw[walk]))))
    # 1-6 판 끝점: ENU 그대로, 중심에서 폭/2
    fm = add_face_points(fmap)[-1]
    L = len(fmap[-1]["lm_id"])
    ends = [i for i in range(L, len(fm["lm_id"])) if fm["lm_id"][i].endswith((SUF_L, SUF_R))]
    half = [abs(np.hypot(*(fm["lm_xy"][i] - fmap[-1]["lm_xy"][list(fmap[-1]["lm_id"]).index(fm["lm_id"][i].split("@")[0])]))
                - fm["lm_json"][i]["size"][0] / 2) for i in ends]
    out["face_points"] = dict(faces=[len(f["lm_id"]) for f in fmap.values()], endpoints_B01=len(ends),
                              max_err_half_width_m=float(max(half)))

    # 2 지도·초기화
    maps = load_maps()
    out["map"] = {}
    for fl, name in paths.FLOOR_NAME.items():
        g = maps[fl]
        ys, xs = np.nonzero(g.walk)
        en = aligned_to_enu(g.to_world(xs, ys), rot)
        f = fmap[fl]
        st = (en - f["T"]) @ f["R"]
        j = np.floor((st[:, 0] - f["x0"]) / f["cell"]).astype(int)
        i = np.floor((st[:, 1] - f["y0"]) / f["cell"]).astype(int)
        H, W = f["walk"].shape
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        comp_ = np.zeros(len(st), bool)
        comp_[ok] = f["computed"][i[ok], j[ok]]
        tw = truth[(truth["floor"] == fl) & (truth["on_esc"] == 0)]
        ta = enu_to_aligned(np.c_[tw["east"], tw["north"]], rot)
        out["map"][name] = dict(walk_in_computed=float(comp_.mean()), truth_in_mask=float(g.free(ta).mean()),
                                **json.loads(paths.MASK_META.read_text(encoding="utf-8"))["noroom"]["report"][name])
    P = team_params(3000)
    r0 = truth[0]
    F = TeamFilter(P, maps, fmap, smeta, -1, [r0["east"], r0["north"]], float(r0["heading_deg"]), 0)
    xy0 = enu_to_aligned([r0["east"], r0["north"]], rot)[0]
    dd = np.hypot(*(F.pf.xy - xy0).T)
    raw = ParticleFilter(team_params(3000, init_pos_sigma=2.0), maps[-1], seed=0)
    raw.init(xy0, 0.0)
    out["init"] = dict(at_start_point=float((dd < 1e-9).mean()), r_med=float(np.median(dd)), r_p90=float(np.percentile(dd, 90)),
                       all_free=bool(maps[-1].free(F.pf.xy).all()),
                       heading_sd_deg=float(np.degrees(np.std(F.pf.th))), k_range=[float(F.pf.k.min()), float(F.pf.k.max())],
                       compass0_truth=float(r0["heading_deg"]), compass0_mean=float(F.estimate()["compass"]),
                       team_init_at_start_point_sigma2=float((np.hypot(*(raw.xy - xy0).T) < 1e-9).mean()))
    (paths.OUT / "i1").mkdir(parents=True, exist_ok=True)
    (paths.OUT / "i1/check_inputs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                    encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
