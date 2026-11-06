"""I1 재생 실행기 (indoornav PF): 조건마다 seed 10개 → 지표 요약 (연결안 7절 3~5단계).

    .venv/Scripts/python integration/run_i1.py --steps indoornav --name inav_pdr_pf            # ③ indoornav PDR + PF
    .venv/Scripts/python integration/run_i1.py --steps simple --name simple_pdr_inav_pf   # ② 임시 PDR + indoornav PF
    .venv/Scripts/python integration/run_i1.py --steps indoornav --sigma-h0 1.5 --name inav_h15  # 5단계 민감도
    --bearing-fix theta|bearing  걸음 사이 회전 보정, --track-limit N  같은 track N초에 1번 (설명: 작업기록 3차)
    --adopted  채택한 비전 설정(integration/vision_options.json)으로. 아래 옵션을 주면 그 값이 이김
    --cls-th exit=0.3,monitor=0.3,...  종류별 문턱값 (관측 프레임에 건 뒤 시간 필터를 다시 돌림)
    --text-check <결정 파일>  글자 확인(vision/inputs/i1/ 안 파일 이름, 끄려면 안 줌), --unread keep|drop  못 읽은 관측 넣음·뺌
녹화: 환경 변수 I1_RECORDING(integration/recordings/<이름>.json, 없으면 1차 녹화). 한 번에 돌리기는 run_recording.py
출력: <녹화 설정 out, 1차 녹화 integration/out/i1>/<name>/summary.json·summary.md (runs.npz는 저장소에 안 넣음)

조건 (기준 run_i1.py와 같은 이름)
    known : 출발점(정답 t=0) ±2 m를 방 뺀 보행 마스크로 자른 분포, 방향 ±20°
    drift : known + PDR 일부러 틀어짐(보폭 ×1.15, 방향 0.4°/걸음)
    관측  : none(비전 끔) / raw(필터 없음) / tf(시간 필터), 설정 기본(5 Hz·σ 5°·바닥값 0.1)·중간(2 Hz·σ 10°·바닥값 0.1)
    비전 설정 고정: 검출 문턱 0.25, 시간 필터 5프레임 중 3번, 화면 끝 판 끝점 비교, vision/inputs의 랜드마크 지도 (관측 파일·역사 패키지 그대로)
    같은 seed = 같은 초기 파티클·같은 입력(비전 끔/켬 비교)
"""
import argparse
import json
import sys
import time

import numpy as np

import paths
import provenance as record
from pool_guard import run_pool
import replay as rp

SETTINGS = {"기본": dict(obs_params=None, obs_hz=None),
            "중간": dict(obs_params={"sigma_bearing": 10.0, "eps": 0.1}, obs_hz=2.0)}
DRIFT = dict(len_scale=1.15, head_per_step=0.4)
SCENARIOS = ("known", "drift")
STEPS = {"indoornav": paths.REC_OUT / "steps_indoornav.csv", "simple": paths.I1 / "steps_simple.csv"}
_G = {}


def _init(steps_kind, n, sigma_h0_deg, opt=None):
    from station_pkg import load_station
    from edge_faces import add_face_points
    from vision_filter import load_maps, pf_params
    from pdr_steps import read_steps
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    t0 = paths.rec_t0()
    opt = opt or {}
    dec = {}
    if opt.get("text_check"):
        from text_check import load_decisions
        dec = dict(decisions=load_decisions(paths.I1 / opt["text_check"])[0], unread=opt.get("unread") or "keep")
    if opt.get("cls_th"):
        from time_filter import apply, load_frames
        frames = rp.threshold_frames(load_frames(paths.I1 / "obs_nofilter.jsonl")[1], opt["cls_th"])
        obs = {"raw": rp.frames_obs(frames, t0, **dec), "tf": rp.frames_obs(apply(frames), t0, **dec)}
    else:
        obs = {"raw": rp.load_obs(paths.I1 / "obs_nofilter.jsonl", t0, **dec),
               "tf": rp.load_obs(paths.I1 / "obs_timefilter.jsonl", t0, **dec)}
    _G.update(fmap=add_face_points(fmap), smeta=smeta, maps=load_maps(), t0=t0, obs=obs,
              truth=rp.load_truth(paths.I1 / "truth.csv"))
    steps = rp.indoornav_sign(read_steps(STEPS[steps_kind]))
    _G["steps"] = {"known": steps, "drift": rp.drift_steps(steps, **DRIFT)}
    _G["rides"], _G["floors"] = rp.baro_events(rp.read_baro(paths.REC / "baro.csv", t0))
    over = {} if sigma_h0_deg is None else dict(sigma_h0=float(np.radians(sigma_h0_deg)))
    _G["P"] = pf_params(n, **over)
    _G["opt"] = opt
    if opt.get("bearing_fix"):
        from bearing_fix import rotation_since_step, fix_bearing
        h = np.load(paths.REC_OUT / "heading_indoornav.npz")
        raw = read_steps(STEPS["indoornav"])
        step_t, step_h = np.array([s["t"] for s in raw]), -np.radians([s["heading_deg"] for s in raw])
        _G["dpsi"] = lambda t: float(np.degrees(rotation_since_step(t + t0, h["t_host"], h["heading"], step_t, step_h)))
        if opt["bearing_fix"] == "bearing":      # 화각 검사는 옛 방향 그대로 (비교용)
            _G["obs"] = {k: [(t, fix_bearing(o, _G["dpsi"](t))) for t, o in v] for k, v in _G["obs"].items()}
            del _G["dpsi"]
    if opt.get("track_limit"):
        from time_filter import load_frames, track_ids
        fr = load_frames(paths.I1 / "obs_nofilter.jsonl")[1]
        ids = track_ids(rp.threshold_frames(fr, opt["cls_th"]) if opt.get("cls_th") else fr)
        _G["obs"] = {k: rp.attach_tracks(v, ids) for k, v in _G["obs"].items()}


def _job(c):
    from vision_filter import VisionFilter
    r0 = _G["truth"][0]
    st = c.get("st") or SETTINGS[c["set"]]
    F = VisionFilter(_G["P"], _G["maps"], _G["fmap"], _G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                   float(r0["heading_deg"]), c["seed"], st["obs_params"])
    obs = _G["obs"]["raw" if c["obs"] == "none" else c["obs"]]
    from time_filter import TrackLimiter
    lim = TrackLimiter(_G["opt"]["track_limit"]) if _G["opt"].get("track_limit") else None
    t = time.perf_counter()
    r = rp.run_one(F, obs, _G["steps"][c["sc"]], _G["rides"], _G["floors"], _G["truth"], _G["t0"],
                   vision=c["obs"] != "none", obs_hz=st["obs_hz"], dpsi=_G.get("dpsi"), limiter=lim)
    d = F.diag
    diag = dict(resamples=F.pf.stats["resamples"], wall_kills=F.pf.stats["wall_kills"], all_dead=F.pf.stats["all_dead"],
                snap_redraw=d["snap_redraw"], miss_mean=d["miss_sum"] / max(d["n_update"], 1), miss_max=d["miss_max"],
                ms_update=1e3 * d["t_update"] / max(d["n_update"], 1),
                ms_predict=1e3 * d["t_predict"] / max(len(_G["steps"][c["sc"]]), 1), s_total=time.perf_counter() - t)
    return c, r, diag


def key(c):
    return f"{c['sc']}/none" if c["obs"] == "none" else f"{c['sc']}/{c['obs']}/{c['set']}"


def provenance(a):
    return record.run_record(STEPS[a.steps], _G["P"].to_dict(), a.seeds, _G["opt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", choices=list(STEPS), default="indoornav")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--sigma-h0", type=float, help="indoornav PF 걸음당 방향 잡음 [도] (기본: pf_home_v3 0.5°)")
    ap.add_argument("--settings", default="기본,중간")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--adopted", action="store_true", help="vision_options.json 설정으로")
    ap.add_argument("--bearing-fix", choices=["theta", "bearing"], help="걸음 사이 회전 보정")
    ap.add_argument("--track-limit", type=float, help="같은 track은 N초에 1번만")
    ap.add_argument("--cls-th", help="종류별 문턱값 exit=0.3,monitor=0.3,...")
    ap.add_argument("--text-check", help="글자 확인 결정 파일 (vision/inputs/i1/ 안 이름)")
    ap.add_argument("--unread", choices=["keep", "drop"], help="못 읽은 관측: 넣음(기본)·뺌")
    a = ap.parse_args()
    opt = dict(bearing_fix=a.bearing_fix, track_limit=a.track_limit, text_check=a.text_check, unread=a.unread,
               cls_th={k: float(v) for k, v in (x.split("=") for x in a.cls_th.split(","))} if a.cls_th else None)
    if a.adopted:
        base = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
        opt = {k: (opt[k] if opt[k] is not None else base[k]) for k in opt}
    sys.stdout.reconfigure(encoding="utf-8")
    sets = a.settings.split(",")
    conds = [dict(sc=sc, obs="none", set="기본", seed=s) for sc in SCENARIOS for s in range(a.seeds)]
    conds += [dict(sc=sc, obs=o, set=st, seed=s) for st in sets for sc in SCENARIOS for o in ("raw", "tf")
              for s in range(a.seeds)]
    res = run_pool(a.procs, _init, (a.steps, a.n, a.sigma_h0, opt), _job, conds)
    _init(a.steps, a.n, a.sigma_h0, opt)
    runs, ms, diags = {}, {}, {}
    for c, r, d in res:
        k = key(c)
        runs.setdefault(k, []).append(r)
        ms.setdefault(k, []).append(rp.metrics(r))
        diags.setdefault(k, []).append(d)
    summ = {k: rp.summarize(ms[k], diags[k]) for k in ms}
    esc = _G["smeta"]["escalators"]
    for sc in SCENARIOS:
        dr = rp.dead_reckon(_G["steps"][sc], _G["floors"], _G["rides"], _G["truth"], _G["t0"], esc)
        summ[f"{sc}/pdr_only"] = rp.dr_metrics(dr, _G["truth"])[0]
        runs[f"{sc}/pdr_only"] = [dr]
    out = paths.REC_OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    info = dict(name=a.name, steps=a.steps, n=a.n, sigma_h0_deg=a.sigma_h0 if a.sigma_h0 is not None else 0.5, options=opt,
                settings={s: SETTINGS[s] for s in sets}, drift=DRIFT, rides=_G["rides"], floors=_G["floors"],
                n_steps=len(_G["steps"]["known"]), provenance=provenance(a))
    (out / "summary.json").write_text(json.dumps(dict(info=info, summary=summ), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    np.savez_compressed(out / "runs.npz", **{f"{k.replace('/', '_')}_{i}": r for k, rs in runs.items()
                                             for i, r in enumerate(rs)})
    print(f"→ {out / 'summary.json'}")
    for k in sorted(summ):
        s = summ[k]
        print(k, {x: (round(s[x], 2) if isinstance(s.get(x), float) else s.get(x))
                  for x in ("med_all", "p90_all", "max_all", "end", "wrong", "ratio", "diag_wall_kills") if x in s})


if __name__ == "__main__":
    main()
