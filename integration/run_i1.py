"""I1 재생 실행기 (indoornav PF): 조건마다 seed 10개 → 지표 요약 (연결안 7절 3~5단계).

    .venv/Scripts/python integration/run_i1.py --steps indoornav --name inav_pdr_pf            # ③ indoornav PDR + PF
    .venv/Scripts/python integration/run_i1.py --steps simple --name simple_pdr_inav_pf   # ② 임시 PDR + indoornav PF
    .venv/Scripts/python integration/run_i1.py --steps indoornav --sigma-h0 1.5 --name inav_h15  # 5단계 민감도
출력: integration/out/i1/<name>/summary.json·summary.md (runs.npz는 저장소에 안 넣음)

조건 (기준 run_i1.py와 같은 이름)
    known : 출발점(정답 t=0) ±2 m를 방 뺀 보행 마스크로 자른 분포, 방향 ±20°
    drift : known + PDR 일부러 틀어짐(보폭 ×1.15, 방향 0.4°/걸음)
    관측  : none(비전 끔) / raw(필터 없음) / tf(시간 필터), 설정 기본(5 Hz·σ 5°·바닥값 0.1)·중간(2 Hz·σ 10°·바닥값 0.1)
    비전 설정 고정: 검출 문턱 0.25, 시간 필터 5프레임 중 3번, 화면 끝 판 끝점 비교, 94면 지도 (관측 파일·역사 패키지 그대로)
    같은 seed = 같은 초기 파티클·같은 입력(비전 끔/켬 비교)
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

import paths
import replay as rp

SETTINGS = {"기본": dict(obs_params=None, obs_hz=None),
            "중간": dict(obs_params={"sigma_bearing": 10.0, "eps": 0.1}, obs_hz=2.0)}
DRIFT = dict(len_scale=1.15, head_per_step=0.4)
SCENARIOS = ("known", "drift")
STEPS = {"indoornav": paths.OUT / "i1/steps_indoornav.csv", "simple": paths.I1 / "steps_simple.csv"}
_G = {}


def _init(steps_kind, n, sigma_h0_deg):
    from station_pkg import load_station
    from edge_faces import add_face_points
    from vision_filter import load_maps, pf_params
    from pdr_steps import read_steps
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    t0 = paths.rec_t0()
    _G.update(fmap=add_face_points(fmap), smeta=smeta, maps=load_maps(), t0=t0,
              obs={"raw": rp.load_obs(paths.I1 / "obs_nofilter.jsonl", t0),
                   "tf": rp.load_obs(paths.I1 / "obs_timefilter.jsonl", t0)},
              truth=rp.load_truth(paths.I1 / "truth.csv"))
    steps = rp.indoornav_sign(read_steps(STEPS[steps_kind]))
    _G["steps"] = {"known": steps, "drift": rp.drift_steps(steps, **DRIFT)}
    _G["rides"], _G["floors"] = rp.baro_events(rp.read_baro(paths.REC / "baro.csv", t0))
    over = {} if sigma_h0_deg is None else dict(sigma_h0=float(np.radians(sigma_h0_deg)))
    _G["P"] = pf_params(n, **over)


def _job(c):
    from vision_filter import VisionFilter
    r0 = _G["truth"][0]
    st = SETTINGS[c["set"]]
    F = VisionFilter(_G["P"], _G["maps"], _G["fmap"], _G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                   float(r0["heading_deg"]), c["seed"], st["obs_params"])
    obs = _G["obs"]["raw" if c["obs"] == "none" else c["obs"]]
    t = time.perf_counter()
    r = rp.run_one(F, obs, _G["steps"][c["sc"]], _G["rides"], _G["floors"], _G["truth"], _G["t0"],
                   vision=c["obs"] != "none", obs_hz=st["obs_hz"])
    d = F.diag
    diag = dict(resamples=F.pf.stats["resamples"], wall_kills=F.pf.stats["wall_kills"], all_dead=F.pf.stats["all_dead"],
                snap_redraw=d["snap_redraw"], miss_mean=d["miss_sum"] / max(d["n_update"], 1), miss_max=d["miss_max"],
                ms_update=1e3 * d["t_update"] / max(d["n_update"], 1),
                ms_predict=1e3 * d["t_predict"] / max(len(_G["steps"][c["sc"]]), 1), s_total=time.perf_counter() - t)
    return c, r, diag


def key(c):
    return f"{c['sc']}/none" if c["obs"] == "none" else f"{c['sc']}/{c['obs']}/{c['set']}"


def sha(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def provenance(a):
    git = lambda *x: subprocess.run(["git", "-C", str(paths.ROOT), *x], capture_output=True, text=True).stdout.strip()
    man = json.loads((paths.VISION / "MANIFEST.json").read_text(encoding="utf-8"))
    files = [STEPS[a.steps], paths.I1 / "obs_nofilter.jsonl", paths.I1 / "obs_timefilter.jsonl", paths.I1 / "truth.csv",
             paths.REC / "baro.csv", paths.INPUTS / "dongchun_station.npz", paths.MASK_META, paths.PF_CONFIG,
             paths.PDR_CONFIG, *paths.MASK_DIR.glob("*.png")]
    import scipy
    return dict(branch_commit=git("rev-parse", "HEAD"), branch_dirty=bool(git("status", "--porcelain")),
                main_commit="9040d70", vision_from=man.get("localization_design_head"),
                files={str(f.relative_to(paths.ROOT)).replace("\\", "/"): sha(f) for f in files},
                pf_params=_G["P"].to_dict(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, seeds=list(range(a.seeds)),
                init="known: N(정답 t=0, 2² I) 보행 마스크 거절 샘플링, 방향 N(정답, 20°), k ~ U(k0_lo, k0_hi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", choices=list(STEPS), default="indoornav")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--sigma-h0", type=float, help="indoornav PF 걸음당 방향 잡음 [도] (기본: pf_home_v3 0.5°)")
    ap.add_argument("--settings", default="기본,중간")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sets = a.settings.split(",")
    conds = [dict(sc=sc, obs="none", set="기본", seed=s) for sc in SCENARIOS for s in range(a.seeds)]
    conds += [dict(sc=sc, obs=o, set=st, seed=s) for st in sets for sc in SCENARIOS for o in ("raw", "tf")
              for s in range(a.seeds)]
    with Pool(a.procs, _init, (a.steps, a.n, a.sigma_h0)) as pool:
        res = pool.map(_job, conds)
    _init(a.steps, a.n, a.sigma_h0)
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
    out = paths.OUT / "i1" / a.name
    out.mkdir(parents=True, exist_ok=True)
    info = dict(name=a.name, steps=a.steps, n=a.n, sigma_h0_deg=a.sigma_h0 if a.sigma_h0 is not None else 0.5,
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
