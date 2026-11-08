"""실시간 조건 재생 실행기: 기록마다 '지금 방식'과 실시간 조건을 단계별로 같은 seed로 돌려 비교한다 (새 파일, 기본 재생 경로는 안 건드림).

    I1_RECORDING=<이름> .venv/Scripts/python integration/run_rt.py --name rt_compare
    .venv/Scripts/python integration/run_rt.py --name rt_compare --rec all          (9개 기록, 기록마다 새 프로세스)
    .venv/Scripts/python integration/run_rt.py --name rt_compare --report           (9개 기록 summary를 모아 표)
    --stages off,step,...   --check-ref exp_d58_r20 (off 단계가 기준 재생 runs.npz와 기록 필드 비트까지 같은지)

권장 단계 = RECOMMENDED("rec": 봉우리 뒤 0.5 s, 이동 묶음 판정 안 기다림, 탑승 의심 무시·3 m 확정부터 탑승, 최대 기다림 5 s).
조건: 출발점 앎, 시간 필터·기본 설정, 채택 비전 옵션(vision_options.json: 걸음 사이 회전 보정 theta, track 1 s 제한), seed 10개 —
기본 재생(exp_d58_r20 = regress_replay 기준)과 같은 PF·관측·지도. 바뀌는 것은 걸음·기압을 언제 무엇으로 아느냐뿐(STAGES).
출력 <녹화 out>/<name>/: summary.json, runs.npz(단계_seed), steps_*.csv(봉우리·판정 시각), ticks_*.npz, baro_rt.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import paths
import provenance as record
from pool_guard import run_pool
import replay as rp
import rt_pdr
import rt_replay as rr
import run_i1

L0 = rt_pdr.LOOKAHEAD_S
# steps: offline(steps_indoornav.csv, 봉우리 순간) | delayed(indoornav run_pdr 지연 필터, 녹화 전체에서 찾고 봉우리 순간)
#        | online(실시간 판정, look = 봉우리 뒤 몇 초) | early(online인데 묶음 판정을 안 기다림)
# baro:  oracle(지금 방식) | hold(실시간, 의심 중 멈춤) | ignore(실시간, 의심 무시·확정부터 탑승)
STAGES = {
    "off": dict(steps="offline", baro="oracle", note="지금 방식"),
    "filter": dict(steps="delayed", baro="oracle", note="저역 필터만 지연 방식"),
    "step": dict(steps="online", baro="oracle", note="걸음만 실시간"),
    "baro": dict(steps="offline", baro="hold", note="기압만 실시간"),
    "both": dict(steps="online", baro="hold", note="둘 다 실시간"),
    "both_ignore": dict(steps="online", baro="ignore", note="둘 다, 탑승 의심 무시"),
    "both_early": dict(steps="early", baro="hold", note="둘 다, 묶음 판정 안 기다림"),
    "step_l05": dict(steps="online", look=0.5, baro="oracle", note="걸음만, 봉우리 뒤 0.5 s"),
    "step_l10": dict(steps="online", look=1.0, baro="oracle", note="걸음만, 봉우리 뒤 1.0 s"),
    "both_w3": dict(steps="online", baro="hold", max_wait=3.0, note="둘 다, 최대 기다림 3 s"),
    "both_w10": dict(steps="online", baro="hold", max_wait=10.0, note="둘 다, 최대 기다림 10 s"),
    "both_abs": dict(steps="online", baro="hold", baro_src="abs", note="둘 다, 탑승 판정을 절대 고도(3 Hz)로"),
    "ignore_early": dict(steps="early", baro="ignore", note="둘 다, 탑승 의심 무시 + 묶음 판정 안 기다림"),
    "ignore_w3": dict(steps="online", baro="ignore", max_wait=3.0, note="둘 다, 탑승 의심 무시, 최대 기다림 3 s"),
    "ignore_w10": dict(steps="online", baro="ignore", max_wait=10.0, note="둘 다, 탑승 의심 무시, 최대 기다림 10 s"),
    "ignore_l05": dict(steps="online", look=0.5, baro="ignore", note="둘 다, 탑승 의심 무시, 봉우리 뒤 0.5 s"),
    "rec": dict(steps="early", look=0.5, baro="ignore", max_wait=5.0,
                note="권장 실시간: 봉우리 뒤 0.5 s + 묶음 판정 안 기다림 + 탑승 의심 무시 + 최대 기다림 5 s"),
}
RECOMMENDED = "rec"     # 실시간 재생 권장 단계(11-08, 실행은 --stages rec). 다른 단계는 비교·재현용으로 그대로 둠
MAX_WAIT = 5.0
RT_KEYS = ("delay_med", "delay_p90", "delay_max", "now_med", "now_p90", "now_max", "dr_med", "dr_p90", "dr_max",
           "late_steps", "forced", "held_lost")
_R = {}


def stage(name):
    return {"look": L0, "max_wait": MAX_WAIT, "baro_src": "rel", **STAGES[name]}


def has_abs():
    return (paths.REC / "baro_abs.csv").exists()


def step_file(out, kind, look):
    return out / ("steps_delayed.csv" if kind == "delayed" else f"steps_{kind}_l{int(round(look * 100)):03d}.csv")


def prepare(out, names):
    """걸음 파일·기압 후보를 만든다(재생 전, 메인 프로세스). → 확인 기록"""
    from indoornav.pdr import PDRParams
    from app_session import load_app_session
    s, info = load_app_session()
    P0 = PDRParams.from_toml(paths.PDR_CONFIG)
    t0s = info["t0_sensor"]
    chk = {}
    out.mkdir(parents=True, exist_ok=True)
    looks = sorted({stage(n)["look"] for n in names if stage(n)["steps"] in ("online", "early")})
    if any(stage(n)["steps"] == "delayed" for n in names):
        rt_pdr.write_rows(rt_pdr.offline_rows(s, P0, t0s, "delayed"), step_file(out, "delayed", 0))
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    avail = rt_pdr.motion_avail(paths.REC, s, t0s)          # 격자 샘플을 받은 시각(바로 뒤 원본 표본)
    chk["motion_avail_lag_max"] = float(np.max(avail - s.t))
    saved = False
    for look in looks:
        st, tk = rt_pdr.online_steps(s, P0, look, rt_pdr.WINDOW_S, avail=avail)
        if not saved:                                       # 실시간 단계 방향 = 이번 Session에서 걸음과 같이 계산한 값
            np.savez_compressed(out / "heading_rt.npz", t_host=s.t + t0s, heading=tk["heading"], avail_host=avail + t0s)
            saved = True
        for kind in ("online", "early"):
            rt_pdr.write_rows(rt_pdr.step_rows(st, t0s, "wait" if kind == "online" else "early"), step_file(out, kind, look))
        np.savez_compressed(out / f"ticks_l{int(round(look * 100)):03d}.npz", tick_t=tk["tick_t"] + t0s, cut_t=tk["cut_t"] + t0s)
        ticks = set((tk["tick_t"] + t0s).tolist())          # 읽어 들인 판정 시각이 모두 실제 갱신 시각과 정확히 같은지
        for kind in ("online", "early"):
            for r in rt_pdr.read_rows(step_file(out, kind, look)):
                for k in ("t_confirm", "t_known"):
                    if np.isfinite(r[k]) and r[k] not in ticks:
                        raise SystemExit(f"{kind} 걸음 {k} {r[k]!r}가 갱신 시각에 없음 → 판정 시각 반올림 어긋남")
        d = np.array([x["t_confirm"] - x["t"] for x in st])
        k = np.array([x["t_known"] - x["t"] for x in st if x["t_known"] is not None])
        q = (lambda f, a: float(f(a)) if len(a) else None)
        chk[f"l{look}"] = dict(n=len(st), undecided=sum(x["moving"] is None for x in st), bout_vs_team=rt_pdr.check_bout(st, P0),
                               confirm_med=q(np.median, d), known_med=q(np.median, k),
                               known_p90=q(lambda a: np.percentile(a, 90), k), known_max=q(np.max, k),
                               heading_same=bool(np.array_equal(tk["heading"], h["heading"])))
    t0 = paths.rec_t0()
    cands, _ = rt_pdr.online_baro(rt_pdr.read_baro_rt(paths.REC / "baro.csv", t0))
    (out / "baro_rt.json").write_text(json.dumps([{k: v for k, v in c.items()} for c in cands], ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    chk["baro"] = dict(cands=len(cands), rides=sum(c["ride"] for c in cands))
    return chk


def _init(out, n):
    run_i1._init("indoornav", n, None, json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8")))
    G = run_i1._G
    if G["opt"].get("bearing_fix") not in (None, "theta"):
        raise SystemExit("bearing_fix theta·끔만 지원")
    out = Path(out)
    t0 = G["t0"]
    _R.update(out=out, steps={}, ticks={})
    for f in out.glob("steps_*.csv"):
        _R["steps"][f.stem] = rt_pdr.read_rows(f)
    for f in out.glob("ticks_*.npz"):
        z = np.load(f)
        _R["ticks"][f.stem] = (z["tick_t"] - t0, z["cut_t"] - t0)
    _R["baro"] = rt_pdr.online_baro(rt_pdr.read_baro_rt(paths.REC / "baro.csv", t0))
    if has_abs():
        _R["baro_abs"] = rt_pdr.online_baro(rt_pdr.read_baro_abs_rt(paths.REC / "baro_abs.csv", t0))
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    _R["heading_all"] = (h["t_host"], h["heading"])                       # 걸음을 미리 아는 단계: 기준 재생 방향 파일 그대로
    f = out / "heading_rt.npz"
    if f.exists():                                                        # 실시간 걸음 단계: 걸음과 같은 Session 방향, 받은 표본만
        a = np.load(f)
        _R["heading_rt"] = (a["t_host"], a["heading"], a["avail_host"])
    _R["bearing"] = G["opt"].get("bearing_fix") == "theta"


def heading_for(sg):
    """실시간 걸음 단계는 받은 표본만, 걸음을 미리 아는 단계(off·filter·baro)는 기준 재생과 같은 전체 배열"""
    return _R["heading_rt"] if sg["steps"] in ("online", "early") else _R["heading_all"]


def knowledge(sg):
    G = run_i1._G
    t0 = G["t0"]
    if sg["steps"] == "offline":
        sk = rr.OracleSteps(G["steps"]["known"], t0)
    elif sg["steps"] == "delayed":
        sk = rr.OracleSteps(_R["steps"]["steps_delayed"], t0)
    else:
        tag = f"l{int(round(sg['look'] * 100)):03d}"
        sk = rr.OnlineSteps(_R["steps"][f"steps_{sg['steps']}_{tag}"], t0, *_R["ticks"][f"ticks_{tag}"])
    if sg["baro"] == "oracle":
        bk = rr.OracleBaro(G["rides"], G["floors"])
    else:
        bk = rr.OnlineBaro(*_R["baro_abs" if sg["baro_src"] == "abs" else "baro"], suspect=sg["baro"])
    return sk, bk


def _job(c):
    from vision_filter import VisionFilter
    from time_filter import TrackLimiter
    G = run_i1._G
    sg = stage(c["stage"])
    r0 = G["truth"][0]
    st = run_i1.SETTINGS["기본"]
    F = VisionFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                     float(r0["heading_deg"]), c["seed"], st["obs_params"])
    lim = TrackLimiter(G["opt"]["track_limit"]) if G["opt"].get("track_limit") else None
    sk, bk = knowledge(sg)
    log = {}
    t = time.perf_counter()
    r = rr.run_rt(F, G["obs"]["tf"], sk, bk, G["truth"], G["t0"], G["rides"], vision=True, obs_hz=st["obs_hz"], limiter=lim,
                  heading=heading_for(sg) if _R["bearing"] else None, max_wait=sg["max_wait"], log=log)
    log["s_total"] = time.perf_counter() - t
    return c, r, log


def check_ref(runs, ref):
    """off 단계 기록(run_one 필드)이 기준 runs.npz known_tf_기본_<seed>와 바이트까지 같은지 → (같은 seed 수, 비교 seed 수, 다른 필드)"""
    z = np.load(ref)
    same, bad = 0, set()
    for seed, r in runs:
        o = z[f"known_tf_기본_{seed}"]
        f = [n for n, _ in rp.REC_FIELDS if len(o) != len(r) or r[n].tobytes() != o[n].tobytes()]
        same += not f
        bad |= set(f)
    return same, len(runs), sorted(bad)


def refuse_overwrite(out, a):
    """출력 폴더가 비교 기준이거나, 이 실행기가 만들지 않은 재생 결과(runs.npz에 단계 키가 아닌 키)가 있으면 쓰기 전에 멈춤"""
    if a.check_ref and os.path.normcase(str(out)) == os.path.normcase(str(paths.REC_OUT / a.check_ref)):
        raise SystemExit(f"출력 이름 {a.name}이 비교 기준과 같음 → 기준을 덮어쓰게 되어 멈춤")
    f = out / "runs.npz"
    if f.exists():
        keys = np.load(f).files
        bad = [k for k in keys if k.rsplit("_", 1)[0] not in STAGES or not k.rsplit("_", 1)[-1].isdigit()]
        if bad:
            raise SystemExit(f"{out.name}/runs.npz에 실시간 재생 단계가 아닌 키({bad[:3]} …) → 다른 결과를 덮어쓰게 되어 멈춤")
    f = out / "summary.json"
    if f.exists():
        summ = json.loads(f.read_text(encoding="utf-8")).get("summary", {})
        bad = [k for k in summ if k not in STAGES]
        if bad:
            raise SystemExit(f"{out.name}/summary.json에 실시간 재생 단계가 아닌 조건({bad[:3]} …) → 다른 결과를 덮어쓰게 되어 멈춤")


def truth_floor_changes(truth):
    f = truth["floor"]
    i = np.flatnonzero(np.diff(f) != 0) + 1
    return [dict(t=float(truth["t"][k]), floor=int(f[k])) for k in i]


def one(a):
    sys.stdout.reconfigure(encoding="utf-8")
    names = [s for s in a.stages.split(",") if stage(s)["baro_src"] != "abs" or has_abs()]   # 1차 녹화는 절대 고도 없음
    out = paths.REC_OUT / a.name
    refuse_overwrite(out, a)
    chk = prepare(out, names)
    conds = [dict(stage=s, seed=k) for s in names for k in range(a.seeds)]
    res = run_pool(a.procs, _init, (str(out), a.n), _job, conds)
    _init(str(out), a.n)
    G = run_i1._G
    runs, ms, logs = {}, {}, {}
    for c, r, lg in res:
        sk, _ = knowledge(stage(c["stage"]))
        runs.setdefault(c["stage"], []).append((c["seed"], r))
        ms.setdefault(c["stage"], []).append(rr.rt_metrics(r, G["truth"], sk.steps, G["t0"], lg, heading_for(stage(c["stage"])),
                                                           getattr(sk, "end_cut", rr.INF)))
        logs.setdefault(c["stage"], []).append(lg)
    summ = {}
    for s in names:
        m = ms[s]
        x = rp.summarize(m)
        for k in RT_KEYS:
            v = [q[k] for q in m if q.get(k) is not None]
            x[k] = float(np.median(v)) if v else None
        x["floor_w"] = m[0].get("floor_w")
        x["floor_t"] = m[0].get("floor_t")
        x["s_total"] = float(np.median([lg["s_total"] for lg in logs[s]]))
        x["unprocessed"] = max(lg["unprocessed"] for lg in logs[s])
        x["note"] = STAGES[s]["note"]
        x["stage"] = stage(s)
        summ[s] = x
    if a.check_ref and "off" in runs:
        same, n, bad = check_ref(runs["off"], paths.REC_OUT / a.check_ref / "runs.npz")
        chk["check_ref"] = dict(ref=a.check_ref, same=same, n=n, fields=bad)
    info = dict(name=a.name, rec=paths.RECORDING, n=a.n, seeds=a.seeds, max_wait=MAX_WAIT, window_s=rt_pdr.WINDOW_S,
                rides_offline=G["rides"], floors_offline=G["floors"], truth_floor=truth_floor_changes(G["truth"]), check=chk,
                provenance=record.run_record(run_i1.STEPS["indoornav"], G["P"].to_dict(), a.seeds, G["opt"]))
    (out / "summary.json").write_text(json.dumps(dict(info=info, summary=summ), ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    np.savez_compressed(out / "runs.npz", **{f"{s}_{seed}": r for s, rs in runs.items() for seed, r in rs})
    print("RESULT " + json.dumps(dict(rec=paths.RECORDING, check=chk), ensure_ascii=False, default=float))
    for s in names:
        x = summ[s]
        print(s, {k: (round(x[k], 2) if isinstance(x.get(k), float) else x.get(k))
                  for k in ("med_all", "p90_all", "max_all", "wrong", "cover2", "delay_med", "delay_p90", "now_med", "dr_med",
                            "late_steps", "forced", "held_lost", "s_total")})


REPORT_KEYS = [("med_all", "오차 중앙"), ("p90_all", "90%"), ("max_all", "최대"), ("wrong", "자신 있게 틀림"), ("cover2", "cover2"),
               ("delay_med", "출력 지연 중앙"), ("delay_p90", "지연 90%"), ("now_med", "보여 준 위치 오차 중앙"), ("dr_med", "앞당긴 위치 오차 중앙")]


def report(a):
    rows = {}
    for rec in record.RECORDINGS:
        rc = json.loads((paths.RECORDINGS / f"{rec}.json").read_text(encoding="utf-8"))
        p = paths.ROOT / rc["out"] / a.name / "summary.json"
        if p.exists():
            rows[rec] = json.loads(p.read_text(encoding="utf-8"))
    out = dict(name=a.name, recs=rows)
    (paths.OUT / f"{a.name}_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for s in a.stages.split(","):
        print(f"\n## {s} ({STAGES[s]['note']})")
        print("| 기록 | " + " | ".join(n for _, n in REPORT_KEYS) + " |")
        print("|---" * (len(REPORT_KEYS) + 1) + "|")
        for rec, d in rows.items():
            x = d["summary"].get(s)
            if x is None:
                continue
            print(f"| {rec} | " + " | ".join(f"{x[k]:.2f}" if isinstance(x.get(k), float) else str(x.get(k)) for k, _ in REPORT_KEYS) + " |")
    print(f"→ {paths.OUT / f'{a.name}_all.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--rec", nargs="+")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--check-ref", default="exp_d58_r20")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a)
    if not a.rec:
        return one(a)
    recs = list(record.RECORDINGS) if a.rec == ["all"] else a.rec
    bad = 0
    for rec in recs:
        t = time.time()
        cmd = [sys.executable, __file__, "--name", a.name, "--stages", a.stages, "--n", str(a.n), "--seeds", str(a.seeds),
               "--procs", str(a.procs), "--check-ref", a.check_ref]
        p = subprocess.run(cmd, env={**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8"}, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        if p.returncode or line is None:
            print(f"{rec}: 실패 (코드 {p.returncode})\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}", flush=True)
            bad += 1
            continue
        print(f"{rec} ({time.time() - t:.0f} s): {line[7:]}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
