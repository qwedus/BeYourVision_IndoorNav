"""음성 대조 재생: 좋아 보이는 변경이 '관측 수·끌림 효과'일 뿐인지 보려고, 쓸모없어야 할 조건을 같은 PF로 돌린다. 기본 설정은 안 바꿈, 채택 실험 아님.

    .venv/Scripts/python integration/run_negative.py --rec s2_a_ex6_ex3 --cond random shift_m5 shift_p5
    .venv/Scripts/python integration/compare_runs.py --rec s2_a_ex6_ex3 --a exp_d58_r20 --b neg_random
    .venv/Scripts/python integration/compare_runs.py --rec all --a exp_d58_r20 --b neg_random     (--rec all로 돌린 뒤)

조건 (시간 필터가 버린 관측 중 일부를 되살림. 기존 입력의 시각·관측은 그대로)
    base      아무것도 안 더함: 기본 재생과 비트까지 같아야 함(경로 확인)
    random    버린 관측에서 무작위로 고름. 고르는 데 정답 안 씀. 개수는 oracle(정답 자리에서 맞는 관측) 수와 맞춤, --random-count로 정하면 정답을 전혀 안 씀
    all       버린 관측 전부. 정답 안 씀
    shift_m5·shift_p5  정답을 -5 s / +5 s 옮긴 틀린 자리에서 PF 후보 + 10° 일치하는 관측. 고르는 자리가 틀리면 나빠져야 함
추가 규칙: 같은 track의 기존 입력과 앞뒤 제한 시간 안이면 안 넣고, 추가끼리도 제한 시간에 1번.
출력 <녹화 out 또는 --out-dir>/neg_<조건>/: runs.npz(키 known_tf_기본_<seed>, compare_runs 기본 조건), summary.json(입력 기록 포함).
읽는 법: 어떤 개선이 base 대비 좋아졌는데 random·shift도 비슷하게 좋아지면 그 개선 근거는 약함. 조건은 출발점 앎·시간 필터·기본, 채택 비전 옵션.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import paths
import provenance as record
import replay as rp
import run_i1
from analyze_recording import camera_offset, pf_candidate_match
from edge_faces import frame_obs
from pool_guard import run_pool
from provenance import RECORDINGS
from time_filter import TrackLimiter, load_frames

CONDS = ("base", "random", "all", "shift_m5", "shift_p5")
SHIFT = {"shift_m5": -5.0, "shift_p5": 5.0}
RANDOM_SEED = 12345
_N = {}


def pick_random(pool, count, seed=RANDOM_SEED):
    """버린 관측에서 무작위로 고름. 같은 입력·seed면 같은 결과"""
    pool = sorted(pool)
    if not pool:
        return set()
    return set(np.random.default_rng(seed).choice(pool, size=min(count, len(pool)), replace=False).tolist())


class AddOnly:
    """기존 입력은 그대로 두고 후보만 추가. 같은 track 기존 입력과 앞뒤 limit초 안이면 안 넣고, 추가끼리도 limit초에 1번."""

    def __init__(self, cand, sched, times, limit):
        self.cand, self.sched, self.times, self.limit = cand, sched, times, limit
        self.last, self.add = {}, 0

    def __call__(self, t, obs):
        out = []
        for o in obs:
            i, tid = o["detection_id"], o["track_id"]
            if i in self.sched.get(t, ()):
                out.append(o)
            elif i in self.cand and not any(abs(t - u) < self.limit - 1e-6 for u in self.times.get(tid, ())) \
                    and t - self.last.get(tid, -1e18) >= self.limit - 1e-6:
                out.append(o)
                self.last[tid] = t
                self.add += 1
        return out


def match_ids(raw, dt=0.0):
    """정답을 dt초 옮긴 자리에서 PF 후보 + 10° 일치하는 관측 id"""
    G = run_i1._G
    t0, truth, fmap = G["t0"], G["truth"], G["fmap"]
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
    off = camera_offset(truth, paths.RC["waypoints"], ty, yaw)
    ids = set()
    for f in raw:
        t, obs = frame_obs(f, t0)
        if t < 0 or t > truth["t"][-1] or paths.excluded(np.array([t]))[0]:
            continue
        tr = rp.truth_at(truth, t + dt)
        if rp.truth_at(truth, t)["on_esc"] or tr["on_esc"]:
            continue
        cam = float((np.interp(t, ty, yaw) + off) % 360)
        fm = fmap[int(tr["floor"])]
        ids |= {o["detection_id"] for o in obs if pf_candidate_match(fm, o, tr, cam, (cam + o["bearing"]) % 360)}
    return ids


def candidates(cond, random_count=None):
    """조건이 되살릴 관측 id. base는 빈 집합"""
    raw = load_frames(paths.I1 / "obs_nofilter.jsonl")[1]
    kept = {o["detection_id"] for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1] for o in f["observations"]}
    dropped = {o["detection_id"] for f in raw for o in f["observations"]} - kept
    if cond == "base":
        return set()
    if cond == "all":
        return dropped
    if cond == "random":
        return pick_random(dropped, random_count if random_count is not None else len(dropped & match_ids(raw)))
    return dropped & match_ids(raw, SHIFT[cond])


def _init(opt, n, cond, cand):
    run_i1._init("indoornav", n, None, opt)
    G = run_i1._G
    sched, times = {}, {}
    lim = TrackLimiter(opt["track_limit"])
    for t, oo in G["obs"]["tf"]:
        if any(a <= t <= b for a, b in G["rides"]):
            continue
        used = lim(t, oo)
        sched[t] = {o["detection_id"] for o in used}
        for o in used:
            times.setdefault(o["track_id"], []).append(t)
    kept = {o["detection_id"] for _, oo in G["obs"]["tf"] for o in oo}
    obs = G["obs"]["tf"] if cond == "base" else [(t, [o for o in oo if o["detection_id"] in kept | cand]) for t, oo in G["obs"]["raw"]]
    _N.update(cond=cond, cand=cand, sched=sched, times=times, limit=opt["track_limit"], obs=obs)


def _job(seed):
    from vision_filter import VisionFilter
    G = run_i1._G
    r0 = G["truth"][0]
    F = VisionFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]], float(r0["heading_deg"]), seed, None)
    lim = TrackLimiter(_N["limit"]) if _N["cond"] == "base" else AddOnly(_N["cand"], _N["sched"], _N["times"], _N["limit"])
    r = rp.run_one(F, _N["obs"], G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"], dpsi=G.get("dpsi"), limiter=lim)
    return seed, r, getattr(lim, "add", 0)


def one(a):
    """기록 하나 (paths가 불러올 때 녹화를 정하므로 기록마다 새 프로세스)"""
    opt = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
    run_i1._init("indoornav", a.n, None, opt)
    for cond in a.cond:
        cand = candidates(cond, a.random_count)
        res = run_pool(a.procs, _init, (opt, a.n, cond, cand), _job, list(range(a.seeds)))
        out = (Path(a.out_dir) if a.out_dir else paths.REC_OUT) / f"neg_{cond}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "runs.npz", **{f"known_tf_기본_{s}": r for s, r, _ in res})
        summ = rp.summarize([rp.metrics(r) for _, r, _ in res])
        added = float(np.median([x for _, _, x in res]))
        info = dict(name=out.name, negative=cond, n=a.n, seeds=a.seeds, candidates=len(cand), added_median=added, random_seed=RANDOM_SEED if cond == "random" else None,
                    random_count=a.random_count if cond == "random" else None, options=opt, provenance=record.run_record(run_i1.STEPS["indoornav"], run_i1._G["P"].to_dict(), a.seeds, opt))
        (out / "summary.json").write_text(json.dumps(dict(info=info, summary={"known/tf/기본": summ}), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{paths.RECORDING} {out.name}: 후보 {len(cand)}, 추가 중앙 {added:g}, 오차 중앙 {summ['med_all']:.2f} / 90% {summ['p90_all']:.2f} m", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", nargs="+", choices=CONDS, default=["random", "shift_m5", "shift_p5"])
    ap.add_argument("--rec", nargs="+", help="기록 이름, all이면 9개. 없으면 I1_RECORDING")
    ap.add_argument("--random-count", type=int, help="random 개수를 직접 정함(정답을 전혀 안 씀). 없으면 oracle 수와 맞춤")
    ap.add_argument("--out-dir", help="결과 폴더 (기본: 기록 out)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not a.rec:
        return one(a)
    recs = list(RECORDINGS) if a.rec == ["all"] else a.rec
    if len(recs) > 1 and a.out_dir:
        raise SystemExit("--out-dir는 기록 하나일 때만 (기록끼리 덮어씀)")
    cmd = [sys.executable, __file__, "--cond", *a.cond, "--n", str(a.n), "--seeds", str(a.seeds), "--procs", str(a.procs)]
    cmd += ["--random-count", str(a.random_count)] if a.random_count is not None else []
    cmd += ["--out-dir", a.out_dir] if a.out_dir else []
    for rec in recs:
        p = subprocess.run(cmd, env={**os.environ, "I1_RECORDING": rec})
        if p.returncode:
            sys.exit(p.returncode)


if __name__ == "__main__":
    main()
