"""앱 OCR 판정(Apple Vision, 2차 기록)을 위치 재생에 넣었을 때와 뺐을 때 비교. 기본 설정(vision_options.json의 OCR 끔)은 안 바꿈, 채택 실험 아님.

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_ocr.py --decisions     결정 파일만 만들고 연결률 출력
    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_ocr.py --cond off on    조건 고르기 (기본: 전부)
    .venv/Scripts/python integration/exp_ocr.py --all                                   앱 판정이 있는 2차 8개 기록 (기록마다 새 프로세스)

결정 파일: 앱 decisions.jsonl(검출마다 그 시각의 OCR 상태) → text_check 형식 <입력 폴더>/text_check_applevision*.jsonl.
    앱 상태        → text_status / decision
    confirmed      → confirmed / keep
    other_text     → mismatch / relabel(앱이 남기고 종류를 바꾼 것, map_types = 앱 label_map[final_label]) 또는 reject(앱이 ocr_other_text로 버린 것)
                     (그 밖의 이유로 앱이 버린 other_text 행은 같은 track의 그 시각 전 마지막 other_text 읽기의 suggested_label로 relabel/reject)
    unreadable, no_text → unread / keep
    pending, not_yet    → unread / keep (기본 파일) · pending / keep (_fail 파일: unread='drop'일 때 읽기 실패만 빠지게)
    skip           → skipped / keep (글자 확인 안 하는 종류)
    off            → skipped / keep (앱이 OCR 전에 버린 것: 문턱 아래·지도에 없는 종류·크기 상한. PC 관측에 남는 것은 크기 상한뿐이고,
                     OCR 판정이 아니라서 이 비교에서는 끔·켬 모두 넣음)
조건 (모두 출발점 앎·PC 시간 필터 관측·기본 설정, 채택 비전 옵션, seed 10개, 같은 seed = 같은 초기 파티클)
    off         OCR 끔 (기본 재생 = exp_d58_r20과 비트까지 같아야 함)
    on          켬: 앱 판정의 버림·바로잡기만 적용 (unread='keep')
    dropfail    켬 + 읽기 실패(unreadable·no_text) 관측 뺌
    dropunread  켬 + 확인 안 된 관측 모두 뺌(읽기 실패 + 읽기 대기 pending·not_yet). 글자 종류는 확인·바로잡기된 것만 남음
    relabel     바로잡기만: 앱이 종류를 바꾼 관측만 바꾸고 아무것도 안 뺌(앱이 버린 관측도 넣음). on과 짝지으면 버림의 효과
    kept        참고: 앱이 최종으로 남긴 검출(앱 시간 필터·OCR·크기 상한·바로잡기, session_to_replay --which kept와 같은 집합). PC 시간 필터 안 거침
    apptf       참고: 앱 시간 필터를 통과한 검출만(OCR 판정 무시). 앱에서 크기 상한은 시간 필터 앞이라 크기 상한에 걸린 검출은 여기서도 빠짐.
                kept와 짝지으면 앱 시간 필터 아래에서의 OCR(버림·바로잡기) 효과
    rnd_<조건>_<k>  대조: 그 조건(on·dropfail·dropunread)이 뺀 관측 수만큼, 글자 확인 대상 종류의 관측에서 무작위로 뺌(OCR·정답 안 씀).
                    k = 뽑기 번호(난수 seed RND_SEED + k). 'OCR이 고른 관측을 뺐기 때문'인지 '관측이 줄었기 때문'인지 가르려고 둠.
                    무작위 조건은 기본 목록에 없음: --cond rnd_dropunread_1 처럼 고름
                    주의: 개수는 track 제한(같은 판 1초에 1번) 전 기준이라, PF가 실제로 쓰는 관측 수는 OCR 조건과 크게 다를 수 있음 → rndu
    rndu_<조건>_<k> 대조: PF가 실제로 쓰는 관측 수(탑승 중 제외·track 제한 뒤, run_one과 같은 규칙)를 그 조건과 맞춤.
                    글자 확인 대상 관측을 (track, 그 track 첫 관측부터 1초 칸) 묶음으로 나눠 무작위 순서로 빼 가다가 실제 사용 수가
                    그 조건 이하가 되면 멈춤(정답·OCR 안 씀). 난수 seed RND_SEED + 100 + k
    farwait     진단(정답 사용): 읽기 대기(pending·not_yet) 관측 중 정답 자리에서 PF 거리 표 20~40 m 판과만 맞는 것(exp_ocr_obs.py의 av_obs.json
                cls = far)만 뺌. 못 읽음 뺌의 이득이 먼 판 관측 때문인지 보려고 둠. 채택 후보가 아님
    a1·a1d·a2   덜 버리기 규칙(exp_ocr_rules.py가 만든 결정 파일 text_check_applevision_<규칙>.jsonl, unread='keep'). 규칙 뜻은 그 파일 설명
    wn50·wn25   낮은 가중치(좁힌 경우): 주변지역 안내도(area_map) 관측 중 앱 읽기 실패(no_text·unreadable)인 것의 우도를 α제곱(0.5·0.25)
    wu50·wu25   낮은 가중치(비교용): 글자 확인 대상 종류 관측 중 확인 안 된 것 전부(읽기 실패 + 대기, 기본 결정 파일 text_status unread)를 α제곱
                가중치 = 관측 하나의 우도 L에 L^α (log 우도 × α). α = 1인 관측은 원래 함수에 한꺼번에 넣고, 나머지는 관측마다 따로 계산해 곱함
                값 0.5·0.25는 결과 보기 전에 정함
    eps03       대조: OCR 끔 + 관측 모델 바닥값 0.1 → 0.3 (과신 원인 문서 4절 후보)
    rndu_<a1|a1d|a2>_1  그 규칙과 PF 실제 사용 수를 맞춘 무작위 대조(뺀 관측이 있는 조건만)
출력: <녹화 out>/av_<조건>/summary.json·runs.npz(키 known_tf_기본_<seed>, compare_runs.py로 짝 비교). 있는 폴더는 --force 없이는 안 덮어씀.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import paths
import provenance as record
import replay as rp
import run_i1
from pool_guard import run_pool

S2_APP = ("s2_a_71_ex2", "s2_a_ex3_ex6", "s2_a_ex6_ex3", "s2_a_platform", "s2_b_elev_pillar", "s2_b_emart_ex5", "s2_b_ex2_31", "s2_b_pillar_map")
CONDS = ("off", "on", "dropfail", "dropunread", "kept", "apptf", "relabel")
RND = tuple(f"rnd_{c}_{k}" for c in ("on", "dropfail", "dropunread") for k in (1, 2))
RNDU = tuple(f"rndu_{c}_{k}" for c in ("on", "dropfail", "dropunread") for k in (1, 2))
DIAG = ("farwait",)
RULES = ("a1", "a1d", "a2")
WEIGHT = {"wn50": ("narrow", 0.5), "wn25": ("narrow", 0.25), "wu50": ("unread", 0.5), "wu25": ("unread", 0.25)}
EXTRA = RULES + tuple(WEIGHT) + ("eps03",) + tuple(f"rndu_{r}_1" for r in RULES)
RND_SEED = 20261107
DEC_MAIN, DEC_FAIL, DEC_RELABEL = "text_check_applevision.jsonl", "text_check_applevision_fail.jsonl", "text_check_applevision_relabel.jsonl"
OPT = {"off": dict(text_check=None, unread=None), "on": dict(text_check=DEC_MAIN, unread="keep"),
       "dropfail": dict(text_check=DEC_FAIL, unread="drop"), "dropunread": dict(text_check=DEC_MAIN, unread="drop"),
       "kept": dict(text_check=None, unread=None), "apptf": dict(text_check=None, unread=None),
       "relabel": dict(text_check=DEC_RELABEL, unread="keep"),
       **{r: dict(text_check=f"text_check_applevision_{r}.jsonl", unread="keep") for r in RULES}}
UNREAD, WAIT = ("unreadable", "no_text"), ("pending", "not_yet")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def app_files():
    """→ (decisions 행 목록, ocr 읽기 목록, 앱 label_map, 설정 이름)"""
    cfg = json.loads((paths.REC / "config.json").read_text(encoding="utf-8"))
    return jl(paths.REC / "decisions.jsonl"), jl(paths.REC / "ocr.jsonl"), cfg["label_map"], cfg.get("config_version")


def to_rows(decs, reads, label_map, fail=False):
    """앱 결정 → text_check 결정 행. → (행 목록, 상태별 수)"""
    by_track = {}
    for o in reads:
        if o.get("verdict") == "other_text":
            by_track.setdefault(o["track_id"], []).append(o)
    out, n = [], Counter()
    for d in decs:
        st, lab = d["ocr_state"], d["label"]
        row = dict(detection_id=d["detection_id"], track_id=d["track_id"], t=d["t"], label=lab, text=d.get("ocr_text") or "",
                   reason=f"app {st}", app_state=st, app_final=d["final"], app_reason=d["reason"])
        if st == "confirmed":
            row.update(decision="keep", text_status="confirmed")
        elif st == "other_text":
            to = None
            if d["final"] == "kept":
                to = d["final_label"] if d["final_label"] and d["final_label"] != lab else None
                if to is None:
                    n["other_text_kept_same_label"] += 1
            elif d["reason"] != "ocr_other_text":          # 다른 이유로 앱이 버린 행: 같은 track의 마지막 other_text 읽기로
                prev = [o for o in by_track.get(d["track_id"], []) if o["t"] <= d["t"] + 1e-9]
                to = prev[-1].get("suggested_label") if prev else None
                n["other_text_from_read"] += 1
            if to:
                row.update(decision="relabel", text_status="mismatch", map_types=label_map[to], relabel_to=to)
            elif d["final"] == "kept":
                row.update(decision="keep", text_status="mismatch")
            else:
                row.update(decision="reject", text_status="mismatch")
        elif st in UNREAD:
            row.update(decision="keep", text_status="unread")
        elif st in WAIT:
            row.update(decision="keep", text_status="pending" if fail else "unread")
        elif st in ("skip", "off"):
            row.update(decision="keep", text_status="skipped")
            if st == "off":
                n[f"off_reason_{d['reason']}"] += 1
        else:
            row.update(decision="keep", text_status="unread")
            n[f"unknown_state_{st}"] += 1
        n[f"{st}->{row['decision']}"] += 1
        out.append(row)
    return out, n


def write_decisions():
    """결정 파일 셋을 입력 폴더에 씀(있으면 내용이 같을 때만 그대로) → {파일 이름: 상태별 수}. relabel = 기본 파일에서 버림을 넣음으로"""
    decs, reads, lm, ver = app_files()
    info = {}
    for name, variant in ((DEC_MAIN, "main"), (DEC_FAIL, "fail"), (DEC_RELABEL, "relabel")):
        rows, n = to_rows(decs, reads, lm, variant == "fail")
        if variant == "relabel":
            rows = [dict(r, decision="keep") if r["decision"] == "reject" else r for r in rows]
            n = Counter({k: v for k, v in n.items() if "->" not in k}) + Counter(f"{r['app_state']}->{r['decision']}" for r in rows)
        head = dict(engine="apple_vision_app", source="app decisions.jsonl + ocr.jsonl", app_config=ver, variant=variant,
                    decisions_sha256=sha(paths.REC / "decisions.jsonl"))
        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in [head, *rows]) + "\n"
        p = paths.I1 / name
        if p.exists() and p.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{p.name}가 이미 있고 내용이 다름. 확인 뒤 지울 것")
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        info[name] = dict(counts=dict(sorted(n.items())), sha256=sha(p), rows=len(rows))
    return info


def link_report():
    """PC 시간 필터 관측과 결정의 연결률, 관측 기준 앱 상태 분포"""
    from text_check import link_stats, load_decisions
    from time_filter import load_frames
    dec = load_decisions(paths.I1 / DEC_MAIN)[0]
    out = {}
    for kind, f in (("tf", "obs_timefilter.jsonl"), ("raw", "obs_nofilter.jsonl")):
        frames = load_frames(paths.I1 / f)[1]
        st = link_stats(frames, dec)
        obs = [o for fr in frames for o in fr["observations"]]
        st["by_state"] = dict(Counter(dec[o["detection_id"]]["app_state"] if o["detection_id"] in dec else "결정 없음" for o in obs))
        st["by_decision"] = dict(Counter(dec[o["detection_id"]]["decision"] if o["detection_id"] in dec else "결정 없음" for o in obs))
        st["n_rows"] = len(obs)
        out[kind] = st
    return out


def app_frames(kind):
    """참고 조건 관측: 관측 없는 필터 프레임(obs_nofilter)에서 앱 kept 또는 앱 시간 필터 통과만. kept는 종류를 final_label로"""
    from time_filter import load_frames
    decs, _, lm, _ = app_files()
    by = {d["detection_id"]: d for d in decs}
    frames = load_frames(paths.I1 / "obs_nofilter.jsonl")[1]
    out, miss = [], 0
    for f in frames:
        f = json.loads(json.dumps(f))
        keep = []
        for o in f["observations"]:
            d = by.get(o["detection_id"])
            if d is None:
                miss += 1
                continue
            if kind == "kept" and d["final"] == "kept":
                if d["final_label"] and d["final_label"] != o["label"]:
                    o["map_types"] = lm[d["final_label"]]
                keep.append(o)
            elif kind == "apptf" and d["stages"].get("time_filter", {}).get("pass", False):   # 크기 상한에 걸린 검출은 앱 시간 필터에 안 들어감
                keep.append(o)
        f["observations"] = keep
        out.append(f)
    return out, miss


def random_drop(ref, k):
    """→ (뺄 detection_id 집합, 정보). 개수 = ref 조건이 시간 필터 관측에서 빼는 수, 후보 = 글자 확인 대상 종류 관측"""
    from text_check import TEXT_CLASSES, load_decisions
    from time_filter import load_frames
    o = OPT[ref]
    dec = load_decisions(paths.I1 / o["text_check"])[0]
    obs = [x for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1] for x in f["observations"] if x["map_types"]]
    gone = [x["detection_id"] for x in obs if x["detection_id"] in dec and (dec[x["detection_id"]]["decision"] == "reject"
            or (o["unread"] == "drop" and dec[x["detection_id"]]["text_status"] == "unread"))]
    pool = sorted(x["detection_id"] for x in obs if x["label"] in TEXT_CLASSES)
    pick = np.random.default_rng(RND_SEED + k).choice(pool, size=len(gone), replace=False).tolist()
    return set(pick), dict(ref=ref, draw=k, seed=RND_SEED + k, n_drop=len(gone), pool=len(pool), overlap_with_ref=len(set(pick) & set(gone)))


def used_count(obs):
    """PF가 실제로 쓰는 관측 수: run_one과 같은 규칙(탑승 중 안 씀, 기본 설정은 obs_hz 없음, 같은 track은 track_limit 초에 1번). PF 상태·정답과 무관"""
    from time_filter import TrackLimiter
    G = run_i1._G
    lim = TrackLimiter(G["opt"]["track_limit"]) if G["opt"].get("track_limit") else None
    n = 0
    for t, val in obs:
        if any(a <= t <= b for a, b in G["rides"]):
            continue
        n += len(lim(t, val) if lim is not None else val)
    return n


def drop_ids(obs, ids):
    return [(t, [o for o in v if o["detection_id"] not in ids]) for t, v in obs]


def matched_drop(ref, k, n):
    """→ (뺄 id 집합, 정보). ref 조건의 실제 사용 관측 수에 맞춰 (track, 1초 칸) 묶음을 무작위로 뺌. 끝나면 _G는 off 상태"""
    from text_check import TEXT_CLASSES
    _init(ref, n)
    target = used_count(run_i1._G["obs"]["tf"])
    _init("off", n)
    base = run_i1._G["obs"]["tf"]
    first, blocks = {}, {}
    for t, v in base:
        for o in v:
            if o["label"] in TEXT_CLASSES and o.get("track_id") is not None:
                t0 = first.setdefault(o["track_id"], t)
                blocks.setdefault((o["track_id"], int((t - t0) // 1.0)), []).append(o["detection_id"])
    keys = sorted(blocks)
    if not keys:
        raise SystemExit(f"{paths.RECORDING}: track 번호가 붙은 관측이 없어(track_limit 꺼짐?) 실제 사용 수를 맞출 수 없음")
    order = np.random.default_rng(RND_SEED + 100 + k).permutation(len(keys))
    used = lambda m: used_count(drop_ids(base, {i for j in order[:m] for i in blocks[keys[j]]}))
    lo, hi = 0, len(keys)                      # 실제 사용 수는 묶음을 더 뺄수록 (거의) 줄어듦 → 이분 탐색, 끝에 실제 값 기록
    while lo < hi:
        mid = (lo + hi) // 2
        if used(mid) <= target:
            hi = mid
        else:
            lo = mid + 1
    ids = {i for j in order[:lo] for i in blocks[keys[j]]}
    got = used(lo)
    if got != target:
        raise SystemExit(f"{paths.RECORDING}: 실제 사용 수를 못 맞춤 (목표 {target}, 결과 {got}) — 무작위 대조로 쓰지 말 것")
    return ids, dict(ref=ref, draw=k, seed=RND_SEED + 100 + k, target_used=target, achieved_used=got, n_blocks=lo, blocks=len(keys),
                     n_drop=len(ids), used_off=used_count(base))


def farwait_ids():
    a = json.loads((paths.REC_OUT / "av_obs.json").read_text(encoding="utf-8"))
    return {o["id"] for o in a["obs"] if o["state"] in WAIT and o["cls"] == "far"}


def weighted_likelihood(x, y, theta, floor, observations, fmap, params=None):
    """관측마다 우도 L^alpha (alpha 없으면 1). alpha = 1인 관측은 원래 함수에 한꺼번에.
    obs_model 기본 unknown='floor'에서만 원래 모델과 같은 뜻 (unknown='median'이면 나눈 호출마다 중앙값이라 다름)"""
    from obs_model import obs_likelihood
    if not observations:
        return obs_likelihood(x, y, theta, floor, observations, fmap, params)
    plain = [o for o in observations if o.get("alpha", 1.0) == 1.0]
    weighted = [o for o in observations if o.get("alpha", 1.0) != 1.0]
    if not weighted:
        return obs_likelihood(x, y, theta, floor, observations, fmap, params)
    logw = np.log(np.maximum(obs_likelihood(x, y, theta, floor, plain, fmap, params), 1e-300)) if plain else np.zeros(len(x))
    for o in weighted:
        logw = logw + o["alpha"] * np.log(np.maximum(obs_likelihood(x, y, theta, floor, [o], fmap, params), 1e-300))
    return np.exp(logw - logw.max())


def weight_ids(kind):
    """→ 낮은 가중치를 줄 detection_id 집합"""
    from text_check import TEXT_CLASSES, load_decisions
    dec = load_decisions(paths.I1 / DEC_MAIN)[0]
    if kind == "narrow":
        return {i for i, d in dec.items() if d["label"] == "area_map" and d["app_state"] in UNREAD}
    return {i for i, d in dec.items() if d["label"] in TEXT_CLASSES and d["text_status"] == "unread"}


def _reset():
    """조건마다 바꾸는 전역(관측 모델 함수·기본 설정)을 원래대로"""
    import obs_model
    import vision_filter
    vision_filter.obs_likelihood = obs_model.obs_likelihood
    run_i1.SETTINGS["기본"]["obs_params"] = None


def _init(cond, n):
    _reset()
    if cond in WEIGHT:
        kind, alpha = WEIGHT[cond]
        _init("off", n)
        ids = weight_ids(kind)
        run_i1._G["obs"]["tf"] = [(t, [dict(o, alpha=alpha) if o["detection_id"] in ids else o for o in v]) for t, v in run_i1._G["obs"]["tf"]]
        import vision_filter
        vision_filter.obs_likelihood = weighted_likelihood
        return
    if cond == "eps03":
        _init("off", n)
        run_i1.SETTINGS["기본"]["obs_params"] = {"eps": 0.3}
        return
    if cond.startswith("rnd_"):
        _, ref, k = cond.split("_")
        _init("off", n)
        drop, _ = random_drop(ref, int(k))
        run_i1._G["obs"]["tf"] = drop_ids(run_i1._G["obs"]["tf"], drop)
        return
    if cond.startswith("rndu_"):
        _, ref, k = cond.split("_")
        drop, _ = matched_drop(ref, int(k), n)
        run_i1._G["obs"]["tf"] = drop_ids(run_i1._G["obs"]["tf"], drop)
        return
    if cond == "farwait":
        _init("off", n)
        run_i1._G["obs"]["tf"] = drop_ids(run_i1._G["obs"]["tf"], farwait_ids())
        return
    base = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
    opt = dict(base)
    opt.update(OPT[cond])                     # None도 덮어씀: off는 채택 설정에 OCR이 켜져 있어도 끔
    run_i1._init("indoornav", n, None, opt)
    if cond in ("kept", "apptf"):            # run_i1._init과 같은 순서: 관측 → (bearing_fix theta는 관측 안 바꿈) → track_limit용 track 번호
        from time_filter import load_frames, track_ids
        fr, _ = app_frames(cond)
        obs = rp.frames_obs(fr, run_i1._G["t0"])
        if run_i1._G["opt"].get("track_limit"):
            obs = rp.attach_tracks(obs, track_ids(load_frames(paths.I1 / "obs_nofilter.jsonl")[1]))
        run_i1._G["obs"]["tf"] = obs


def seed_row(r):
    from compare_runs import seed_metrics
    m = rp.metrics(r)
    s = seed_metrics(r)
    return dict(med_all=m["med_all"], p90_all=m["p90_all"], max_all=m["max_all"], wrong=m["wrong"], wrong_s=s["wrong_s"],
                cover2=m["cover2"], in95=m["in95"], n_obs_used=int(r["n_obs"].sum()))


def run_cond(cond, seeds, procs, n, force):
    out = paths.REC_OUT / f"av_{cond}"
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"{paths.RECORDING}: {out.name}가 이미 있음 (--force로 다시)")
    conds = [dict(sc="known", obs="tf", set="기본", seed=s) for s in range(seeds)]
    res = run_pool(procs, _init, (cond, n), run_i1._job, conds)
    _init(cond, n)
    G = run_i1._G
    runs = {c["seed"]: r for c, r, _ in res}
    ms = [rp.metrics(runs[s]) for s in sorted(runs)]
    summ = {"known/tf/기본": rp.summarize(ms, [d for _, _, d in res])}
    per_seed = {s: seed_row(runs[s]) for s in sorted(runs)}
    n_obs_in = sum(len(v) for _, v in G["obs"]["tf"])
    rnd = random_drop(*cond.split("_")[1:2], int(cond.split("_")[2]))[1] if cond.startswith("rnd_") else None
    if cond.startswith("rndu_"):
        rnd = matched_drop(cond.split("_")[1], int(cond.split("_")[2]), n)[1]
        _init(cond, n)
        G = run_i1._G
    if cond == "farwait":
        rnd = dict(n_drop=len(farwait_ids()))
    if cond in WEIGHT:
        rnd = dict(kind=WEIGHT[cond][0], alpha=WEIGHT[cond][1], n_weighted_in=sum(o.get("alpha", 1.0) != 1.0 for _, v in G["obs"]["tf"] for o in v))
    if cond == "eps03":
        rnd = dict(obs_params=run_i1.SETTINGS["기본"]["obs_params"])
    n_used = used_count(G["obs"]["tf"])
    extra = [paths.REC / "decisions.jsonl", paths.REC / "ocr.jsonl", paths.REC / "config.json"]
    if cond == "farwait":                     # 고른 관측의 출처(exp_ocr_obs.py 출력)도 입력 기록에
        extra.append(paths.REC_OUT / "av_obs.json")
    info = dict(name=out.name, cond=cond, random=rnd, n_obs_used=n_used, options=G["opt"], settings={"기본": run_i1.SETTINGS["기본"]}, n=n, seeds=seeds,
                n_obs_in=n_obs_in, provenance=record.run_record(run_i1.STEPS["indoornav"], G["P"].to_dict(), seeds, G["opt"], extra=extra))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(info=info, summary=summ, per_seed=per_seed), ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / "runs.npz", **{f"known_tf_기본_{s}": runs[s] for s in sorted(runs)})
    s = summ["known/tf/기본"]
    print(f"{paths.RECORDING} {cond}: 관측 {n_obs_in}, 오차 중앙 {s['med_all']:.2f} / 90% {s['p90_all']:.2f} / 최대 {s['max_all']:.2f} m, "
          f"자신 있게 틀림 {s['wrong']}/{s['n']}, cover2 {s['cover2']:.3f}", flush=True)


def check_off():
    """off가 기본 재생(exp_d58_r20)과 모든 필드 비트까지 같은지"""
    from regress_replay import diff
    z = np.load(paths.REC_OUT / "av_off" / "runs.npz")
    new = {k: z[k] for k in z.files}
    d = diff(new, paths.REC_OUT / "exp_d58_r20" / "runs.npz")
    ok = sum(v["same"] for v in d.values())
    print(f"{paths.RECORDING} av_off vs exp_d58_r20: 전체 필드 같음 {ok}/{len(d)} seed", flush=True)
    return ok == len(d)


def one(a):
    if a.check_off:                           # 저장된 결과만 비교(파일 안 씀)
        sys.exit(0 if check_off() else 1)
    info = write_decisions()
    link = link_report()
    (paths.REC_OUT / "av_decisions.json").write_text(json.dumps(dict(files=info, link=link), ensure_ascii=False, indent=1), encoding="utf-8")
    t = link["tf"]
    print(f"{paths.RECORDING}: PC 시간 필터 관측 {t['n_obs']}개 중 결정 이어짐 {t['linked']}개 ({t['obs_rate']:.1%}), 결정 {t['n_dec']}개. "
          f"관측 상태 {t['by_state']}, 결정 {t['by_decision']}", flush=True)
    if a.decisions:
        return
    for c in a.cond:
        run_cond(c, a.seeds, a.procs, a.n, a.force)
    if "off" in a.cond and not check_off():
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", nargs="+", default=list(CONDS), choices=CONDS + RND + RNDU + DIAG + EXTRA)
    ap.add_argument("--decisions", action="store_true", help="결정 파일·연결률만")
    ap.add_argument("--all", action="store_true", help="2차 8개 기록")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-off", action="store_true", help="저장된 av_off가 exp_d58_r20과 같은지만")
    ap.add_argument("--rec", nargs="+", default=list(S2_APP), help="--all과 같이: 기록 고르기")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not a.all:
        return one(a)
    bad = 0
    for rec in a.rec:
        cmd = [sys.executable, __file__, "--cond", *a.cond, "--seeds", str(a.seeds), "--procs", str(a.procs), "--n", str(a.n)]
        cmd += ["--decisions"] if a.decisions else []
        cmd += ["--force"] if a.force else []
        cmd += ["--check-off"] if a.check_off else []
        p = subprocess.run(cmd, env={**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8"}, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(f"{rec}: 실패 (코드 {p.returncode})\n{p.stderr[-1500:]}", flush=True)
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
