"""실시간 재생 시험 도우미 (test_rt_replay.py가 기록마다 새 프로세스로 부름: paths가 I1_RECORDING을 불러올 때 읽기 때문).

    I1_RECORDING=<이름> python tests/vision_integration/rt_check_helper.py <검사> --tmp <폴더> [--T 60,76] [--n 500]
검사: oracle | trunc | steps | filter | baro. 마지막 줄 "RESULT <json>"이 결과.
자르기(trunc·filter)는 원본 motion.csv·baro.csv에서 벽시계 T까지 받은 줄만 남긴 녹화 사본을 만들어 app_session.load_app_session으로
다시 읽는다(격자 값이 바로 뒤 원본 표본에 기대는 것까지 실제와 같은 조건).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import replay as rp  # noqa: E402
import rt_pdr  # noqa: E402
import rt_replay as rr  # noqa: E402

STEP_KEYS = ("idx", "t", "t_confirm", "length", "heading", "dheading")
BARO_KEYS = ("t_first", "k_suspect", "k_confirm", "t_end", "k_end", "ride", "delta")


def _params():
    from indoornav.pdr import PDRParams
    return PDRParams.from_toml(paths.PDR_CONFIG)


def _session(rec=None):
    from app_session import load_app_session
    return load_app_session() if rec is None else load_app_session(rec=rec)


def _cut_lines(src, dst, keep):
    """CSV 원본 줄을 글자 그대로 옮기되 keep(줄 dict)이 참인 줄만 (값 글자를 안 바꿔 같은 파싱)"""
    import csv
    with open(src, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    head = next(csv.reader([lines[0]]))
    out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() and keep(dict(zip(head, next(csv.reader([ln]))))):
            out.append(ln)
    Path(dst).write_text("".join(out), encoding="utf-8", newline="")
    return len(out) - 1


def cut_rec(tmp, T, t0):
    """벽시계 T(재생 시각)까지 앱이 받은 원본만 남긴 녹화 폴더 사본 <tmp>/rec:
    motion.csv는 호스트 시각 t ≤ T + t0인 줄, baro.csv·baro_abs.csv는 받은 시각 t_recv ≤ T + t0인 줄, meta.json 그대로"""
    import shutil
    d = Path(tmp) / "rec"
    d.mkdir(parents=True, exist_ok=True)
    Th = T + t0
    n = dict(motion=_cut_lines(paths.REC / "motion.csv", d / "motion.csv", lambda r: float(r["t"]) <= Th),
             baro=_cut_lines(paths.REC / "baro.csv", d / "baro.csv", lambda r: float(r["t_recv"]) <= Th))
    if (paths.REC / "baro_abs.csv").exists():
        n["baro_abs"] = _cut_lines(paths.REC / "baro_abs.csv", d / "baro_abs.csv", lambda r: float(r["t_recv"]) <= Th)
    shutil.copyfile(paths.REC / "meta.json", d / "meta.json")
    return d, n


def _prefix_same(a, b):
    """자른 Session a가 전체 Session b의 앞부분과 비트까지 같은지 (기압 보간 pressure는 걸음에 안 쓰여 뺌)"""
    n = len(a.t)
    return bool(all(np.array_equal(getattr(a, k), getattr(b, k)[:n]) for k in ("t", "acc_user", "gravity", "gyro")))


def _rows_equal(a, b):
    """두 기록 배열이 모든 필드 바이트까지 같은지 → 다른 필드 목록"""
    if len(a) != len(b):
        return ["<len>"]
    return [n for n in a.dtype.names if a[n].tobytes() != b[n].tobytes()]


def _init_pf(n):
    import run_i1
    run_i1._init("indoornav", n, None, json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8")))
    return run_i1


def _filter(seed):
    """run_rt._job과 같은 VisionFilter·TrackLimiter"""
    import run_i1
    from vision_filter import VisionFilter
    from time_filter import TrackLimiter
    G = run_i1._G
    r0 = G["truth"][0]
    st = run_i1.SETTINGS["기본"]
    F = VisionFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                     float(r0["heading_deg"]), seed, st["obs_params"])
    lim = TrackLimiter(G["opt"]["track_limit"]) if G["opt"].get("track_limit") else None
    return F, lim, st


# ---------------------------------------------------------------- 1. 기준 지식 = run_one
def check_oracle(a):
    run_i1 = _init_pf(a.n)
    G = run_i1._G
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    F1, lim1, st = _filter(a.seed)
    r1 = rp.run_one(F1, G["obs"]["tf"], G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"], vision=True,
                    obs_hz=st["obs_hz"], dpsi=G["dpsi"], limiter=lim1)
    F2, lim2, st = _filter(a.seed)
    r2 = rr.run_rt(F2, G["obs"]["tf"], rr.OracleSteps(G["steps"]["known"], G["t0"]), rr.OracleBaro(G["rides"], G["floors"]),
                   G["truth"], G["t0"], G["rides"], vision=True, obs_hz=st["obs_hz"], limiter=lim2,
                   heading=(h["t_host"], h["heading"]))
    names = [n for n, _ in rp.REC_FIELDS]
    bad = ["<len>"] if len(r1) != len(r2) else [n for n in names if r1[n].tobytes() != r2[n].tobytes()]
    tobs = np.array([t for t, _ in G["obs"]["tf"]])          # 프레임 시각(f8). 기록 t 필드는 f4라 w와 직접 비교 안 함
    return dict(n1=len(r1), n2=len(r2), n_frames=len(tobs), bad=bad,
                w_eq_t=bool(len(r2) == len(tobs) and np.array_equal(r2["w"], tobs)),
                t64_eq_frames=bool(len(r2) == len(tobs) and np.array_equal(r2["t64"], tobs)),
                w_eq_t64=bool(np.array_equal(r2["w"], r2["t64"])), fields=list(r2.dtype.names),
                w_minus_t_max=float(np.max(np.abs(r2["w"] - r2["t"].astype("f8")))) if len(r2) else None,
                rides=G["rides"], floors=G["floors"], n_obs_used=int((r1["n_obs"] > 0).sum()))


# ---------------------------------------------------------------- 2·3. 자르기 시험
def build_inputs(s, info, rec, baro, tmp, tag, P0, pending_at=None, bout="wait", no_avail=False, look=rt_pdr.LOOKAHEAD_S):
    """run_rt.prepare·_init과 같은 길(파일 쓰고 다시 읽기 → 같은 반올림)로 걸음·갱신·방향·기압 후보를 만든다.
    rec: 그 Session을 만든 녹화 폴더(motion_avail이 원본 motion.csv 시각을 읽음).
    pending_at: 자른 입력 끝에서 이동 판정을 기다리는 걸음을 '판정은 pending_at(Session 시각)에 남'으로 넣음
    (예전 step_rows가 이런 걸음을 빼던 결함을 피하던 길. 지금 step_rows는 t_known = nan으로 남김). None이면 run_rt.prepare와 같음.
    bout: step_rows 방식("wait" = online 단계, "early" = 묶음 판정 안 기다림).
    no_avail: 음성 대조용 — 받은 시각 대신 격자 시각을 씀(고치기 전 방식: 격자 값이 뒤 원본 표본에 기대는 것을 무시).
    look: 봉우리 뒤 몇 초에 걸음을 판정하나(run_rt 단계의 look, 권장 단계 rec = 0.5)."""
    t0s, t0 = info["t0_sensor"], paths.rec_t0()
    avail = s.t.copy() if no_avail else rt_pdr.motion_avail(rec, s, t0s)
    st, tk = rt_pdr.online_steps(s, P0, look, rt_pdr.WINDOW_S, avail=avail)
    f = Path(tmp) / f"steps_{tag}.csv"
    src = st if pending_at is None else [x if x["moving"] is not None else dict(x, moving=True, t_known=pending_at) for x in st]
    rt_pdr.write_rows(rt_pdr.step_rows(src, t0s, bout), f)
    rows = rt_pdr.read_rows(f)
    np.savez_compressed(Path(tmp) / f"ticks_{tag}.npz", tick_t=tk["tick_t"] + t0s, cut_t=tk["cut_t"] + t0s)
    z = np.load(Path(tmp) / f"ticks_{tag}.npz")
    ticks = (z["tick_t"] - t0, z["cut_t"] - t0)
    cands, smp = rt_pdr.online_baro(baro)
    return dict(st=st, tk=tk, rows=rows, ticks=ticks, cands=cands, smp=smp, avail=avail,
                heading=(s.t + t0s, tk["heading"], avail + t0s))


def _run(inp, obs, suspect, seed, G, max_wait=5.0):
    """run_rt._job과 같은 PF·옵션, 실시간 걸음(OnlineSteps)·기압(OnlineBaro)·받은 방향 표본만(heading 3번째 = 받은 시각)"""
    F, lim, st = _filter(seed)
    sk = rr.OnlineSteps(inp["rows"], G["t0"], *inp["ticks"])
    bk = rr.OnlineBaro(inp["cands"], inp["smp"], suspect=suspect)
    log = {}
    r = rr.run_rt(F, obs, sk, bk, G["truth"], G["t0"], G["rides"], vision=True, obs_hz=st["obs_hz"], limiter=lim,
                  heading=inp["heading"], max_wait=max_wait, log=log)
    return r, log


def _baro_view(c, T):
    """w ≤ T에 보이는 후보 지식"""
    v = dict(t_first=c["t_first"], k_suspect=c["k_suspect"])
    v["k_confirm"] = c["k_confirm"] if c["k_confirm"] is not None and c["k_confirm"] <= T else None
    if c["k_end"] is not None and c["k_end"] <= T:
        v.update(t_end=c["t_end"], k_end=c["k_end"], ride=c["ride"], delta=c["delta"])
    return v


def _first_diff(a, b, T):
    """w ≤ T 행에서 처음 다른 행 (시각, 필드)"""
    for i in range(min(len(a), len(b))):
        f = [n for n in a.dtype.names if a[n][i].tobytes() != b[n][i].tobytes()]
        if f:
            return dict(i=i, t=float(a["t"][i]), w=float(a["w"][i]), fields=f)
    return None


VARIANTS = {"early_hold": ("early", "hold", 5.0), "early_ignore": ("early", "ignore", 5.0),
            "w3_hold": ("wait", "hold", 3.0), "w3_ignore": ("wait", "ignore", 3.0)}     # 이름: (step_rows 방식, 의심, 최대 기다림)
REC_LOOK = 0.5      # 권장 단계 run_rt.STAGES["rec"] = early 걸음 행, 봉우리 뒤 0.5 s, 탑승 의심 무시, 최대 기다림 5 s
REC_VAR = ("early", "ignore", 5.0)


def check_trunc(a):
    run_i1 = _init_pf(a.n)
    G = run_i1._G
    t0 = G["t0"]
    P0 = _params()
    s, info = _session()
    t0s = info["t0_sensor"]
    baro = rt_pdr.read_baro_rt(paths.REC / "baro.csv", t0)
    obs = G["obs"]["tf"]
    full = build_inputs(s, info, paths.REC, baro, a.tmp, "full", P0)
    full_e = build_inputs(s, info, paths.REC, baro, a.tmp, "full_early", P0, bout="early")
    out = dict(T=[], full_cands=[{k: c[k] for k in BARO_KEYS} for c in full["cands"]])
    # 전체 입력 길이 run_rt.prepare가 쓰는 것과 같은지 (같은 함수·같은 반올림, 방향 받은 시각·갱신 포함)
    import run_rt
    prep = Path(a.tmp) / "prep"
    run_rt.prepare(prep, ["both", "both_early"])
    tag = f"l{int(round(rt_pdr.LOOKAHEAD_S * 100)):03d}"
    ref = rt_pdr.read_rows(run_rt.step_file(prep, "online", rt_pdr.LOOKAHEAD_S))
    ref_e = rt_pdr.read_rows(run_rt.step_file(prep, "early", rt_pdr.LOOKAHEAD_S))
    hz = np.load(prep / "heading_rt.npz")
    tz = np.load(prep / f"ticks_{tag}.npz")
    out["prepare_same"] = bool(ref == full["rows"] and ref_e == full_e["rows"]
                               and np.array_equal(hz["t_host"], full["heading"][0])
                               and np.array_equal(hz["avail_host"], full["heading"][2])
                               and np.array_equal(tz["tick_t"] - t0, full["ticks"][0])
                               and np.array_equal(tz["cut_t"] - t0, full["ticks"][1]))
    out["avail_lag_max"] = float(np.max(full["avail"] - s.t))
    # 권장 단계 rec: run_rt.prepare(["rec"])가 쓰는 걸음·갱신 파일 이름과 내용이 시험 입력(early, 봉우리 뒤 0.5 s)과 같은지
    rec_T = {float(x) for x in a.rec_T.split(",")} if a.rec_T else set()
    full_rec = None
    if rec_T:
        sg = run_rt.stage("rec")
        out["rec_stage"] = dict(steps=sg["steps"], look=sg["look"], baro=sg["baro"], max_wait=sg["max_wait"])
        full_rec = build_inputs(s, info, paths.REC, baro, a.tmp, "full_rec", P0, bout="early", look=REC_LOOK)
        prep_r = Path(a.tmp) / "prep_rec"
        run_rt.prepare(prep_r, ["rec"])
        tag_r = f"l{int(round(REC_LOOK * 100)):03d}"
        files = sorted(f.name for f in prep_r.glob("steps_*.csv")) + sorted(f.name for f in prep_r.glob("ticks_*.npz"))
        tz_r = np.load(prep_r / f"ticks_{tag_r}.npz")
        out["rec_prepare"] = dict(files=files,
                                  rows_same=rt_pdr.read_rows(run_rt.step_file(prep_r, "early", REC_LOOK)) == full_rec["rows"],
                                  ticks_same=bool(np.array_equal(tz_r["tick_t"] - t0, full_rec["ticks"][0])
                                                  and np.array_equal(tz_r["cut_t"] - t0, full_rec["ticks"][1])),
                                  differs_from_l075=full_rec["rows"] != full_e["rows"])
    runs_full = {m: _run(full, obs, m, a.seed, G)[0] for m in ("hold", "ignore")}
    full_var = {"early": full_e, "wait": full}
    out["fields"] = list(runs_full["hold"].dtype.names)
    Ts = [float(x) for x in a.T.split(",")] if a.T else []
    if a.auto_suspect:                  # 거짓 탑승 의심 중간 시각
        c = next((c for c in full["cands"] if not c["ride"] and c["k_end"] is not None and c["k_end"] - c["k_suspect"] > 0.3), None)
        if c is not None:
            Ts.append(round((c["k_suspect"] + c["k_end"]) / 2, 3))
    var_T = {float(x) for x in a.var_T.split(",")} if a.var_T else set()
    for T in Ts:
        rec_c, nraw = cut_rec(Path(a.tmp) / f"rec{T:g}", T, t0)
        sc, info_c = _session(rec_c)                        # app_session.load_app_session을 자른 원본에 그대로
        baro_c = rt_pdr.read_baro_rt(rec_c / "baro.csv", t0)
        cut = build_inputs(sc, info_c, rec_c, baro_c, a.tmp, f"cut{T:g}", P0, pending_at=T + t0 - t0s + 100.0)
        drop = build_inputs(sc, info_c, rec_c, baro_c, a.tmp, f"drop{T:g}", P0)      # step_rows 그대로(대기 걸음 = t_known nan)
        ob = [(t, o) for t, o in obs if t <= T]
        d = dict(T=T, n_sensor=len(sc.t), n_raw=nraw, n_baro=len(baro_c), n_obs=len(ob),
                 t0_same=info_c["t0_sensor"] == t0s, session_prefix=_prefix_same(sc, s),
                 baro_rows_same=bool(np.array_equal(baro_c, baro[baro[:, 2] <= T])),
                 avail_prefix=bool(np.array_equal(cut["avail"], full["avail"][:len(sc.t)])),
                 avail_max=float(cut["avail"].max()) + t0s - t0,
                 heading_prefix=bool(np.array_equal(cut["heading"][1], full["heading"][1][:len(sc.t)])))
        # 갱신: 자른 입력의 갱신 = 전체 입력에서 갱신 시각 ≤ T인 것
        mt = full["ticks"][0] <= T
        d["ticks_same"] = bool(np.array_equal(cut["ticks"][0], full["ticks"][0][mt])
                               and np.array_equal(cut["ticks"][1], full["ticks"][1][mt]))
        # 걸음 단계
        fs, cs = full["st"], cut["st"]
        Ts_ = T + t0 - t0s                                  # Session 시각으로 된 T
        fc = [tuple(x[k] for k in STEP_KEYS) for x in fs if x["t_confirm"] <= Ts_]
        cc = [tuple(x[k] for k in STEP_KEYS) for x in cs if x["t_confirm"] <= Ts_]
        fk = [(x["moving"], x["t_known"]) for x in fs if x["t_known"] is not None and x["t_known"] <= Ts_]
        ck = [(x["moving"], x["t_known"]) for x in cs if x["t_known"] is not None and x["t_known"] <= Ts_]
        d.update(steps_confirm_same=fc == cc, n_confirm=len(fc), steps_known_same=fk == ck, n_known=len(fk),
                 n_pending_cut=sum(x["moving"] is None for x in cs))
        # 기압 단계
        fv = [_baro_view(c, T) for c in full["cands"] if c["k_suspect"] <= T]
        cv = [_baro_view(c, T) for c in cut["cands"] if c["k_suspect"] <= T]
        d.update(baro_same=fv == cv, n_cands=len(fv), baro_views=cv)
        # PF 단계
        for m in ("hold", "ignore"):
            rf = runs_full[m]
            rc, lg = _run(cut, ob, m, a.seed, G)
            A, B = rf[rf["w"] <= T], rc[rc["w"] <= T]
            rd, _ = _run(drop, ob, m, a.seed, G)
            D = rd[rd["w"] <= T]
            d[m] = dict(n_full=len(A), n_cut=len(B), bad=_rows_equal(A, B), first=_first_diff(A, B, T),
                        n_full_after=int((rf["w"] > T).sum()), forced_cut=lg["forced"],
                        drop_n=len(D), drop_bad=_rows_equal(A, D))
        # 다른 단계(묶음 판정 안 기다림 early, 최대 기다림 3 s)도 같은 자르기 비교 (--var-T로 지정한 T에서만)
        if T in var_T:
            d["var"] = {}
            cut_var = {"early": build_inputs(sc, info_c, rec_c, baro_c, a.tmp, f"cute{T:g}", P0, bout="early"), "wait": drop}
            for nm, (bt, m, mw) in VARIANTS.items():
                if nm not in runs_full:
                    runs_full[nm] = _run(full_var[bt], obs, m, a.seed, G, mw)[0]
                rf = runs_full[nm]
                rc, lg = _run(cut_var[bt], ob, m, a.seed, G, mw)
                A, B = rf[rf["w"] <= T], rc[rc["w"] <= T]
                H = runs_full[m][runs_full[m]["w"] <= T]
                d["var"][nm] = dict(n_full=len(A), n_cut=len(B), bad=_rows_equal(A, B), first=_first_diff(A, B, T),
                                    n_full_after=int((rf["w"] > T).sum()), forced_cut=lg["forced"],
                                    differs_from_base=_rows_equal(A, H))
            # 음성 대조: 받은 시각 없이(격자 시각 = 받은 시각으로 봄) 같은 비교를 하면 T 전 출력이 달라져야 함
            fn = build_inputs(s, info, paths.REC, baro, a.tmp, f"fna{T:g}", P0, no_avail=True)
            cn = build_inputs(sc, info_c, rec_c, baro_c, a.tmp, f"cna{T:g}", P0, no_avail=True)
            rf, rc = _run(fn, obs, "hold", a.seed, G)[0], _run(cn, ob, "hold", a.seed, G)[0]
            A, B = rf[rf["w"] <= T], rc[rc["w"] <= T]
            d["neg_avail"] = dict(n_full=len(A), n_cut=len(B), bad=_rows_equal(A, B),
                                  ticks_extra=int((fn["ticks"][0] <= T).sum()) - len(cn["ticks"][0]))
        # 권장 단계 rec(early 행·봉우리 뒤 0.5 s·의심 무시·최대 기다림 5 s)도 같은 자르기 비교 (--rec-T로 지정한 T에서만)
        if T in rec_T:
            bt, m, mw = REC_VAR
            if "rec" not in runs_full:
                runs_full["rec"] = _run(full_rec, obs, m, a.seed, G, mw)[0]
            rf = runs_full["rec"]
            cut_r = build_inputs(sc, info_c, rec_c, baro_c, a.tmp, f"cutr{T:g}", P0, bout=bt, look=REC_LOOK)
            rc, lg = _run(cut_r, ob, m, a.seed, G, mw)
            A, B = rf[rf["w"] <= T], rc[rc["w"] <= T]
            H = runs_full[m][runs_full[m]["w"] <= T]                       # 기본 단계(both_ignore와 같은 입력)
            E = runs_full["early_ignore"] if "early_ignore" in runs_full else None
            d["rec"] = dict(n_full=len(A), n_cut=len(B), bad=_rows_equal(A, B), first=_first_diff(A, B, T),
                            n_full_after=int((rf["w"] > T).sum()), forced_cut=lg["forced"],
                            differs_from_base=_rows_equal(A, H),
                            differs_from_early_ignore=None if E is None else _rows_equal(A, E[E["w"] <= T]))
        # 3. 음성 대조: 오프라인 걸음(전체 filtfilt)·기압(표본 시각으로 자름)
        off_f = rt_pdr.offline_rows(s, P0, t0s)
        off_c = rt_pdr.offline_rows(sc, P0, t0s)
        Th = T + t0
        kf = [tuple(r[k] for k in ("idx", "length_m", "heading_deg", "moving")) for r in off_f if r["t"] <= Th]
        kc = [tuple(r[k] for k in ("idx", "length_m", "heading_deg", "moving")) for r in off_c if r["t"] <= Th]
        nd = sum(x != y for x, y in zip(kf, kc)) + abs(len(kf) - len(kc))
        big = [dict(idx=x[0], t=round(float(s.t[x[0]]) + t0s - t0, 3), dlen=round(y[1] - x[1], 4), moving=(x[3], y[3]))
               for x, y in zip(kf, kc) if abs(x[1] - y[1]) > 1e-3 or x[3] != y[3] or x[0] != y[0]]
        rf_, ff_ = rp.baro_events(baro[:, :2])
        rc_, fc_ = rp.baro_events(baro[baro[:, 0] <= T][:, :2])
        clip = lambda rs: [(x, min(y, T)) for x, y in rs if x <= T]
        d["neg"] = dict(off_steps_diff=int(nd), n_off=len(kf), n_off_cut=len(kc), off_big=big, rides_full=clip(rf_), rides_cut=clip(rc_),
                        floors_full=[f for f in ff_ if f[0] <= T], floors_cut=[f for f in fc_ if f[0] <= T])
        d["neg"]["baro_diff"] = d["neg"]["rides_full"] != d["neg"]["rides_cut"] or d["neg"]["floors_full"] != d["neg"]["floors_cut"]
        # PF 단계 음성 대조: 오프라인 걸음 + 기준 기압을 자른 입력으로 (봉우리 순간에 안다고 봄 → w = t)
        def oracle_run(rows, rides, floors, ob_):
            F, lim, st = _filter(a.seed)
            return rr.run_rt(F, ob_, rr.OracleSteps(rows, t0), rr.OracleBaro(rides, floors), G["truth"], t0, G["rides"],
                             vision=True, obs_hz=st["obs_hz"], limiter=lim, heading=full["heading"])
        of = oracle_run(rp.indoornav_sign([dict(r) for r in off_f]), rf_, ff_, obs)
        oc = oracle_run(rp.indoornav_sign([dict(r) for r in off_c]), rc_, fc_, ob)
        A, B = of[of["w"] <= T], oc[oc["w"] <= T]
        d["neg"]["pf_bad"] = _rows_equal(A, B)
        d["neg"]["pf_first"] = _first_diff(A, B, T)
        out["T"].append(d)
    return out


# ---------------------------------------------------------------- 4. 걸음 판정 속
def check_steps(a):
    from indoornav.pdr import run_pdr
    from indoornav.pdr.pipeline import walk_signal
    P0 = _params()
    s, info = _session()
    st, tk = rt_pdr.online_steps(s, P0)
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    early = rt_pdr.step_rows(st, info["t0_sensor"], "early")
    out = dict(n=len(st), check_bout=rt_pdr.check_bout(st, P0), heading_same=bool(np.array_equal(tk["heading"], h["heading"])),
               t_host_same=bool(np.array_equal(s.t + info["t0_sensor"], h["t_host"])),
               early_n=len(early), early_n_confirmed=len(st), early_all_moving=all(r["moving"] for r in early),
               early_known_eq=all(r["t_known"] == r["t_confirm"] for r in early))
    if not a.sig:
        return out
    team = run_pdr(s, P0).steps                # indoornav 오프라인 설정(filtfilt) 결과
    sig = walk_signal(s, P0)
    st2, tk2 = rt_pdr.online_steps(s, P0, a.look, 1e5, sig=sig)
    last_cut = float(tk2["cut_t"][-1])
    tidx = team["idx"].to_numpy()
    tin = tidx[team["t"].to_numpy() <= last_cut]     # 녹화 끝 look+0.5 s 안 봉우리는 판정할 갱신이 없음
    on_idx = [x["idx"] for x in st2]
    tm = dict(zip(tidx.tolist(), team["moving"].tolist()))
    out.update(sig_idx_same=on_idx == tin.tolist(), n_team=len(tidx), n_team_in=len(tin), n_online=len(on_idx),
               team_after=len(tidx) - len(tin), last_cut=last_cut, t_end=float(s.t[-1]),
               n_moving_decided=sum(x["moving"] is not None for x in st2),
               moving_bad=int(sum(x["moving"] != tm.get(x["idx"]) for x in st2 if x["moving"] is not None)),
               undecided=sum(x["moving"] is None for x in st2))
    return out


# ---------------------------------------------------------------- 5. 지연 필터 인과성
def check_filter(a):
    from indoornav.pdr.pipeline import walk_signal
    P0 = _params()
    p = rt_pdr.delayed_params(P0)
    s, info = _session()
    full = walk_signal(s, p)
    d, hop = int(round(p.preprocess.delay_s * p.fs)), max(1, int(round(p.preprocess.hop_s * p.fs)))
    out = []
    for T in [float(x) for x in a.T.split(",")]:
        rec_c, _ = cut_rec(Path(a.tmp) / f"rec{T:g}", T, paths.rec_t0())
        sc, _ = _session(rec_c)                          # 원본 motion.csv를 자르고 load_app_session으로 다시 만듦
        cs = walk_signal(sc, p)
        now = (len(sc.t) // hop) * hop
        e = now - d
        out.append(dict(T=T, n=len(sc.t), e=e, same=bool(np.array_equal(cs[:e], full[:e])), prefix=_prefix_same(sc, s),
                        tail_diff=bool(not np.array_equal(cs[e:], full[e:len(cs)]))))
    return dict(d=d, hop=hop, cuts=out)


# ---------------------------------------------------------------- 6. 실시간 기압 = 오프라인 규칙
def check_baro(a):
    t0 = paths.rec_t0()
    b = rt_pdr.read_baro_rt(paths.REC / "baro.csv", t0)
    rides, floors = rp.baro_events(b[:, :2])
    cands, smp = rt_pdr.online_baro(b)
    return dict(rides=rides, floors=floors, t_last=float(b[-1, 0]),
                cands=[{k: c[k] for k in BARO_KEYS} for c in cands],
                causal_order=bool(all((c["k_confirm"] is None or c["k_suspect"] <= c["k_confirm"]) and
                                      (c["k_end"] is None or (c["k_confirm"] or c["k_suspect"]) <= c["k_end"]) for c in cands)))


# ---------------------------------------------------------------- 7. run_rt.prepare 짧은 입력·방향 파일
def _session_view(s, n=None, gyro_bias=0.0):
    """Session 앞 n표본(없으면 전체) 사본. gyro_bias: z축 자이로에 더하는 값(rad/s, 방향이 저장된 heading_indoornav와 달라지게)"""
    import dataclasses
    n = len(s.t) if n is None else n
    g = s.gyro[:n].copy()
    g[:, 2] += gyro_bias
    return dataclasses.replace(s, t=s.t[:n].copy(), acc_user=s.acc_user[:n].copy(), gravity=s.gravity[:n].copy(), gyro=g,
                               pressure=None if s.pressure is None else s.pressure[:n].copy())


def _prepare_with(s, info, out, names):
    """app_session.load_app_session이 s를 돌려주게 바꾼 채로 run_rt.prepare(out, names)"""
    import app_session
    import run_rt
    orig = app_session.load_app_session
    app_session.load_app_session = lambda *a_, **k_: (s, info)
    try:
        return run_rt.prepare(out, names)
    finally:
        app_session.load_app_session = orig


def check_prep_short(a):
    """짧은 Session(앞 2 s, 첫 걸음 판정 직후 = 판정만 되고 이동 판정 전)으로 prepare가 멈추지 않고 이동 판정 통계는 None"""
    P0 = _params()
    s, info = _session()
    look = rt_pdr.LOOKAHEAD_S
    out = dict(cases=[])
    first = None
    for sec in np.arange(2.0, 40.0, 0.5):           # 걸음이 판정되고 아직 이동 판정이 없는 가장 짧은 앞부분
        st, _ = rt_pdr.online_steps(_session_view(s, int(sec * s.fs)), P0, look, rt_pdr.WINDOW_S)
        if st and all(x["moving"] is None for x in st):
            first = float(sec)
            break
    for sec in [2.0] + ([first] if first is not None else []):
        sv = _session_view(s, int(sec * s.fs))
        try:
            chk = _prepare_with(sv, info, Path(a.tmp) / f"p{sec:g}", ["both"])
        except Exception as e:                      # noqa: BLE001 — 시험이 원인을 보게 그대로 돌려줌
            return dict(error_in_prepare=f"{sec} s: {type(e).__name__}: {e}")
        out["cases"].append(dict(sec=sec, chk=chk[f"l{look}"]))
    out["first"] = first
    return out


def check_prep_heading(a):
    """방향이 저장된 heading_indoornav.npz와 다른 Session(z 자이로 +0.01 rad/s)으로 prepare → heading_rt.npz 방향 = 그 Session의
    online_steps 방향, run_rt._init 뒤 실시간 단계 heading_for = 그 파일 값(받은 시각 포함), 미리 아는 단계 = heading_indoornav"""
    import run_rt
    P0 = _params()
    s0, info = _session()
    s = _session_view(s0, gyro_bias=0.01)
    out_dir = Path(a.tmp) / "prep"
    _prepare_with(s, info, out_dir, ["both"])
    t0s = info["t0_sensor"]
    avail = rt_pdr.motion_avail(paths.REC, s, t0s)
    _, tk = rt_pdr.online_steps(s, P0, rt_pdr.LOOKAHEAD_S, rt_pdr.WINDOW_S, avail=avail)
    z = np.load(out_dir / "heading_rt.npz")
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    res = dict(n=len(s.t), file_eq_online=bool(np.array_equal(z["heading"], tk["heading"])),
               file_t_eq=bool(np.array_equal(z["t_host"], s.t + t0s)), file_avail_eq=bool(np.array_equal(z["avail_host"], avail + t0s)),
               differs_from_indoornav=bool(len(h["heading"]) != len(tk["heading"]) or not np.array_equal(h["heading"], tk["heading"])))
    run_rt._init(str(out_dir), a.n)
    rt = run_rt.heading_for(run_rt.stage("both_ignore"))
    off = run_rt.heading_for(run_rt.stage("off"))
    res.update(rt_len=len(rt), rt_eq=bool(len(rt) == 3 and all(np.array_equal(x, z[k]) for x, k in
                                                                 zip(rt, ("t_host", "heading", "avail_host")))),
               off_len=len(off), off_eq=bool(len(off) == 2 and np.array_equal(off[0], h["t_host"])
                                             and np.array_equal(off[1], h["heading"])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["oracle", "trunc", "steps", "filter", "baro", "prep_short", "prep_heading"])
    ap.add_argument("--tmp", default=".")
    ap.add_argument("--T", default="")
    ap.add_argument("--auto-suspect", action="store_true")
    ap.add_argument("--var-T", default="", help="trunc: early·최대 기다림 3 s 단계도 비교할 T (쉼표)")
    ap.add_argument("--rec-T", default="", help="trunc: 권장 단계 rec(early·봉우리 뒤 0.5 s·의심 무시·최대 기다림 5 s)도 비교할 T (쉼표)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--look", type=float, default=3.0)
    ap.add_argument("--sig", action="store_true")
    a = ap.parse_args()
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    res = globals()[f"check_{a.check}"](a)
    sys.stdout.reconfigure(encoding="utf-8")
    print("RESULT " + json.dumps(dict(rec=paths.RECORDING, **res), ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    main()
