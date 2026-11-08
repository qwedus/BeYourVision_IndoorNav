"""과신 원인 분리: 실제 걸음·PF·관측 시각·추적 번호는 그대로 두고, 비전 관측의 방위만 바꿔 재생한다.
기본 설정(configs, vision_options.json)은 안 바꿈, 채택 실험 아님. 관측을 만들 때 정답을 쓰므로 진단 전용.

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_overconf.py --name ovc
    ... --scen real S0 S2 --freq tl1 tl5      조건·관측 빈도 고르기 (기본: 전부)
    ... --check                               잡음 0 합성 관측의 정답 자리 잔차가 0인지, real·tl1이 기본 재생(exp_d58_r20)과 비트까지 같은지
    ... --residuals                           실제 맞는 관측의 정답 자리 방위 잔차 구조만 계산 (재생 안 함, <name>/residuals.json)
    ... --cross                               9개 기록 residuals.json으로: 다른 기록에서 맞춘 판별 치우침이 남긴 기록의 판별 잔차를 설명하나

맞는 관측: 관측마다 정답 자리·정답 카메라 방향(자이로 yaw + 곧은 구간 보정, analyze_recording.camera_offset)에서
    PF 후보(가시성 표·20 m·화각·앙각, 관측 모델 기본값)인 같은 종류 판 행 중 방위 잔차가 가장 작은 행을 짝으로 하고,
    잔차 ≤ 10°면 '맞는 관측'(analyze_recording PF 후보 일치와 같은 정의). 화면 끝 관측은 판 끝점 행과 짝지음.
    지나감(pass) = 같은 판의 맞는 관측이 3 s 넘게 끊기지 않고 이어진 묶음.
조건 (scen). S1~S4는 S0에 원인 하나만 더함
    real  실제 관측 그대로 (tl1 = 기본 재생)
    S0    맞는 관측만, 방위 = 정답 방위 + N(0, 5°) 독립 (관측 모델 σ와 같음). 편향·상관·잘못된 대응 없음 → 대조
    S1    S0와 전체 크기(5°)는 같고, 그중 비율 ρ를 지나감마다 공통 잡음으로: N(0, 5√ρ) + 독립 N(0, 5√(1-ρ))  → (a) 같은 판 반복 관측의 상관
    S2    S1과 같은 크기, 공통 부분을 판마다 고정(기록 전체에서 같음)                                → (b) 판별 방위 편향
          ρ 기본값 = 그 기록 실제 맞는 관측 잔차에서 (판 평균² + 판 안 지나감 평균² + 전체 평균²) ÷ (전체 제곱 평균)
    S3    S0, 단 판마다 확률 30%로 무작위 방향 1~3 m 옮긴 자리로 방위 계산 (PF 지도는 그대로) → (b) 지도 판 위치 오차
    S4    S0 + 실제 맞지 않는 관측 자리(시각·종류·추적 번호)에 화각 안 무작위 방위 U(±30°) → (c) 무작위 잘못된 대응
    H1    맞는 관측만, 실제 방위 (실제 잔차 구조, 잘못된 대응 뺌)
    H2    S0 + 실제 맞지 않는 관측(실제 방위)                     → 실제 잘못된 대응·먼 판만 넣음
    H3    H1 − 판별 실제 잔차 중앙 (판별 편향만 없앰. 중앙은 재생 모집단에서: 시간 필터 통과·탑승 아님, add는 버린 것 포함)
    H4    H1 잔차를 같은 모집단(시간 필터 통과 / 버린 것 따로, 탑승 아님) 안에서 무작위로 섞음 (시간·판 구조 없앰, 분포 유지)
    H5    H1 − 같은 모집단 실제 잔차 평균 (공통 치우침만 없앰, 판별 차이는 남김)
잔차 통계(ρ, H3·H4·H5 기준, --residuals, --cross)는 PF가 쓰는 관측만: 기압 탑승 구간 관측은 뺌.
    H2f   S0 + 실제 맞지 않는 관측 중 정답 자리에 같은 종류 PF 후보가 아예 없는 것만 (20 m 밖·안 보이는 판)  (기본 목록에 없음)
    H2n   S0 + 실제 맞지 않는 관측 중 후보는 있으나 10° 밖인 것만                                           (기본 목록에 없음)
--eps x: 관측 모델 바닥값만 바꿔 재생 (우도 완화 진단, 기본 0.1). 결과 폴더를 따로 쓸 것
관측 빈도 (freq): tl0.5·tl1·tl5 = 같은 track N초에 1번 / add = tl1 + 시간 필터가 버린 맞는 관측 되살림
    (같은 track 기존 입력과 1 s 안이면 안 넣음, 추가끼리도 1 s에 1번: run_negative.AddOnly, 판정 때 restore_add와 같은 규칙)
정답 카메라 방향 (--cam): gyro = 자이로 + 기록 보정 하나(기본) / path = 정답 진행 방향에 맞춰 앞뒤 10 s로 천천히 바뀌는 보정.
잡음 난수: (seed, 조건)마다 정함. 관측 하나의 잡음은 빈도가 달라도 같음. 같은 seed = 같은 초기 파티클.
출력 <녹화 out>/<name>/: runs.npz(키 <scen>_<freq>_<seed>), summary.json(조건별 seed 지표·중앙, 관측 수, 잔차 구조, 입력 기록)
"""
import argparse
import json
import sys
import time
import zlib

import numpy as np

import paths
import provenance as record
import replay as rp
import run_i1
from analyze_recording import camera_offset
from compare_runs import seed_metrics
from edge_faces import base_id
from obs_model import DEFAULT, _cells, _kind_rows, kind_types
from pool_guard import run_pool
from run_negative import AddOnly
from time_filter import TrackLimiter

TOL, SIG, PASS_GAP, FOV, CAM_WIN = 10.0, DEFAULT["sigma_bearing"], 3.0, 30.0, 10.0
SCENS = ("real", "S0", "S1", "S2", "S3", "S4", "H1", "H2", "H3", "H4", "H5")
EXTRA = ("H2f", "H2n")
FREQS = {"tl0.5": 0.5, "tl1": 1.0, "tl5": 5.0, "add": 1.0}
WITH_UNMATCHED = ("real", "H2", "S4")         # 맞지 않는 관측도 넣는 조건
REAL_MATCHED = ("real", "H1")                  # 맞는 관측을 실제 방위로
SEGS = {"s2_b_emart_ex5": [(118, 168)], "s2_a_ex3_ex6": [(203, 233)], "s2_a_ex6_ex3": [(38, 53), (75.6, 111.7)],
        "s2_a_71_ex2": [(92, 116)]}            # 판정의 문제 구간 (구간 지표용)
_S = {}


def wrap(a):
    return (np.asarray(a, float) + 180) % 360 - 180


# ---------------------------------------------------------------- 짝짓기
def camera_track(mode=None):
    """정답 카메라 방향 함수와 보정값 (나침반 도).
    gyro: cam(t) = 자이로 yaw(t) + off (기록 하나에 보정 하나, analyze_recording과 같음)
    path: cam(t) = yaw(t) + off(t). off(t) = 앞뒤 CAM_WIN초 곧은 구간에서 (정답 진행 방향 - yaw)의 중앙 (표본이 적으면 off).
          자이로가 천천히 틀어지거나 정답 경로 방향과 어긋나는 몫을 빼고, 흔들림·회전은 yaw 그대로"""
    G = run_i1._G
    mode = mode or _S.get("cam", "gyro")
    h = np.load(paths.REC_OUT / "heading_indoornav.npz")
    ty, yaw = h["t_host"] - G["t0"], -np.degrees(np.unwrap(h["heading"]))
    off = camera_offset(G["truth"], paths.RC["waypoints"], ty, yaw)
    if mode == "gyro":
        return (lambda t: float((np.interp(t, ty, yaw) + off) % 360)), off
    tr = G["truth"]
    t = tr["t"]
    tw = np.array([w["t"] for w in paths.RC["waypoints"]])
    ok = (np.min(np.abs(t[:, None] - tw[None]), 1) > 3) & (tr["on_esc"] == 0) & (t > 1)
    d = wrap(tr["heading_deg"] - np.interp(t, ty, yaw) - off)
    loc = np.array([np.median(d[ok & (np.abs(t - ti) <= CAM_WIN)]) if (ok & (np.abs(t - ti) <= CAM_WIN)).sum() >= 10 else 0.0
                    for ti in t])
    return (lambda q: float((np.interp(q, ty, yaw) + off + np.interp(q, t, loc)) % 360)), off


def match_one(fm, o, x, y, cam):
    """관측 하나 → (행, 상대 방위, 잔차, 수평 거리) 또는 None (정답 자리에 같은 종류 PF 후보 없음)"""
    k = np.where(np.isin(fm["lm_type"], kind_types(o)))[0]
    if not len(k):
        return None
    X, Y, TH = np.array([x]), np.array([y]), np.array([cam])
    i, j, ins = _cells(fm, X, Y)
    cand, hd, rel = (a[:, 0] for a in _kind_rows(fm, k, X, Y, TH, i, j, ins, DEFAULT))
    if not cand.any():
        return None
    res = wrap(o["bearing"] - rel)
    a = np.flatnonzero(cand)[np.argmin(np.abs(res[cand]))]
    return int(k[a]), float(rel[a]), float(res[a]), float(hd[a])


def match_table(frames):
    """관측 id → 짝 정보. frames: [(t, [obs])] (필터 없는 관측 전부)"""
    G = run_i1._G
    cam_at, off = camera_track()
    tab = {}
    for t, obs in frames:
        tr = rp.truth_at(G["truth"], t)
        fl = int(tr["floor"])
        fm = G["fmap"].get(fl)
        x, y, cam = float(tr["east"]), float(tr["north"]), cam_at(t)
        for o in obs:
            e = dict(t=float(t), match=False, track=o.get("track_id"), walk=not any(a <= t <= b for a, b in G["rides"]))
            m = match_one(fm, o, x, y, cam) if fm is not None else None
            if m is not None:
                row, rel, res, hd = m
                e.update(floor=fl, row=row, lm=base_id(fm["lm_id"][row]), rel=rel, res=res, hd=hd, x=x, y=y, cam=cam,
                         match=abs(res) <= TOL)
            tab[o["detection_id"]] = e
    by = {}
    for d in sorted(tab, key=lambda d: (tab[d]["t"], d)):
        if tab[d]["match"]:
            by.setdefault(tab[d]["lm"], []).append(d)
    n = 0
    for lm in sorted(by):                       # 지나감 번호
        last = None
        for d in by[lm]:
            if last is None or tab[d]["t"] - last > PASS_GAP:
                n += 1
            tab[d]["pass"] = n
            last = tab[d]["t"]
    return tab, off


def matched_rows(tab, ids):
    """맞는 관측 한 줄씩 (판 위치 맞춤용): 시각·판·정답 자리·판 행 위치·거리·잔차"""
    G = run_i1._G
    out = []
    for d in sorted(ids):
        e = tab[d]
        if e["match"]:
            lx, ly = map(float, G["fmap"][e["floor"]]["lm_xy"][e["row"]])
            out.append(dict(t=e["t"], lm=e["lm"], x=e["x"], y=e["y"], lx=lx, ly=ly, hd=e["hd"], res=e["res"]))
    return out


def cross(name, min_n=5, lams=(1.0, 9.0, 100.0)):
    """판별 치우침이 걸음마다 같은가 (탑승 중 관측 뺀 시간 필터 통과 맞는 관측, (기록, 판) 묶음이 min_n개 이상인 것만, 학습·평가 모두).
    모형: 잔차 = 기록 공통 항 + 판별 항 + 잡음. 판별 항 M1 = 판마다 상수 / M2 = 판 위치 옮김(2D, 능선 lam [도²/m²]).
    학습 기록으로 기록 공통 항과 판별 항을 함께 맞추고(번갈아 20번), 남긴 기록에서는 그 기록 공통 항을 예측과 함께 다시 맞춘 뒤
    (판, 기록) 묶음 평균이 줄어드는 비율을 냄. 기준(예측 없음) = 남긴 기록 평균만 뺀 묶음 평균. 같은 묶음에서 표본 안(모든 기록으로 맞춤) 값도 냄"""
    from provenance import RECORDINGS
    rows = []
    for rec in RECORDINGS:
        rc = json.loads((paths.RECORDINGS / f"{rec}.json").read_text(encoding="utf-8"))
        for o in json.loads((paths.ROOT / rc["out"] / name / "residuals.json").read_text(encoding="utf-8"))["rows"]:
            u, v = o["lx"] - o["x"], o["ly"] - o["y"]
            h2 = u * u + v * v
            rows.append((rec, o["lm"], o["res"], np.degrees(v / h2), np.degrees(-u / h2)))
    rec_a = np.array([r[0] for r in rows])
    lm_a = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    A = np.array([[r[3], r[4]] for r in rows])
    grp = np.array([f"{r}|{l}" for r, l in zip(rec_a, lm_a)])
    g_u, g_c = np.unique(grp, return_counts=True)
    keep = np.isin(grp, g_u[g_c >= min_n])
    rec_a, lm_a, y, A, grp = rec_a[keep], lm_a[keep], y[keep], A[keep], grp[keep]

    def pred(par, kind, idx):
        out = np.zeros(int(idx.sum()))
        for l, p in par.items():
            s = lm_a[idx] == l
            out[s] = p if kind == "M1" else A[idx][s] @ p
        return out

    def fit(mask, kind, lam):
        c = {r: float(y[mask & (rec_a == r)].mean()) for r in np.unique(rec_a[mask])}
        par = {}
        for _ in range(20):
            res = y - np.array([c.get(r, 0.0) for r in rec_a])
            for l in np.unique(lm_a[mask]):
                q = mask & (lm_a == l)
                par[l] = float(res[q].mean()) if kind == "M1" else np.linalg.solve(A[q].T @ A[q] + lam * np.eye(2), A[q].T @ res[q])
            pr = np.zeros(len(y))
            pr[mask] = pred(par, kind, mask)
            c = {r: float((y - pr)[mask & (rec_a == r)].mean()) for r in c}
        return par

    def groups(rec, par, kind):
        idx = (rec_a == rec) & np.isin(lm_a, list(par))
        if not idx.any():
            return [], []
        p = pred(par, kind, idx)
        yy, gg = y[idx], grp[idx]
        c0, c1 = yy.mean(), (yy - p).mean()
        u = np.unique(gg)
        return [float((yy[gg == g] - c0).mean()) for g in u], [float((yy[gg == g] - c1 - p[gg == g]).mean()) for g in u]

    def summ(b, a):
        b, a = np.array(b), np.array(a)
        return dict(groups=len(b), rms_before=float(np.sqrt((b ** 2).mean())), rms_after=float(np.sqrt((a ** 2).mean())),
                    explained=float(1 - (a ** 2).mean() / (b ** 2).mean()))

    out = dict(rows=len(y), groups_total=len(np.unique(grp)))
    for kind, lam in [("M1", 0.0)] + [("M2", l) for l in lams]:
        key = kind if kind == "M1" else f"M2_lam{lam:g}"
        full = fit(np.ones(len(y), bool), kind, lam)
        b, a, bi, ai = [], [], [], []
        for rec in np.unique(rec_a):
            par = fit(rec_a != rec, kind, lam)
            x, z = groups(rec, par, kind)
            b += x
            a += z
            sub = {l: full[l] for l in par}                             # 표본 안도 남긴 기록 예측과 같은 묶음에서
            x, z = groups(rec, sub, kind)
            bi += x
            ai += z
        out[key + "_loo"] = summ(b, a)
        out[key + "_in"] = summ(bi, ai)
        if kind == "M2":
            mags = [float(np.hypot(*v)) for v in full.values()]
            out[key + "_in"].update(shift_med_m=float(np.median(mags)), shift_max_m=float(max(mags)))
    return out


def residual_structure(tab, ids):
    """맞는 관측(ids 중) 잔차 구조 [도]. 판 평균·지나감 평균으로 나눈 퍼짐과 지나감 안 이웃 상관"""
    e = sorted((tab[d] for d in sorted(ids) if tab[d]["match"]), key=lambda x: x["t"])    # 같은 시각은 관측 id 순 (실행마다 같게)
    if len(e) < 3:
        return dict(n=len(e))
    r = np.array([x["res"] for x in e])
    lm = np.array([x["lm"] for x in e])
    ps = np.array([x["pass"] for x in e])
    lm_m = {l: r[lm == l].mean() for l in set(lm)}
    ps_m = {p: r[ps == p].mean() for p in set(ps)}
    a_lm = np.array([lm_m[l] for l in lm])
    a_ps = np.array([ps_m[p] for p in ps])
    within = r - a_ps
    by = {}                                     # 이웃 상관: 지나감별로 시각마다 평균 낸 잔차의 이웃 시각 쌍 (같은 시각 관측 순서와 무관)
    for x, w in zip(e, within):
        q = by.setdefault((x["pass"], x["t"]), [[], []])
        q[0].append(x["res"] - r.mean())
        q[1].append(w)
    seq = sorted(by)
    adj = [(a, b) for a, b in zip(seq, seq[1:]) if a[0] == b[0]]
    pairs = [(np.mean(by[a][1]), np.mean(by[b][1])) for a, b in adj]
    raw_pairs = [(np.mean(by[a][0]), np.mean(by[b][0])) for a, b in adj]
    corr = lambda p: float(np.corrcoef(np.array(p).T)[0, 1]) if len(p) > 5 else None
    per_lm = {l: dict(n=int((lm == l).sum()), med=float(np.median(r[lm == l])), passes=len(set(ps[lm == l]))) for l in sorted(lm_m)}
    n_ps = np.array([(ps == p).sum() for p in sorted(ps_m)])
    return dict(n=len(e), mean=float(r.mean()), median=float(np.median(r)), sd=float(r.std()),
                sd_lm=float((a_lm - r.mean()).std()), sd_pass_in_lm=float((a_ps - a_lm).std()), sd_within=float(within.std()),
                lag1_within=corr(pairs), lag1_raw=corr(raw_pairs), n_lm=len(lm_m), n_pass=len(ps_m),
                obs_per_pass_med=float(np.median(n_ps)), per_lm=per_lm)


# ---------------------------------------------------------------- 합성 관측
def draws(scen, seed, tab, prm):
    """(seed, 조건)마다 정해지는 난수. 관측 id·지나감·판 정렬 순서로 뽑아 빈도와 상관없이 같은 관측은 같은 값"""
    rng = np.random.default_rng([seed, zlib.crc32(scen.encode())])
    ids = sorted(tab)
    m_ids = [d for d in ids if tab[d]["match"]]
    lms = sorted({tab[d]["lm"] for d in m_ids})
    pss = sorted({tab[d]["pass"] for d in m_ids})
    split = scen in ("S1", "S2")                # 전체 σ는 같게, 비율 ρ만 공통 부분으로
    z = dict(zip(ids, rng.normal(0, prm["sig"] * (np.sqrt(1 - prm["rho"]) if split else 1.0), len(ids))))
    u = dict(zip(ids, rng.uniform(-FOV, FOV, len(ids))))
    a_ps = dict(zip(pss, rng.normal(0, prm["sig"] * np.sqrt(prm["rho"]), len(pss))))
    b_lm = dict(zip(lms, rng.normal(0, prm["sig"] * np.sqrt(prm["rho"]), len(lms))))
    ang = rng.uniform(0, 2 * np.pi, len(lms))
    mag = rng.uniform(*prm["shift"], len(lms)) * (rng.random(len(lms)) < prm["shift_frac"])
    shift = {l: (m * np.sin(g), m * np.cos(g)) for l, g, m in zip(lms, ang, mag)}
    perm = {}                                   # H4: 재생에 쓰는 모집단 안에서만 섞음 (시간 필터 통과·버린 것 따로, 탑승 중 관측 뺌)
    for pool in ([d for d in m_ids if tab[d]["walk"] and d in _S["kept"]], [d for d in m_ids if tab[d]["walk"] and d in _S["dropped"]]):
        perm.update(zip(pool, rng.permutation([tab[d]["res"] for d in pool])))
    return dict(z=z, u=u, a_ps=a_ps, b_lm=b_lm, shift=shift, perm=perm)


def lm_medians(tab, ids):
    by = {}
    for d in ids:
        e = tab[d]
        if e["match"]:
            by.setdefault(e["lm"], []).append(e["res"])
    return {l: float(np.median(v)) for l, v in by.items()}


def synth_bearing(scen, d, o, tab, R, ref):
    """조건 scen에서 관측 d의 방위 (None = 이 조건에서 안 씀)"""
    e = tab[d]
    if not e["match"]:
        if scen in ("real", "H2"):
            return o["bearing"]
        if scen in EXTRA:
            return o["bearing"] if ("res" in e) == (scen == "H2n") else None
        return float(R["u"][d]) if scen == "S4" else None
    if scen in REAL_MATCHED:
        return o["bearing"]
    if scen == "H3":
        return float(o["bearing"] - ref["med"].get(e["lm"], 0.0))
    if scen == "H5":
        return float(o["bearing"] - ref["mean"])
    if scen == "H4":
        return float(e["rel"] + R["perm"].get(d, e["res"]))       # 탑승 중 관측(PF가 안 씀)은 자기 잔차
    if scen == "S3":
        fm = run_i1._G["fmap"][e["floor"]]
        sx, sy = R["shift"][e["lm"]]
        lx, ly = fm["lm_xy"][e["row"], 0] + sx, fm["lm_xy"][e["row"], 1] + sy
        return float(wrap(np.degrees(np.arctan2(lx - e["x"], ly - e["y"])) - e["cam"]) + R["z"][d])
    b = e["rel"] + R["z"][d]
    if scen == "S1":
        b += R["a_ps"][e["pass"]]
    elif scen == "S2":
        b += R["b_lm"][e["lm"]]
    return float(wrap(b))


def scen_obs(scen, seed, freq, prm=None):
    """→ (관측 프레임 목록, 되살릴 id 집합). real·tl0.5/tl1/tl5는 기본 재생 입력(시간 필터 관측) 그대로"""
    G, tab = run_i1._G, _S["tab"]
    add = {d for d in _S["dropped"] if tab[d]["match"]} if freq == "add" else set()
    if scen == "real" and not add:
        return G["obs"]["tf"], add
    R = draws(scen, seed, tab, prm or _S["prm"])
    out = []
    for t, oo in G["obs"]["raw"]:
        v = []
        for o in oo:
            d = o["detection_id"]
            if d not in _S["kept"] and d not in add:
                continue
            b = synth_bearing(scen, d, o, tab, R, _S["ref_add" if add else "ref"])
            if b is not None:
                v.append(o if b == o["bearing"] else dict(o, bearing=b))
        out.append((t, v))
    return out, add


def schedule(obs, limit):
    """기존 입력 (시각 → 통과 id), (track → 통과 시각들): TrackLimiter를 탑승 구간 빼고 돌린 결과 (run_negative와 같음)"""
    G = run_i1._G
    sched, times = {}, {}
    lim = TrackLimiter(limit)
    for t, oo in obs:
        if any(a <= t <= b for a, b in G["rides"]):
            continue
        used = lim(t, oo)
        sched[t] = {o["detection_id"] for o in used}
        for o in used:
            times.setdefault(o["track_id"], []).append(t)
    return sched, times


def synth_structure(scens, seed=0):
    """조건마다 seed 0 합성 관측(시간 필터 통과 + 맞는 관측)의 정답 자리 잔차 구조: 합성이 의도한 구조인지 확인용"""
    tab, out = _S["tab"], {}
    ids = [d for d in _S["kept"] if tab[d]["match"] and tab[d]["walk"]]
    for s in scens:
        if s == "real":
            continue
        R = draws(s, seed, tab, _S["prm"])
        t2 = {d: dict(tab[d], res=float(wrap(synth_bearing(s, d, {"bearing": tab[d]["rel"] + tab[d]["res"]}, tab, R, _S["ref"])
                                               - tab[d]["rel"]))) for d in ids}
        st = residual_structure(t2, ids)
        out[s] = {k: st[k] for k in st if k != "per_lm"}
    return out


# ---------------------------------------------------------------- 재생
def _init(opt, n, prm, obs_params=None, cam="gyro"):
    _S["cam"] = cam
    run_i1._init("indoornav", n, None, opt)
    G = run_i1._G
    kept = {o["detection_id"] for _, oo in G["obs"]["tf"] for o in oo}
    allid = {o["detection_id"] for _, oo in G["obs"]["raw"] for o in oo}
    tab, off = match_table(G["obs"]["raw"])
    walk_kept = {d for d in kept if tab[d]["walk"]}                 # 잔차 통계는 PF가 쓰는 관측만 (탑승 중 뺌)
    walk_all = {d for d in tab if tab[d]["walk"]}
    st = residual_structure(tab, walk_kept)
    if prm.get("rho") is None:                  # 실제 잔차의 공통(판·지나감·전체 평균) 부분 비율
        tot = st["sd"] ** 2 + st["mean"] ** 2
        prm = dict(prm, rho=float((st["sd_lm"] ** 2 + st["sd_pass_in_lm"] ** 2 + st["mean"] ** 2) / tot))
    m_all = [tab[d]["res"] for d in walk_all if tab[d]["match"]]
    _S.update(tab=tab, off=off, kept=kept, dropped=allid - kept, prm=prm, obs_params=obs_params,
              ref=dict(med=lm_medians(tab, walk_kept), mean=st["mean"]),                          # H3·H5 기준: 시간 필터 통과
              ref_add=dict(med=lm_medians(tab, walk_all), mean=float(np.mean(m_all))))           # 되살림(add)일 때: 통과 + 버린 것


def run_cond(scen, freq, seed):
    from vision_filter import VisionFilter
    G = run_i1._G
    obs, add = scen_obs(scen, seed, freq)
    if freq == "add":
        base = [(t, [o for o in oo if o["detection_id"] not in add]) for t, oo in obs]
        lim = AddOnly(add, *schedule(base, 1.0), 1.0)
    else:
        lim = TrackLimiter(FREQS[freq])
    r0 = G["truth"][0]
    F = VisionFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                     float(r0["heading_deg"]), seed, _S.get("obs_params"))
    r = rp.run_one(F, obs, G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"], dpsi=G.get("dpsi"), limiter=lim)
    return r, getattr(lim, "add", 0)


def _job(c):
    t = time.perf_counter()
    r, added = run_cond(*c)
    return c, r, added, time.perf_counter() - t


def run_metrics(r):
    """seed 하나: compare_runs 지표 + 걷는 구간(t>2) 오차·불확실도 중앙·90%, 관측 수, 문제 구간 오차"""
    m = seed_metrics(r)
    t = r["t"]
    cw = ~r["esc"] & ~r["riding"] & ~paths.excluded(t) & (t > 2)
    e, u = r["err"][cw], r["unc"][cw]
    m.update(err_med=float(np.median(e)), err_p90=float(np.percentile(e, 90)), unc_med=float(np.median(u)),
             ratio=float(np.median(e / np.maximum(u, 1e-3))), n_obs=int(r["n_obs"].sum()))
    walk = ~r["esc"] & ~r["riding"] & ~paths.excluded(t)
    for a, b in SEGS.get(paths.RECORDING, []):
        s = walk & (t >= a) & (t <= b)
        m[f"seg_{a}-{b}"] = dict(err_med=float(np.median(r["err"][s])), unc_med=float(np.median(r["unc"][s])),
                                cover2=float(np.mean(r["err"][s] <= 2 * r["unc"][s])))
    return m


def med_over(ms):
    out = {}
    for k in ms[0]:
        if isinstance(ms[0][k], dict):
            out[k] = {q: float(np.median([m[k][q] for m in ms])) for q in ms[0][k]}
        else:
            out[k] = float(np.median([m[k] for m in ms]))
    return out


# ---------------------------------------------------------------- 확인
def check(a, opt, prm):
    """1) 잡음 0 S0 관측은 정답 자리에서 같은 행과 잔차 0으로 짝지어짐 2) real·tl1이 기본 재생(exp_d58_r20)과 비트까지 같음"""
    _init(opt, a.n, prm, None, a.cam)
    G, tab = run_i1._G, _S["tab"]
    obs, _ = scen_obs("S0", 0, "tl1", dict(_S["prm"], sig=0.0))
    worst, same_row, n = 0.0, 0, 0
    cam_at, _ = camera_track()
    for t, oo in obs:
        for o in oo:
            e = tab[o["detection_id"]]
            m = match_one(G["fmap"][e["floor"]], o, e["x"], e["y"], cam_at(t))
            worst = max(worst, abs(m[2]))
            same_row += m[0] == e["row"]
            n += 1
    n_unm = sum(1 for d in _S["kept"] if not tab[d]["match"])
    print(f"{paths.RECORDING} 합성 S0(잡음 0): 맞는 관측 {n}개, 같은 행 {same_row}/{n}, 정답 자리 잔차 최대 {worst:.2e}°, 버린 맞지 않는 관측 {n_unm}")
    z = np.load(paths.REC_OUT / "exp_d58_r20" / "runs.npz")
    res = run_pool(a.procs, _init, (opt, a.n, prm, None, a.cam), _job, [("real", "tl1", s) for s in range(a.seeds)])
    ok = 0
    for (_, _, s), r, _, _ in res:
        o = z[f"known_tf_기본_{s}"]
        ok += o.dtype == r.dtype and len(o) == len(r) and all(r[f].tobytes() == o[f].tobytes() for f in r.dtype.names)
    print(f"{paths.RECORDING} real·tl1 vs exp_d58_r20: 전체 필드 같음 {ok}/{len(res)} seed")
    return worst < 1e-9 and same_row == n and ok == len(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="ovc")
    ap.add_argument("--scen", nargs="+", choices=SCENS + EXTRA, default=list(SCENS))
    ap.add_argument("--freq", nargs="+", choices=list(FREQS), default=list(FREQS))
    ap.add_argument("--rho", type=float, help="S1·S2 공통 부분 분산 비율 (기본: 그 기록 실제 잔차에서)")
    ap.add_argument("--shift", type=float, nargs=2, default=(1.0, 3.0), help="S3 판 위치 옮김 크기 범위 [m]")
    ap.add_argument("--shift-frac", type=float, default=0.3, help="S3 옮기는 판 비율")
    ap.add_argument("--sig", type=float, default=SIG, help="합성 독립 잡음 σ [도] (기본: 관측 모델 σ)")
    ap.add_argument("--cam", choices=("gyro", "path"), default="gyro", help="정답 카메라 방향: gyro 보정 하나 / path 정답 진행 방향에 맞춘 천천히 바뀌는 보정")
    ap.add_argument("--eps", type=float, help="관측 모델 바닥값 (진단용, 기본: 관측 모델 0.1)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--residuals", action="store_true")
    ap.add_argument("--cross", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    opt = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
    prm = dict(rho=a.rho, shift=tuple(a.shift), shift_frac=a.shift_frac, sig=a.sig)
    if a.check:
        sys.exit(0 if check(a, opt, prm) else 1)
    if a.cross:
        res = cross(a.name)
        for k, v in res.items():
            print(k, {q: round(w, 3) if isinstance(w, float) else w for q, w in v.items()} if isinstance(v, dict) else v)
        return
    obs_params = {"eps": a.eps} if a.eps is not None else None
    _init(opt, a.n, prm, obs_params, a.cam)
    tab = _S["tab"]
    walk_kept = {d for d in _S["kept"] if tab[d]["walk"]}
    resid = dict(kept=residual_structure(tab, walk_kept), all=residual_structure(tab, {d for d in tab if tab[d]["walk"]}), cam_offset=_S["off"], rho=_S["prm"]["rho"],
                 synth=synth_structure(a.scen),
                 n_kept=len(_S["kept"]), n_kept_match=sum(tab[d]["match"] for d in _S["kept"]),
                 n_dropped=len(_S["dropped"]), n_dropped_match=sum(tab[d]["match"] for d in _S["dropped"]))
    out = paths.REC_OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.residuals:
        resid["rows"] = matched_rows(tab, walk_kept)
        (out / "residuals.json").write_text(json.dumps(resid, ensure_ascii=False, indent=1), encoding="utf-8")
        k = resid["kept"]
        print(paths.RECORDING, {x: (round(k[x], 2) if isinstance(k.get(x), float) else k.get(x)) for x in k if x != "per_lm"})
        return
    if (out / "runs.npz").exists():
        raise SystemExit(f"{out} 에 결과가 이미 있음 (덮어쓰지 않음). --name을 바꿀 것")
    conds = [(s, f, seed) for s in a.scen for f in a.freq for seed in range(a.seeds)]
    res = run_pool(a.procs, _init, (opt, a.n, prm, obs_params, a.cam), _job, conds)
    runs, ms, adds, secs = {}, {}, {}, []
    for (s, f, seed), r, added, sec in res:
        runs[f"{s}_{f}_{seed}"] = r
        ms.setdefault(f"{s}/{f}", []).append(dict(seed=seed, added=added, **run_metrics(r)))
        secs.append(sec)
    summ = {k: dict(median=med_over([{x: m[x] for x in m if x != "seed"} for m in v]), seeds=v) for k, v in ms.items()}
    info = dict(name=a.name, scen=a.scen, freq=a.freq, params=dict(_S["prm"], tol=TOL, pass_gap=PASS_GAP, fov=FOV), n=a.n, seeds=a.seeds,
                options=opt, obs_params=obs_params, cam=a.cam, sec_per_run=float(np.median(secs)),
                provenance=record.run_record(run_i1.STEPS["indoornav"], run_i1._G["P"].to_dict(), a.seeds, opt,
                                             extra=[paths.HERE / "exp_overconf.py"]))
    (out / "summary.json").write_text(json.dumps(dict(info=info, residuals=resid, summary=summ), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    np.savez_compressed(out / "runs.npz", **runs)
    print(f"→ {out}  ({len(res)}회, 1회 {np.median(secs):.1f} s)")
    for k in summ:
        m = summ[k]["median"]
        print(f"{paths.RECORDING} {k:10s} 관측 {m['n_obs']:6.0f}  err {m['err_med']:.2f}/{m['err_p90']:.2f}  unc {m['unc_med']:.2f}"
              f"  err/unc {m['ratio']:.2f}  cover2 {m['cover2']:.3f}  틀림 {m['wrong_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
