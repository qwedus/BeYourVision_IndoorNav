"""I1 오프라인 재생 (indoornav PF용): 관측 + 걸음 + 기압 → VisionFilter → 정답과 비교 (연결안 3절 7, 브랜치 원본).

localization_design cv_test/i1/replay_pf.py(시뮬 PF 재생기)와 같은 입력·같은 지표 정의를 쓰되 indoornav PF에 맞춰 바꾼 점
    - 녹화 시작 시각: 고정 REC_T0 대신 meta.json recording.first_t (paths.rec_t0)
    - 걸음: moving을 indoornav predict에 넘김. 첫 걸음 dheading도 넣음(indoornav PDR 방향 기준 = 녹화 시작, 초기 방향도 t=0 정답)
    - 탑승 중(기압 타는 구간): 걸음 위치 반영 안 함(predict 안 부름), 회전은 모아 두었다가 하차(층 바뀜) 뒤 한 번에.
      관측도 안 넣음(기준과 같음). 같은 시각 이벤트 순서: 층 전환 → 걸음 → 관측
    - 기압 이벤트(탑승 구간·층 바뀜)는 replay_pf.baro_events와 같은 규칙(아래 baro_events, 기준 summary.json 값과 같은지 확인함)
지표(metrics)는 localization_design run_i1.metrics·sweep_obs.calib와 같은 정의 + 연결안 7절 추가 지표.
"""
import csv

import numpy as np

import paths
from edge_faces import frame_obs
from text_check import apply_decisions
from time_filter import load_frames

RIDE_RATE, FLOOR_DZ = 0.12, 3.0
HOLD, T_STOP = 3.0, 114.5
REC_FIELDS = [("t", "f4"), ("err", "f4"), ("unc", "f4"), ("unc95", "f4"), ("herr", "f4"), ("esc", "?"), ("floor", "i1"),
              ("efloor", "i1"), ("n_obs", "i2"), ("ex", "f4"), ("ey", "f4"), ("riding", "?"), ("k", "f4"),
              ("k_std", "f4"), ("ess", "f4"), ("suspect", "?"), ("m1x", "f4"), ("m1y", "f4"), ("m1m", "f4"),
              ("m2x", "f4"), ("m2y", "f4"), ("m2m", "f4")]


# ---------------------------------------------------------------- 입력
def load_obs(path, t0, mode="end", decisions=None, unread="keep"):
    """cv_module 결과 → [(t, [obs dict])] (화면 끝 검출은 edge_faces 판 끝점 비교). decisions: 글자 확인 결정(text_check, 없으면 끔)."""
    return frames_obs(load_frames(path)[1], t0, mode, decisions, unread)


def frames_obs(frames, t0, mode="end", decisions=None, unread="keep"):
    if decisions is not None:
        frames = apply_decisions(frames, decisions, unread)
    return sorted((frame_obs(f, t0, mode) for f in frames), key=lambda x: x[0])


def threshold_frames(frames, th):
    """종류별 문턱값을 cv_module 프레임에 건 사본(점수가 그 종류 문턱보다 낮은 관측을 뺌). 문턱 0.25는 cv_module에서 이미 적용됨.
    시간 필터는 이 뒤에 다시 돌려야 한다(문턱에 걸린 프레임은 hit가 아님)."""
    return [dict(f, observations=[o for o in f["observations"] if o["score"] >= th.get(o["label"], 0.0)]) for f in frames]


def read_baro(path, t0):
    with open(path, encoding="utf-8") as f:
        return np.array([(float(r["t"]) - t0, float(r["rel_alt_m"])) for r in csv.DictReader(f)])


def baro_events(baro):
    """→ (타는 구간 [(시작, 끝)], 층 바뀜 [(시각, ±1)]). replay_pf.baro_events와 같은 규칙:
    2 s 사이 높이 변화율 |v| > 0.12 m/s면 타는 중, 끝날 때 누적 변화가 3 m 넘으면 층 ±1."""
    t, z = baro[:, 0], baro[:, 1]
    rate = np.full(len(t), 0.0)
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - 2.0)
        if t[i] - t[j] > 0.5:
            rate[i] = (z[i] - z[j]) / (t[i] - t[j])
    riding = np.abs(rate) > RIDE_RATE
    rides, floors = [], []
    i = 0
    while i < len(t):
        if riding[i]:
            j = i
            while j + 1 < len(t) and riding[j + 1]:
                j += 1
            a = max(0, np.searchsorted(t, t[i] - 2.0))
            e = min(j + 1, len(t) - 1)
            dz = z[e] - z[a]
            if abs(dz) >= FLOOR_DZ:
                rides.append((float(t[a]), float(t[e])))
                floors.append((float(t[e]), int(np.sign(dz)) * max(1, int(round(abs(dz) / 4.8)))))
            i = j + 1
        else:
            i += 1
    return rides, floors


def load_truth(path):
    with open(path, encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    names = list(rows[0])
    return np.array([tuple(r[k] for k in names) for r in rows], dtype=[(k, "f8") for k in names])


def truth_at(truth, t):
    i = min(max(np.searchsorted(truth["t"], t), 0), len(truth) - 1)
    return truth[i]


def drift_steps(steps, len_scale, head_per_step):
    """일부러 틀어지게 한 PDR (run_i1.drift_steps와 같음): 보폭 ×len_scale, 방향 걸음마다 head_per_step°(나침반 +) 쌓임.
    indoornav 부호 dheading_rad도 같이 바꿈."""
    out = []
    for i, s in enumerate(steps):
        s = dict(s)
        s["length_m"] *= len_scale
        if i:
            s["dheading_deg"] += head_per_step
        s["dheading_rad"] = -np.radians(s["dheading_deg"])
        s["heading_deg"] = (s["heading_deg"] + head_per_step * i) % 360
        out.append(s)
    return out


def indoornav_sign(steps):
    """임시 PDR 걸음(나침반 부호 dheading_deg만 있음)에 indoornav 부호·moving 채움"""
    return [dict(s, dheading_rad=s.get("dheading_rad", -np.radians(s["dheading_deg"])), moving=s.get("moving", True))
            for s in steps]


# ---------------------------------------------------------------- 한 번 재생
def attach_tracks(obs_frames, ids):
    """관측에 track_id 붙임 (ids = time_filter.track_ids 결과)"""
    return [(t, [dict(o, track_id=ids[o["detection_id"]]) for o in obs]) for t, obs in obs_frames]


def run_one(F, obs_frames, steps, rides, floors, truth, t0, vision=True, obs_hz=None, dpsi=None, limiter=None):
    """F: VisionFilter(초기화된 것). steps: t(호스트 시계)·length_m·dheading_rad·moving. 반환: 프레임마다 기록.
    dpsi: 시각 → 걸음 사이 회전 Δ[도] (bearing_fix, 없으면 끔). limiter: time_filter.TrackLimiter (없으면 끔)."""
    ev = [(t, 0, d) for t, d in floors] + [(s["t"] - t0, 1, s) for s in steps] + [(t, 2, o) for t, o in obs_frames]
    ev.sort(key=lambda e: (e[0], e[1]))
    riding = lambda t: any(a <= t <= b for a, b in rides)
    rec, last_used, held, n_held = [], -1e9, 0.0, 0
    for t, kind, val in ev:
        if kind == 0:
            F.change_floor(val)
            if n_held:
                F.rotate(held)
                held, n_held = 0.0, 0
        elif kind == 1:
            if riding(t):
                held += val["dheading_rad"]
                n_held += 1
                continue
            if n_held:          # 층 바뀜 없이 탑승이 끝난 경우
                F.rotate(held)
                held, n_held = 0.0, 0
            F.predict(val["length_m"], val["dheading_rad"], val["moving"])
        else:
            rd = riding(t)
            used = vision and not rd and (obs_hz is None or t - last_used >= 1.0 / obs_hz - 1e-6)
            if used:
                last_used = t
                if limiter is not None:
                    val = limiter(t, val)
                F.update(val, dpsi(t)) if dpsi is not None else F.update(val)
            e = F.estimate()
            tr = truth_at(truth, t)
            rec.append((t, float(np.hypot(e["x"] - tr["east"], e["y"] - tr["north"])), e["unc"], e["unc95"],
                        float(abs((e["compass"] - tr["heading_deg"] + 180) % 360 - 180)), bool(tr["on_esc"]),
                        int(tr["floor"]), e["floor"], len(val) if used else 0, e["x"], e["y"], rd, e["k"], e["k_std"],
                        e["ess"], e["pdr_suspect"], *e["m1"], e["m1_mass"], *e["m2"], e["m2_mass"]))
    return np.array(rec, dtype=REC_FIELDS)


def dead_reckon(steps, floors, rides, truth, t0, escalators):
    """PDR만(지도·필터 없음): 출발점·방향을 정답으로 주고 걸음만 쌓음(첫 걸음 회전 포함, 탑승 중 걸음은 회전만).
    층이 바뀌면 가장 가까운 에스컬레이터 끝으로. 좌표 ENU, 방향 나침반."""
    r0 = truth[0]
    x, y, h, fl = r0["east"], r0["north"], r0["heading_deg"], int(r0["floor"])
    ev = sorted([(t, 0, d) for t, d in floors] + [(s["t"] - t0, 1, s) for s in steps], key=lambda e: (e[0], e[1]))
    out = [(0.0, x, y, fl)]
    for t, kind, v in ev:
        if kind == 1:
            h += v["dheading_deg"]
            if not any(a <= t <= b for a, b in rides):
                x += v["length_m"] * np.sin(np.radians(h))
                y += v["length_m"] * np.cos(np.radians(h))
        else:
            src, dst = ("top", "bottom") if v < 0 else ("bottom", "top")
            k = int(np.argmin([np.hypot(x - e[src][0], y - e[src][1]) for e in escalators]))
            x, y = escalators[k][dst]
            fl += v
        out.append((t, x, y, fl))
    return np.array(out)


# ---------------------------------------------------------------- 지표 (run_i1.metrics·sweep_obs.calib와 같은 정의 + 추가)
def first_hold(t, ok, hold=HOLD):
    start = None
    for ti, o in zip(t, ok):
        if o:
            start = ti if start is None else start
            if ti - start >= hold:
                return float(start)
        else:
            start = None
    return None


def metrics(r):
    walk = ~r["esc"] & ~r["riding"] & ~((r["t"] > 78.3) & (r["t"] < 103.0))
    t = r["t"]
    tc = first_hold(t[walk], r["unc"][walk] < 2.0)
    t15 = first_hold(t[walk], r["err"][walk] < 1.5)
    after = walk & (t >= (tc if tc is not None else 1e9))
    b2 = (r["floor"] == -2) & (t >= 103.0)
    stop = t >= T_STOP
    e_all = r["err"][walk]
    cw = walk & (t > 2)                                     # sweep_obs.calib 정의
    m1 = np.c_[r["m1x"], r["m1y"]]
    m2 = np.c_[r["m2x"], r["m2y"]]
    sep = np.hypot(*(m1 - m2).T)
    two = walk & (r["m2m"] > 0.1)
    return dict(conv=tc, t15=t15,
                med=float(np.median(r["err"][after])) if after.any() else None,
                p90=float(np.percentile(r["err"][after], 90)) if after.any() else None,
                med_all=float(np.median(e_all)), p90_all=float(np.percentile(e_all, 90)), max_all=float(e_all.max()),
                b1_med=float(np.median(r["err"][walk & (t < 78.3)])),
                b2_med=float(np.median(r["err"][walk & (t >= 103.0)])),
                wrong=bool(((r["unc"] < 2.0) & (r["err"] > 3.0) & walk).any()),
                floor_ok=float((r["efloor"][b2] == -2).mean()) if b2.any() else None,
                end=float(r["err"][stop].mean()), herr_med=float(np.median(r["herr"][walk])),
                ratio=float(np.median(r["err"][cw] / np.maximum(r["unc"][cw], 1e-3))),
                cover2=float(np.mean(r["err"][cw] <= 2 * r["unc"][cw])),
                in95=float(np.mean(r["err"][cw] <= r["unc95"][cw])),
                start_med=float(np.median(r["err"][(t >= 4) & (t <= 16)])),
                two_mode=float(two.mean() / max(walk.mean(), 1e-9)),
                sep_med=float(np.median(sep[two])) if two.any() else None,
                k_end=float(r["k"][-1]), k_std_end=float(r["k_std"][-1]),
                suspect=float(r["suspect"][walk].mean()))


SUMM_KEYS = ("conv", "t15", "med", "p90", "med_all", "p90_all", "max_all", "b1_med", "b2_med", "floor_ok", "end",
             "herr_med", "ratio", "cover2", "in95", "start_med", "two_mode", "sep_med", "k_end", "k_std_end", "suspect")


def summarize(ms, diags=None):
    def med(k):
        v = [m[k] for m in ms if m.get(k) is not None]
        return float(np.median(v)) if v else None
    out = {k: med(k) for k in SUMM_KEYS}
    out["n"] = len(ms)
    out["not_conv"] = sum(m["conv"] is None for m in ms)
    out["conv_10s"] = sum(m["conv"] is not None and m["conv"] <= 10 for m in ms)
    out["conv_20s"] = sum(m["conv"] is not None and m["conv"] <= 20 for m in ms)
    out["wrong"] = sum(m["wrong"] for m in ms)
    if diags:
        for k in diags[0]:
            out["diag_" + k] = float(np.median([d[k] for d in diags]))
    return out


def dr_metrics(dr, truth):
    t = truth["t"]
    ex = np.interp(t, dr[:, 0], dr[:, 1])
    ey = np.interp(t, dr[:, 0], dr[:, 2])
    e = np.hypot(ex - truth["east"], ey - truth["north"])
    walk = (truth["on_esc"] == 0) & ~((t > 78.3) & (t < 103.0))
    return dict(med_all=float(np.median(e[walk])), p90_all=float(np.percentile(e[walk], 90)),
                max_all=float(e[walk].max()), b1_med=float(np.median(e[walk & (t < 78.3)])),
                b2_med=float(np.median(e[walk & (t >= 103)])), end=float(e[t >= T_STOP].mean()),
                err_at_78=float(e[np.searchsorted(t, 78.0)])), e
