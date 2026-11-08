"""실시간 조건 걸음·기압: 녹화를 시각순으로 흘리며 그 시각까지 들어온 값만으로 걸음·탑승·층 바뀜을 정한다.

indoornav PDR(pdr_tuned_v4) 코드는 고치지 않고 함수를 불러 감싸서 쓴다. 바꾸는 것은 설정 하나와 '언제 무엇을 넘기나'뿐이다.
    저역 필터   preprocess.mode = "delayed"(indoornav delayed_filtfilt, 0.5 s 지연·3 s 창·0.1 s 갱신)로 바꾼 사본 설정
                → 샘플 i는 '지금' 샘플 수 now가 0.1 s 격자에서 now - 50 > i가 되는 순간 확정(delayed_filtfilt 출력 규칙)
    걸음 찾기   0.1 s마다 확정된 신호의 최근 window_s만 detect_steps(find_peaks 그대로)에 넣고,
                봉우리 뒤 lookahead_s가 확정된 봉우리만 그 갱신에서 한 번 판정(채택·버림, 다시 안 봄)
    걸음 구간   step_windows를 '지금까지 걸음'에 불러 마지막 것 사용(첫 걸음·재시작 길이 = 지금까지 간격 중앙값)
    보폭·방향   step_lengths·step_headings를 그 걸음 하나에 부름(구간이 봉우리까지라 앞쪽 값 안 씀). 방향 적분은 원래 앞쪽 값 안 씀
    이동 묶음   bout_mask와 같은 규칙(간격 3 s 이하로 이어진 걸음이 4개 이상이면 이동)을 들어온 걸음으로 판정:
                4번째 걸음이 확정되면 그 묶음 이동, 마지막 걸음 뒤 3 s까지 판정이 끝났는데 4개 미만이면 제자리.
                bout="wait": 판정이 날 때까지 걸음을 PF에 안 넘김(t_known = 판정 시각), "early": 확정 즉시 이동으로 넘김
기압 (baro.csv)
    앱이 받은 시각 t_recv 순으로 들어온다고 본다. 높이 변화율·탑승 문턱·층 계산은 replay.baro_events와 같은 식(2 s 변화율 > 0.12 m/s,
    기준 높이 = 첫 탑승 표본 2 s 앞 표본). 다른 점: 탑승 구간은 첫 탑승 표본 시각부터(2 s 앞당김 없음),
    '의심'(첫 탑승 표본 수신) → '확정'(누적 3 m 넘은 표본 수신) 또는 '취소'(3 m 못 넘고 끝난 표본 수신), 층 바뀜은 끝 표본 수신 때 앎.
시각: 걸음은 호스트 시계(steps_indoornav.csv와 같은 약속), 기압 사건은 재생 시각(호스트 - 첫 영상 프레임).
"""
import csv

import numpy as np

from indoornav.pdr import PDRParams
from indoornav.pdr.heading import heading_series, step_headings
from indoornav.pdr.pipeline import walk_signal
from indoornav.pdr.step_detect import detect_steps, step_windows
from indoornav.pdr.step_length import bout_mask, step_frequency, step_lengths

import replay as rp

LOOKAHEAD_S, WINDOW_S = 0.75, 4.0     # 봉우리 뒤 0.75 s: 9개 기록 기준 걸음과 99.2% 일치(0.5 s 98.1%, 1.0 s 99.6%)
STEP_FIELDS = ("t", "length_m", "heading_deg", "dheading_deg", "moving", "dheading_rad", "t_confirm", "t_known", "idx",
               "length_raw")


def delayed_params(params):
    """PDR 설정(pdr_tuned_v4)의 사본: 저역 필터만 지연 방식(delay_s·window_s·hop_s는 설정 파일 값 그대로)"""
    return params.with_(**{"preprocess.mode": "delayed"})


# ---------------------------------------------------------------- 걸음
def check_realtime_params(p):
    """이 감싸기가 앞쪽 값 없이 도는 설정인지. indoornav의 다른 선택지는 녹화 전체를 씀:
    step.signal="norm"(pipeline.walk_signal 전체 평균 빼기), heading.zaru(가운데 창 정지 판정·정지 구간 평균),
    heading.source="orientation"(yaw - yaw[0]은 괜찮지만 확인 안 함), step.method="adaptive"(창 판단, 감싸지 않음)"""
    bad = []
    if p.step.signal != "vertical":
        bad.append(f"step.signal={p.step.signal}")
    if p.step.method != "peak":
        bad.append(f"step.method={p.step.method}")
    if p.heading.source != "gyro":
        bad.append(f"heading.source={p.heading.source}")
    if p.heading.zaru:
        bad.append("heading.zaru=true")
    if bad:
        raise ValueError("실시간 감싸기가 확인하지 않은 설정(앞쪽 값을 쓸 수 있음): " + ", ".join(bad))


def online_steps(session, params, lookahead_s=LOOKAHEAD_S, window_s=WINDOW_S, sig=None, avail=None):
    """→ (걸음 목록, 갱신 기록). 걸음 dict: idx, t(봉우리, Session 시각), t_confirm(봉우리 판정), t_known(이동 판정까지 끝난 시각,
    끝날 때까지 못 정하면 None), length(이동 판정 전 보폭), heading·dheading(indoornav 부호 rad), moving(None = 못 정함), win.
    갱신 기록: tick_t(갱신 시각 = 마지막으로 받은 샘플 시각), cut_t(그 갱신까지 걸음 판정이 끝난 마지막 샘플 시각).
    sig: 시험용 신호 바꿔 넣기(없으면 delayed 필터 walk_signal).
    avail: 격자 샘플마다 '받은' 시각(Session 시각). 100 Hz 격자 값은 바로 뒤 원본 표본으로 보간되므로 그 원본이 온 시각
    (motion_avail). 없으면 격자 시각 그대로."""
    p = delayed_params(params) if sig is None else params
    fs, sp, lp, pp = p.fs, p.step, p.length, p.preprocess
    check_realtime_params(p)
    if sig is None:
        sig = walk_signal(session, p)
    h = heading_series(session, p.heading, fs)
    n = len(sig)
    d, hop = int(round(pp.delay_s * fs)), max(1, int(round(pp.hop_s * fs)))
    look, win_n = int(round(lookahead_s * fs)), int(round(window_s * fs))
    dist = max(1, int(round(sp.min_interval_s * fs)))
    gap_n = int(round(lp.bout_gap_s * fs))
    t = session.t
    steps, idx = [], []
    bout = []               # 지금 열린 묶음의 걸음 번호
    ticks_t, ticks_cut = [], []
    prev_cut = -1

    def decide(members, moving, when):
        for k in members:
            s = steps[k]
            if s["moving"] is None:
                s["moving"] = moving
                s["t_known"] = max(when, s["t_confirm"]) if s["t_known"] is None else s["t_known"]

    for now in range(hop, n + 1, hop):          # now = 지금까지 받은 샘플 수 (delayed_filtfilt와 같은 격자)
        e = now - d                             # 확정된 신호 sig[0:e]
        cut = e - 1 - look                      # 이번 갱신까지 판정 끝나는 봉우리 위치
        when = float(t[now - 1] if avail is None else avail[now - 1])
        if cut <= prev_cut:
            continue
        a = max(0, e - win_n)
        pk = detect_steps(sig[a:e], fs, sp) + a
        for q in pk[(pk > prev_cut) & (pk <= cut)]:
            q = int(q)
            if idx and q - idx[-1] < dist:      # 앞 갱신에서 채택한 봉우리와 너무 가까움(창이 달라 생길 수 있음)
                continue
            if bout and q - idx[-1] > gap_n:    # 묶음 끊김: 이전 묶음은 이미 아래에서 닫혔어야 함
                decide(bout, len(bout) >= lp.min_bout_steps, when)
                bout = []
            idx.append(q)
            arr = np.array(idx)
            w = step_windows(arr, fs, sp, e)[-1]
            freq = step_frequency(arr, fs, sp.max_interval_s)[-1:]
            L = float(step_lengths(sig, [w], lp, freq, np.zeros(1, bool))[0])
            hs = float(step_headings(h, np.array([q]), [w], p.heading)[0])
            dh = hs - (steps[-1]["heading"] if steps else 0.0)
            steps.append(dict(idx=q, t=float(t[q]), t_confirm=when, t_known=None, length=L, heading=hs, dheading=dh,
                              moving=None, win=w))
            bout.append(len(steps) - 1)
            if lp.min_bout_steps <= 1:
                decide(bout, True, when)
            elif len(bout) >= lp.min_bout_steps:
                decide(bout, True, when)        # 4번째 걸음: 묶음 전체 이동 (앞 3걸음 판정 시각 = 지금)
        prev_cut = cut
        if bout and cut - idx[-1] >= gap_n:     # 마지막 걸음 뒤 3 s까지 판정 끝남 → 묶음 닫힘
            decide(bout, len(bout) >= lp.min_bout_steps, when)
            bout = []
        ticks_t.append(when)
        ticks_cut.append(float(t[max(cut, 0)]))
    return steps, dict(tick_t=np.array(ticks_t), cut_t=np.array(ticks_cut), sig=sig, heading=h)


def motion_avail(rec, session, t0_sensor):
    """motion.csv 원본 시각 → 격자 샘플마다 받은 시각(Session 시각) = 격자 시각보다 뒤의 첫 원본 표본 시각.
    app_session.load_app_session은 원본을 np.interp로 격자에 옮기므로 격자 값 i는 그 원본이 와야 정해짐(약 10 ms 안)"""
    import pandas as pd
    raw = pd.read_csv(rec / "motion.csv", usecols=["t"])["t"].to_numpy(float) - t0_sensor
    if len(raw) > 1 and not np.all(np.diff(raw) > 0):
        raise ValueError("motion.csv 시각이 늘어나는 순서가 아님")
    k = np.minimum(np.searchsorted(raw, session.t, side="right"), len(raw) - 1)
    return np.maximum(raw[k], session.t)


def check_bout(steps, params):
    """판정 끝난 걸음의 이동 여부가 indoornav bout_mask(같은 걸음 배열)와 같은지 → 다른 걸음 수"""
    done = [s for s in steps if s["moving"] is not None]
    if not done:
        return 0
    ref = bout_mask(np.array([s["idx"] for s in steps]), params.fs, params.length.min_bout_steps, params.length.bout_gap_s)
    return int(sum(s["moving"] != bool(ref[k]) for k, s in enumerate(steps) if s["moving"] is not None))


def step_rows(steps, t0_sensor, bout="wait"):
    """online_steps → steps_indoornav.csv 형식 행(호스트 시계) + t_confirm·t_known(호스트).
    bout="wait": 보폭은 이동 판정대로(제자리 0). 입력 끝까지 이동 판정을 못 한 걸음은 t_known = nan(모름)으로 남김
    → 재생기가 '판정 기다리는 걸음'으로 보고 PF에는 안 넣음. "early": 확정된 걸음 전부 이동으로, t_known = t_confirm."""
    rows = []
    for s in steps:
        if bout == "early":
            moving, known = True, s["t_confirm"]
        elif s["moving"] is None:
            moving, known = False, float("nan") - t0_sensor
        else:
            moving, known = s["moving"], s["t_known"]
        rows.append(dict(t=s["t"] + t0_sensor, length_m=s["length"] if moving else 0.0, heading_deg=-np.degrees(s["heading"]),
                         dheading_deg=-np.degrees(s["dheading"]), moving=bool(moving), dheading_rad=s["dheading"],
                         t_confirm=s["t_confirm"] + t0_sensor, t_known=known + t0_sensor, idx=s["idx"], length_raw=s["length"]))
    return rows


def offline_rows(session, params, t0_sensor, mode="offline"):
    """indoornav run_pdr를 녹화 전체에 (mode 바꾼 설정으로) → 같은 형식 행. t_confirm = t_known = 봉우리 시각(지연 없음으로 둠)"""
    from indoornav.pdr import run_pdr
    p = params if mode == "offline" else params.with_(**{"preprocess.mode": mode})
    st = run_pdr(session, p).steps
    return [dict(t=float(r.t) + t0_sensor, length_m=float(r.length), heading_deg=-np.degrees(r.heading),
                 dheading_deg=-np.degrees(r.dheading), moving=bool(r.moving), dheading_rad=float(r.dheading),
                 t_confirm=float(r.t) + t0_sensor, t_known=float(r.t) + t0_sensor, idx=int(r.idx), length_raw=float(r.length))
            for r in st.itertuples()]


def write_rows(rows, path):
    """pdr_steps.write_steps와 같은 자릿수(재생 입력이 같은 반올림을 거치게) + 판정 시각 열.
    판정 시각(t_confirm·t_known)은 갱신 시각과 정확히 같아야 해서 반올림 없이 씀(소수 6자리로 쓰면 갱신 시각보다 1 µs 안쪽으로
    늦어져, 경계가 봉우리를 지났는데 걸음이 아직 안 보이는 순간이 생김)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(r[k])) if k in ("t_confirm", "t_known") else f"{r[k]:.6f}" if k in ("t", "dheading_rad")
                            else int(r[k]) if k in ("moving", "idx") else f"{r[k]:.4f}") for k in STEP_FIELDS})


def read_rows(path):
    """write_rows 파일 → 재생 입력(pdr_steps.read_steps와 같은 변환 + indoornav 부호)"""
    from pdr_steps import read_steps
    return rp.indoornav_sign(read_steps(path))


# ---------------------------------------------------------------- 기압
def read_baro_rt(path, t0):
    """→ 배열 (t, rel_alt_m, t_recv), 재생 시각. 수신 순서 = 샘플 순서인지 확인"""
    with open(path, encoding="utf-8") as f:
        a = np.array([(float(r["t"]) - t0, float(r["rel_alt_m"]), float(r["t_recv"]) - t0) for r in csv.DictReader(f)])
    if len(a) > 1 and not (np.all(np.diff(a[:, 0]) > 0) and np.all(np.diff(a[:, 2]) >= 0)):
        raise ValueError("기압 표본 시각·수신 시각이 순서대로가 아님")
    return a


def read_baro_abs_rt(path, t0):
    """baro_abs.csv(절대 고도, 약 3 Hz) → (표본 시각 추정, alt_m, t_recv), 재생 시각. 표본 시계(t)가 호스트 시계와 달라서
    표본 시각 = t + (그때까지 받은 표본의 t_recv - t 최솟값)으로 둠(그때까지 가장 빨리 온 표본의 지연을 0으로 가정 → 실제 지연은
    이보다 큼). 뒤 표본으로 앞 표본 시각을 고치지 않음. 시각이 거꾸로 가면 바로 앞 표본 + 1 µs"""
    with open(path, encoding="utf-8") as f:
        a = np.array([(float(r["t"]), float(r["alt_m"]), float(r["t_recv"]) - t0) for r in csv.DictReader(f)])
    a[:, 0] += np.minimum.accumulate(a[:, 2] - a[:, 0])
    for i in range(1, len(a)):
        a[i, 0] = max(a[i, 0], a[i - 1, 0] + 1e-6)
    if len(a) > 1 and not (np.all(np.diff(a[:, 0]) > 0) and np.all(np.diff(a[:, 2]) >= 0)):
        raise ValueError("절대 고도 표본 시각·수신 시각이 순서대로가 아님")
    return a


def online_baro(baro):
    """→ 후보 목록. 표본 i는 t_recv[i]에 들어오고, 그때까지 들어온 표본만 씀(변화율 식은 replay.baro_events와 같음).
    후보 dict: t_first(첫 탑승 표본 시각), k_suspect(그 표본 수신), k_confirm(누적 3 m 넘은 표본 수신, 없으면 None),
    t_end(끝 표본 = 처음 탑승 아닌 표본 시각), k_end(그 수신, 녹화 끝까지 안 끝나면 None), ride(끝에서 3 m 넘음), delta(층 ±)"""
    t, z, tr = baro[:, 0], baro[:, 1], baro[:, 2]
    rate = np.zeros(len(t))
    for i in range(len(t)):                    # 표본 i의 변화율은 i까지 표본만 씀 (j ≤ i)
        j = np.searchsorted(t, t[i] - 2.0)
        if t[i] - t[j] > 0.5:
            rate[i] = (z[i] - z[j]) / (t[i] - t[j])
    riding = np.abs(rate) > rp.RIDE_RATE
    out, cur = [], None
    for i in range(len(t)):
        if cur is None:
            if riding[i]:
                a = max(0, int(np.searchsorted(t, t[i] - 2.0)))
                cur = dict(t_first=float(t[i]), k_suspect=float(tr[i]), k_confirm=None, t_end=None, k_end=None, ride=False,
                           delta=0, base=a, i_first=i)
                if abs(z[i] - z[a]) >= rp.FLOOR_DZ:
                    cur["k_confirm"] = float(tr[i])
        elif riding[i]:
            if cur["k_confirm"] is None and abs(z[i] - z[cur["base"]]) >= rp.FLOOR_DZ:
                cur["k_confirm"] = float(tr[i])
        else:
            dz = z[i] - z[cur["base"]]
            cur.update(t_end=float(t[i]), k_end=float(tr[i]), i_end=i, dz=float(dz))
            if abs(dz) >= rp.FLOOR_DZ:
                cur.update(ride=True, delta=int(np.sign(dz)) * max(1, int(round(abs(dz) / 4.8))))
                if cur["k_confirm"] is None:     # 끝 표본에서 처음 3 m를 넘음 → 확정 = 끝 수신
                    cur["k_confirm"] = float(tr[i])
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)                        # 녹화 끝까지 안 끝난 후보 (층 바뀜 없음)
    return out, dict(t=t, t_recv=tr, riding=riding)
