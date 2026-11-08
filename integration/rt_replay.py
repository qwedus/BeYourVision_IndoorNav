"""실시간 조건 재생: 걸음·기압·관측을 '알게 된 시각' 순으로 받아 PF에 시각순으로 넣는다 (replay.run_one의 실시간판, 새 파일).

PF에 넣는 일(층 전환 → 걸음 → 관측 순서, 탑승 중 걸음은 회전만 모음, 관측은 탑승 중 안 넣음, 시간 필터·track 1 s 제한·
걸음 사이 회전 보정·화면 끝 처리)은 run_one과 같다. 다른 것은 '언제 넣나'뿐:
    벽시계 w마다 그때까지 알게 된 사건만 줄에 넣고, 시각 C(w)까지의 사건을 시각순으로 꺼내 PF에 넣는다.
    C(w) = min(걸음 경계, 기압 경계), 단 w - max_wait보다 늦지는 않음(오래 기다린 사건은 그때 아는 것으로 처리)
        걸음 경계: 걸음 판정이 끝난 시각, 이동 판정을 기다리는 걸음이 있으면 그 걸음 앞
        기압 경계: 탑승 의심 중이면 첫 탑승 표본 앞(hold) / 확정된 탑승 중이면 마지막으로 받은 표본까지 / 아니면 없음
    늦게 안 사건(이미 PF가 지나간 시각)은 알게 된 즉시 넣는다(순서 어긋남, 개수 기록).
    탑승 여부는 그 사건을 넣는 순간 아는 것으로 정함: 확정·끝난 탑승 구간 안, 또는 아직 안 끝난 후보(의심·확정)의 시작 뒤면 탑승.
기록: run_one 필드 + w(위치가 나온 벽시계), ec(추정 방향, 나침반), riding_rt(이번에 쓴 탑승 판정), t64(촬영 시각 f8, t 필드는 f4라
    벽시계와 비교할 때 씀), href(그 출력의 PF 방향이 기준으로 삼는 PDR 방향 = 마지막으로 PF에 들어간 걸음 방향 − 아직 안 넣은 모은 회전),
    ts(그 출력의 PF 상태가 나타내는 시각 = 지금까지 넣은 사건 중 가장 늦은 시각. 관측이 늦게 오면 촬영 시각보다 뒤일 수 있음).
    riding 필드는 채점 마스크용으로 기준 재생의 탑승 구간(replay.baro_events)을 그대로 적음 → 두 방식이 같은 프레임을 채점.
모든 시각은 재생 시각(호스트 - 첫 영상 프레임).
"""
import heapq

import numpy as np

import replay as rp

RT_FIELDS = rp.REC_FIELDS + [("w", "f8"), ("ec", "f4"), ("riding_rt", "?"), ("t64", "f8"), ("href", "f8"), ("ts", "f8")]
INF = float("inf")


# ---------------------------------------------------------------- 아는 것(지식) 모형
class OracleBaro:
    """기준 재생과 같은 기압: 탑승 구간·층 바뀜을 처음부터 앎(2 s 앞당김 포함)"""

    def __init__(self, rides, floors):
        self.rides, self.floors = rides, [(t, d, -INF) for t, d in floors]

    def riding(self, s, w):
        return any(a <= s <= b for a, b in self.rides)

    def bound(self, w):
        return (INF, 9)

    def instants(self):
        return []


class OnlineBaro:
    """rt_pdr.online_baro 후보로 그 순간 아는 탑승 상태. suspect="hold": 의심 중에는 PF를 첫 탑승 표본 앞에서 멈춤,
    "ignore": 의심은 무시하고 확정부터 탑승으로 봄(확정 전 사건은 탑승 아님으로 이미 들어감)"""

    def __init__(self, cands, samples, suspect="hold"):
        self.c, self.suspect = cands, suspect
        self.recv, self.ts = samples["t_recv"], samples["t"]
        self.floors = [(c["t_end"], c["delta"], c["k_end"]) for c in cands if c["ride"] and c["k_end"] is not None]

    def _state(self, c, w):
        if c["k_suspect"] > w:
            return None
        if c["k_end"] is not None and c["k_end"] <= w:
            return "ride" if c["ride"] else "cancel"
        if c["k_confirm"] is not None and c["k_confirm"] <= w:
            return "confirmed"
        return "suspect"

    def riding(self, s, w):
        for c in self.c:
            st = self._state(c, w)
            if st == "ride" and c["t_first"] <= s <= c["t_end"]:
                return True
            if st == "confirmed" and s >= c["t_first"]:
                return True
            if st == "suspect" and self.suspect == "hold" and s >= c["t_first"]:
                return True
        return False

    def bound(self, w):
        b = (INF, 9)
        for c in self.c:
            st = self._state(c, w)
            if st == "suspect" and self.suspect == "hold":
                b = min(b, (c["t_first"], 0))
            elif st == "confirmed":          # 받은 마지막 표본까지는 탑승 중으로 앎
                k = int(np.searchsorted(self.recv, w, side="right")) - 1
                b = min(b, (float(self.ts[k]), 9))
        return b

    def instants(self):
        return list(self.recv)


class OracleSteps:
    """걸음을 봉우리 순간에 앎(기준 재생과 같음)"""

    def __init__(self, steps, t0):
        self.steps = steps
        self.t = [s["t"] - t0 for s in steps]
        self.known = list(self.t)

    def bound(self, w):
        return (w, 9)

    def instants(self):
        return []


class OnlineSteps:
    """걸음 판정 시각·이동 판정 시각으로 앎. ticks: 갱신 시각·그때 판정 끝난 시각(재생 시각)"""

    def __init__(self, steps, t0, tick_t, cut_t):
        self.steps = steps
        self.t = [s["t"] - t0 for s in steps]
        self.known = [s["t_known"] - t0 for s in steps]
        self.confirm = np.array([s["t_confirm"] - t0 for s in steps])
        self.tick_t = np.asarray(tick_t)
        # 판정 끝난 시각을 걸음 봉우리 시각과 같은 정밀도(호스트 시계 소수 6자리, 걸음 파일과 같은 반올림)로 맞춤.
        # 안 맞추면 같은 격자 봉우리가 경계보다 수 ps 커져 한 갱신(0.1 s) 늦게 넣거나, 같은 시각 관측보다 늦게 들어갈 수 있음
        self.cut_t = np.array([float(f"{c + t0:.6f}") for c in np.asarray(cut_t)]) - t0
        self.end_cut = float(self.cut_t[-1]) if len(self.cut_t) else -INF

    def bound(self, w):
        k = int(np.searchsorted(self.tick_t, w, side="right")) - 1
        b = (float(self.cut_t[k]) if k >= 0 else -INF, 9)
        for i in np.flatnonzero(self.confirm <= w):      # 판정됐지만 이동 판정 기다리는 걸음 (nan = 입력 끝까지 모름)
            if not self.known[i] <= w:
                b = min(b, (self.t[i], 1))
                break
        return b

    def instants(self):
        return list(self.tick_t)


# ---------------------------------------------------------------- 재생
def heading_lookup(heading, t0):
    """(t_host, h[, avail_host]) → f(t 재생 시각, w 벽시계) = 시각 t의 indoornav 방향. avail이 있으면 w까지 받은 표본만 보간
    (그 뒤는 마지막 값 유지). 없으면 전체 배열(기준 재생 bearing_fix.rotation_since_step과 같은 np.interp)"""
    if heading is None:
        return None
    ht, hh = np.asarray(heading[0]), np.asarray(heading[1])
    if len(heading) < 3 or heading[2] is None:
        return lambda t, w: np.interp(t + t0, ht, hh)
    av = np.asarray(heading[2])

    def f(t, w):
        k = int(np.searchsorted(av, w + t0, side="right"))
        return np.interp(t + t0, ht[:k], hh[:k]) if k else 0.0
    return f


def run_rt(F, obs_frames, steps_k, baro_k, truth, t0, score_rides, vision=True, obs_hz=None, limiter=None, heading=None,
           max_wait=5.0, grid=0.05, obs_lag=0.0, log=None):
    """F: VisionFilter. steps_k: OracleSteps|OnlineSteps (걸음 dict: t(호스트)·length_m·dheading_rad·moving·heading_deg).
    baro_k: OracleBaro|OnlineBaro. score_rides: 채점 마스크용 기준 탑승 구간. heading: (t_host, heading[, avail_host])
    걸음 사이 회전 보정(없으면 끔). avail_host가 있으면 방향 표본은 그 시각에 받은 것으로 보고 벽시계까지 받은 표본만 보간
    (없으면 전체 배열 = 기준 재생과 같음). obs_lag: 관측이 촬영 뒤 몇 초에 오나. log: dict를 주면 늦은 걸음·층 전환 등 기록.
    관측 회전 보정 Δ = 관측 시각 방향 − PF 상태의 PDR 방향(마지막으로 꺼낸 걸음 방향, 모은 회전은 관측 전에 PF에 넣음)."""
    steps = steps_k.steps
    ev = []                              # (알게 된 시각, 키, 값)
    for k, (t, d, kn) in enumerate(baro_k.floors):
        ev.append((max(kn, t), (t, 0, k), ("floor", d)))
    for k, s in enumerate(steps):
        if not np.isfinite(steps_k.known[k]):     # 입력 끝까지 이동 판정 못 함 → PF에 안 넣음(경계에만 씀)
            continue
        ev.append((max(steps_k.known[k], steps_k.t[k]), (steps_k.t[k], 1, k), ("step", s)))
    for k, (t, o) in enumerate(obs_frames):
        ev.append((t + obs_lag, (t, 2, k), ("obs", o)))
    ev.sort(key=lambda e: (e[0], e[1]))
    info_t = np.unique(np.r_[steps_k.instants(), baro_k.instants()])   # 걸음 갱신·기압 수신 시각(정보가 바뀌는 때)
    in_rides = lambda t: any(a <= t <= b for a, b in score_rides)
    q, rec = [], []
    last_key, last_used, held, n_held = (-INF, -1, -1), -1e9, 0.0, 0
    last_h = 0.0                         # 마지막으로 꺼낸 걸음의 indoornav 방향(rad, 녹화 시작 0). 모은 회전 held는 아직 PF에 없음
    hd_at = heading_lookup(heading, t0)
    lg = log if log is not None else {}
    lg.update(late_steps=0, late_other=0, forced=0, floors=[], held_lost=0)
    j, w = 0, -INF
    while j < len(ev) or q:
        # 다음 벽시계: 격자·다음 사건 도착·정보 바뀜·줄에 있는 사건의 기다림 한도 중 가장 이른 때 (모두 그때까지 아는 값)
        nxt = [np.floor(w / grid + 1e-9) * grid + grid if w > -INF else 0.0]
        if j < len(ev):
            nxt.append(ev[j][0])
        k = int(np.searchsorted(info_t, w, side="right"))
        if k < len(info_t):
            nxt.append(float(info_t[k]))
        if q:
            nxt.append(q[0][0][0] + max_wait)
        w = min(nxt)
        while j < len(ev) and ev[j][0] <= w:
            heapq.heappush(q, (ev[j][1], j, ev[j][2]))
            j += 1
        C = min(steps_k.bound(w), baro_k.bound(w))
        while q and (q[0][0] <= C or q[0][0][0] + max_wait <= w or q[0][0] < last_key):
            key, _, (kind, val) = heapq.heappop(q)
            t = key[0]
            if key < last_key:
                lg["late_steps" if kind == "step" else "late_other"] += 1
            elif key > C:
                lg["forced"] += 1
            last_key = max(last_key, key)
            if kind == "floor":
                F.change_floor(val)
                lg["floors"].append(dict(w=float(w), t=float(t), delta=int(val)))
                if n_held:
                    F.rotate(held)
                    held, n_held = 0.0, 0
                continue
            rd = baro_k.riding(t, w)
            if kind == "step":
                last_h = -np.radians(val["heading_deg"])
                if rd:
                    held += val["dheading_rad"]
                    n_held += 1
                    if val["moving"] and val["length_m"] > 0:
                        lg["held_lost"] += 1
                    continue
                if n_held:
                    F.rotate(held)
                    held, n_held = 0.0, 0
                F.predict(val["length_m"], val["dheading_rad"], val["moving"])
                continue
            used = vision and not rd and (obs_hz is None or t - last_used >= 1.0 / obs_hz - 1e-6)
            if used:
                if n_held:                       # 탑승으로 모은 회전(층 바뀜 없이 끝남)을 관측 비교 전에 PF에 넣음
                    F.rotate(held)
                    held, n_held = 0.0, 0
                last_used = t
                if limiter is not None:
                    val = limiter(t, val)
                if hd_at is not None:
                    F.update(val, float(np.degrees(hd_at(t, w) - last_h)))
                else:
                    F.update(val)
            e = F.estimate()
            tr = rp.truth_at(truth, t)
            rec.append((t, float(np.hypot(e["x"] - tr["east"], e["y"] - tr["north"])), e["unc"], e["unc95"],
                        float(abs((e["compass"] - tr["heading_deg"] + 180) % 360 - 180)), bool(tr["on_esc"]),
                        int(tr["floor"]), e["floor"], len(val) if used else 0, e["x"], e["y"], in_rides(t), e["k"],
                        e["k_std"], e["ess"], e["pdr_suspect"], *e["m1"], e["m1_mass"], *e["m2"], e["m2_mass"],
                        float(w), e["compass"], rd, float(t), float(last_h - held), float(last_key[0])))
    lg["unprocessed"] = len(q) + len(ev) - j
    return np.array(rec, dtype=RT_FIELDS)


# ---------------------------------------------------------------- 실시간 지표
DR_SPAN, DR_DT = 2.0, 0.1


def now_errors(r, truth, steps=None, t0=0.0, heading=None):
    """벽시계 기준 오차: 촬영 시각 t_k(= 벽시계)에 화면에 있는 위치 = w ≤ t_k인 마지막 출력. → (보여 준 위치 오차, 앞당긴 위치 오차)
    앞당김(화면용 후보, PF에는 안 넣음): 마지막 출력 위치에서 (t_k - 그 출력의 PF 상태 시각 ts)만큼, 속도 = 그 출력 시각 앞 2 s 안 걸음 보폭 합
    ÷ 2 s(t_k까지 판정된 것만, 이동 판정 전 걸음은 이동으로 봄), 방향 = 출력 PF 방향 + (그 시각 자이로 방향 − 출력의 PDR 기준 href).
    방향 표본은 t_k까지 받은 것만 씀(heading[2] = 받은 시각, 없으면 표본 시각으로 봄). 시각은 f8(t64)로 비교."""
    t = r["t64"] if "t64" in r.dtype.names else r["t"].astype("f8")
    ts = r["ts"] if "ts" in r.dtype.names else t          # 출력의 PF 상태 시각(앞당김 시작점)
    w = r["w"]
    k = np.searchsorted(w, t, side="right") - 1
    ok = k >= 0
    tr_e = np.interp(t, truth["t"], truth["east"])
    tr_n = np.interp(t, truth["t"], truth["north"])
    kk = np.maximum(k, 0)
    e_now = np.where(ok, np.hypot(r["ex"][kk] - tr_e, r["ey"][kk] - tr_n), np.nan)
    if steps is None or heading is None:
        return e_now, None
    st = np.array([s["t"] - t0 for s in steps])
    conf = np.array([s.get("t_confirm", s["t"]) - t0 for s in steps])
    known = np.array([s.get("t_known", s["t"]) - t0 for s in steps])
    ln = np.array([s["length_m"] for s in steps])                       # 이동 판정 뒤 보폭(제자리 0)
    raw = np.array([s.get("length_raw", s["length_m"]) for s in steps])  # 판정 전에는 이동으로 봄
    ht, hh = np.asarray(heading[0]) - t0, np.asarray(heading[1])
    av = np.asarray(heading[2]) - t0 if len(heading) > 2 and heading[2] is not None else ht
    e_dr = np.full(len(t), np.nan)
    for i in np.flatnonzero(ok):
        j = kk[i]
        x, y, t1 = r["ex"][j], r["ey"][j], ts[j]
        if t[i] - t1 > 1e-6:
            m = (st > t1 - DR_SPAN) & (st <= t1) & (conf <= t[i])
            v = float(np.where(known[m] <= t[i], ln[m], raw[m]).sum()) / DR_SPAN
            n = int(np.searchsorted(av, t[i], side="right"))        # t_k까지 받은 방향 표본
            tau = np.arange(t1, t[i], DR_DT) + DR_DT / 2
            h_tau = np.interp(np.minimum(tau, t[i]), ht[:n], hh[:n]) if n else np.zeros(len(tau))
            c = r["ec"][j] - np.degrees(h_tau - r["href"][j])
            dt = np.minimum(DR_DT, t[i] - (tau - DR_DT / 2))
            x = x + v * float(np.sum(dt * np.sin(np.radians(c))))
            y = y + v * float(np.sum(dt * np.cos(np.radians(c))))
        e_dr[i] = np.hypot(x - tr_e[i], y - tr_n[i])
    return e_now, e_dr


def rt_metrics(r, truth, steps=None, t0=0.0, log=None, heading=None, end_t=INF):
    """run_one 지표(replay.metrics) + 출력 지연·벽시계 오차·층 전환 시각.
    end_t: 녹화가 끝나 걸음 판정이 멈춘 시각. 그 뒤 프레임은 지연·벽시계 지표에서 뺌(녹화 끝 때문에 기다림 한도까지 밀림)"""
    import paths
    m = rp.metrics(r)
    t = r["t"]
    walk = ~r["esc"] & ~r["riding"] & ~paths.excluded(t)
    wv = walk & (t <= end_t)
    t8 = r["t64"] if "t64" in r.dtype.names else t.astype("f8")
    dl = (r["w"] - t8)[wv]
    m.update(delay_med=float(np.median(dl)), delay_p90=float(np.percentile(dl, 90)), delay_max=float(dl.max()))
    e_now, e_dr = now_errors(r, truth, steps, t0, heading)
    ok = wv & np.isfinite(e_now)
    m.update(now_med=float(np.median(e_now[ok])), now_p90=float(np.percentile(e_now[ok], 90)), now_max=float(e_now[ok].max()))
    if e_dr is not None:
        ok = wv & np.isfinite(e_dr)
        m.update(dr_med=float(np.median(e_dr[ok])), dr_p90=float(np.percentile(e_dr[ok], 90)), dr_max=float(e_dr[ok].max()))
    if log is not None:
        m.update(late_steps=log.get("late_steps", 0), forced=log.get("forced", 0), held_lost=log.get("held_lost", 0),
                 floor_w=[f["w"] for f in log["floors"]], floor_t=[f["t"] for f in log["floors"]])
    return m
