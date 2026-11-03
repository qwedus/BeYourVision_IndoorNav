"""기록 세션 재생: PDR 걸음 → PF, 경로 중간 체크포인트로 채점.

체크포인트 정답(집 복도 home_loop):
  - START: 첫 걸음 시각
  - 모서리(NE, NW, SW, SE): 자이로 누적 헤딩이 회전 중간(45°, 135°, ...)을 지나는 시각 = 그 모서리
  - 끝점: 마지막 걸음 시각 (경로별 종료 위치)
  자이로 시각은 ±0.5 s 정도로 정확하다 (Annotation은 시계 문제로 보조 확인용).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filter import PFParams, ParticleFilter
from .map import WallMap

TH0 = np.pi / 2   # 기본 출발 방향 = 북(+y)

# 경로별 출발 자세: (출발 체크포인트 이름, 출발 방향 [rad, 지도 기준])
ROUTE_START = {
    "L4": ("START_W", np.pi),   # 시계 방향 폐루프: SE 모서리 서쪽 0.883 m 에서 서쪽 보고 출발 (2026-10-29)
}


def route_start(experiment: str) -> tuple[str, float]:
    return ROUTE_START.get(experiment, ("START", TH0))


@dataclass
class Checkpoint:
    t: float
    name: str
    xy: np.ndarray


@dataclass
class ReplayResult:
    session_id: str
    rows: list[dict] = field(default_factory=list)   # 체크포인트별 오차
    track: np.ndarray | None = None                   # [n_steps, 2] PF 추정 궤적
    pdr_track: np.ndarray | None = None               # [n_steps, 2] PDR 단독 궤적 (지도 좌표)
    k_final: float = float("nan")
    unc95_final: float = float("nan")
    stats: dict = field(default_factory=dict)


def _crossings(t, h_deg, level, direction=+1):
    s = np.sign(h_deg - level)
    idx = np.where((s[:-1] != s[1:]) & (np.sign(s[1:] - s[:-1]) == direction))[0]
    return [float(t[i + 1]) for i in idx]


def route_checkpoints(experiment: str, pdr, session_t: np.ndarray, wm: WallMap) -> list[Checkpoint]:
    """경로 종류별 체크포인트 (시각, 이름, 지도 좌표). 지원 안 하는 경로는 빈 목록."""
    cp = wm.checkpoints
    st = pdr.steps
    if st.empty:
        return []
    t0, t1 = float(st.t.iloc[0]), float(st.t.iloc[-1])
    h = np.degrees(pdr.heading - pdr.heading[0])
    first = lambda lv, d=+1: next(iter(_crossings(session_t, h, lv, d)), None)
    start_name, _ = route_start(experiment)
    out = [Checkpoint(t0, "START", cp[start_name])]
    if experiment == "S1":
        out.append(Checkpoint(t1, "STRAIGHT_END", cp["STRAIGHT_END"]))
    elif experiment == "S2":
        tm = first(90)
        if tm is not None:
            out.append(Checkpoint(tm, "STRAIGHT_END", cp["STRAIGHT_END"]))
        out.append(Checkpoint(t1, "START_END", cp["START"]))
    elif experiment in ("L1", "L2"):
        for lv, name in [(45, "NE"), (135, "NW"), (225, "SW")]:
            tm = first(lv)
            if tm is not None:
                out.append(Checkpoint(tm, name, cp[name]))
        if experiment == "L1":
            out.append(Checkpoint(t1, "SE_END", cp["SE"]))
        else:
            tm = first(315)
            if tm is not None:
                out.append(Checkpoint(tm, "SE", cp["SE"]))
            out.append(Checkpoint(t1, "START_END", cp["START"]))
    elif experiment == "T2":
        tm = first(45)
        if tm is not None:
            out.append(Checkpoint(tm, "NE", cp["NE"]))
        tm = first(180)   # L자 끝점 = NW 모서리 (180° 돌아서는 중간, 2026-10-29 확인)
        if tm is not None:
            out.append(Checkpoint(tm, "NW", cp["NW"]))
        back = [x for x in _crossings(session_t, h, 225, -1)]
        if back:
            out.append(Checkpoint(back[0], "NE_back", cp["NE"]))
        out.append(Checkpoint(t1, "START_END", cp["START"]))
    elif experiment == "L4":   # 시계 방향: 헤딩이 음수로 감
        for lv, name in [(-45, "SW"), (-135, "NW"), (-225, "NE"), (-315, "SE")]:
            tm = first(lv, -1)
            if tm is not None:
                out.append(Checkpoint(tm, name, cp[name]))
        out.append(Checkpoint(t1, "START_END", cp["START_W"]))
    else:
        return []
    return sorted(out, key=lambda c: c.t)


def pdr_to_map(xy_pdr: np.ndarray, start_xy, th0: float = TH0) -> np.ndarray:
    """PDR 좌표(출발 방향 +x, 왼쪽 +y) → 지도 좌표 (출발 방향 = th0)."""
    c, s = np.cos(th0), np.sin(th0)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(start_xy) + xy_pdr @ R.T


def replay(session_id: str, experiment: str, pdr, session_t, wm: WallMap, params: PFParams,
           obs_names: set[str] | None = None, seed: int = 0,
           obs_sigma: float | None = None, obs_bias: float = 0.0) -> ReplayResult:
    """한 세션을 PF로 재생. obs_names에 든 체크포인트는 관측으로 넣고, 나머지(START 제외)는 채점."""
    obs_names = obs_names or set()
    cps = route_checkpoints(experiment, pdr, session_t, wm)
    res = ReplayResult(session_id)
    if not cps:
        return res
    st = pdr.steps
    hrel = (st.heading.to_numpy() - pdr.heading[0])
    dh = np.diff(np.r_[0.0, hrel])
    L = st.length.to_numpy()
    T = st.t.to_numpy()
    MV = st["moving"].to_numpy() if "moving" in st else np.ones(len(st), bool)
    start_name, th0 = route_start(experiment)
    start_xy = wm.checkpoints[start_name]
    pdr_map = pdr_to_map(st[["x", "y"]].to_numpy(), start_xy, th0)

    pf = ParticleFilter(params, wm, seed=seed)
    pf.init(start_xy, th0)
    track = np.zeros((len(st), 2))
    ci = 1   # cps[0] = START (초기화에 사용)
    rng = np.random.default_rng(10_000 + seed)

    def handle(c: Checkpoint, i_last: int):
        est = pf.estimate()
        pdr_xy = pdr_map[i_last] if i_last >= 0 else start_xy
        used = c.name in obs_names
        res.rows.append({
            "session_id": session_id, "checkpoint": c.name, "t": c.t, "observed": used,
            "err_pf": float(np.linalg.norm(est["xy"] - c.xy)),
            "err_pdr": float(np.linalg.norm(pdr_xy - c.xy)),
            "unc95": est["unc95"], "k": est["k"], "ess": est["ess"],
        })
        if used:
            z = c.xy + rng.normal(0, obs_sigma or 0, 2) + obs_bias
            pf.observe_position(z)

    for i in range(len(st)):
        while ci < len(cps) and cps[ci].t < T[i]:
            handle(cps[ci], i - 1)
            ci += 1
        pf.predict(float(L[i]), float(dh[i]), bool(MV[i]))
        track[i] = pf.estimate()["xy"]
    while ci < len(cps):
        handle(cps[ci], len(st) - 1)
        ci += 1
    est = pf.estimate()
    res.track, res.pdr_track = track, pdr_map
    res.k_final, res.unc95_final = est["k"], est["unc95"]
    res.stats = dict(pf.stats)
    return res
