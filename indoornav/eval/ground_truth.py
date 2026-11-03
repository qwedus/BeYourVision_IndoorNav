"""정답(GT) 읽기: docs/sessions.csv(세션별) + docs/routes.csv(경로별)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class GroundTruth:
    session_id: str
    experiment_id: str
    split: str                    # tune / eval
    distance_m: float | None      # 실제 이동 거리
    steps: int | None             # 직접 센 걸음 수
    closed: bool                  # 종료 위치 = 출발 위치 인가
    net_turn_deg: float | None    # 출발~종료 전체 회전량 (부호 무시하고 비교)
    flag: str = ""
    end_xy: tuple[float, float] = (0.0, 0.0)   # closed 경로의 실제 종료 위치 (출발 방향 기준: 앞 +x, 왼쪽 +y)


def load_ground_truth(sessions_csv: str | Path, routes_csv: str | Path) -> dict[str, GroundTruth]:
    s = pd.read_csv(sessions_csv, dtype={"date": str, "start_time": str}).fillna("")
    r = pd.read_csv(routes_csv).set_index("experiment_id")
    out = {}
    for _, row in s.iterrows():
        exp = row["experiment_id"]
        route = r.loc[exp] if exp in r.index else None
        out[row["session_id"]] = GroundTruth(
            session_id=row["session_id"],
            experiment_id=exp,
            split=row["split"],
            distance_m=_num(row["distance_m"]),
            steps=int(_num(row["manual_step_count"])) if _num(row["manual_step_count"]) is not None else None,
            closed=str(route["closed"]).strip().lower() == "true" if route is not None else False,
            net_turn_deg=_num(route["net_turn_deg"]) if route is not None else None,
            flag=str(row["flag"]),
            end_xy=_end_xy(route),
        )
    return out


def _num(v) -> float | None:
    try:
        f = float(v) if v != "" else None
    except (TypeError, ValueError):
        return None
    return None if f is None or f != f else f   # NaN → None


def _end_xy(route) -> tuple[float, float]:
    if route is None or "end_fwd_m" not in route.index:
        return (0.0, 0.0)
    return (_num(route["end_fwd_m"]) or 0.0, _num(route["end_left_m"]) or 0.0)
