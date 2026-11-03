"""여러 세션을 한 번에 돌리고 지표 표를 만든다 (scripts/ 에서 사용)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import load_session
from ..pdr import PDRParams, run_pdr
from ..pdr.motion import walking_bounds
from .ground_truth import GroundTruth, load_ground_truth
from .metrics import evaluate


def find_session_path(data_dir: Path, session_id: str) -> Path | None:
    for cand in (data_dir / session_id, data_dir / f"{session_id}.zip"):
        if cand.exists():
            return cand
    return None


def load_all(data_dir: str | Path, sessions_csv: str | Path, routes_csv: str | Path,
             split: str | None = None, fs: float = 100.0, include_ref: bool = False):
    """(session, gt) 목록. flag가 bad/redo 인 세션과 파일이 없는 세션은 건너뛴다.
    flag=ref(극단 조건 참고용, 튜닝 제외)는 include_ref=True 일 때만 포함."""
    data_dir = Path(data_dir)
    gts = load_ground_truth(sessions_csv, routes_csv)
    out, missing = [], []
    for sid, gt in gts.items():
        if gt.flag in ("bad", "redo") or (gt.flag == "ref" and not include_ref) or (split and gt.split != split):
            continue
        p = find_session_path(data_dir, sid)
        if p is None:
            missing.append(sid)
            continue
        out.append((load_session(p, fs=fs), gt))
    return out, missing


def evaluate_all(pairs: list[tuple[object, GroundTruth]], params: PDRParams) -> pd.DataFrame:
    rows = []
    for s, gt in pairs:
        r = run_pdr(s, params)
        m = evaluate(r, gt, s.iphone_steps)
        wb = walking_bounds(s)
        m["rec_s"] = s.duration
        m["walk_s"] = wb.duration_s
        m["still_s"] = wb.lead_still_s + wb.tail_still_s   # 앞뒤 정지 구간 합
        m["speed_gt"] = gt.distance_m / wb.duration_s if gt.distance_m and wb.duration_s else float("nan")
        m["cadence_gt"] = gt.steps / wb.duration_s if gt.steps and wb.duration_s else float("nan")
        rows.append(m)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """split별 평균 절대 오차."""
    cols = ["step_err_pct", "dist_err_pct", "end_err_pct", "turn_err_deg"]
    return df.groupby("split")[cols].agg(lambda x: x.abs().mean()).round(2)
