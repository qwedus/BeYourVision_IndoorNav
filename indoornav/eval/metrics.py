"""PDR 결과 ↔ 정답 비교 지표."""

from __future__ import annotations

import numpy as np

from .ground_truth import GroundTruth


def evaluate(result, gt: GroundTruth, iphone_steps: int | None = None) -> dict:
    """한 세션의 오차 지표 dict. 정답이 없는 항목은 NaN."""
    nan = float("nan")
    m = {
        "session_id": gt.session_id,
        "experiment": gt.experiment_id,
        "split": gt.split,
        "steps_gt": gt.steps if gt.steps is not None else nan,
        "steps_est": result.n_steps,
        "steps_iphone": iphone_steps if iphone_steps is not None else nan,
        "dist_gt": gt.distance_m if gt.distance_m is not None else nan,
        "dist_est": result.distance,
    }
    m["step_err_pct"] = _pct(m["steps_est"], m["steps_gt"])
    # 걷지 않은 세션(정지·제자리 회전, 정답 0보)은 % 대신 가짜 걸음 수로 본다
    m["false_steps"] = result.n_steps if gt.steps == 0 else nan
    n_moving = int(result.steps["moving"].sum()) if "moving" in result.steps else result.n_steps
    m["moving_false_steps"] = n_moving if gt.steps == 0 else nan    # 보폭이 0 이 아닌 걸음 (위치를 실제로 옮긴 것)
    m["dist_err_pct"] = _pct(m["dist_est"], m["dist_gt"])
    m["mean_len_est"] = result.distance / result.n_steps if result.n_steps else nan

    if gt.closed:
        end_err = float(np.linalg.norm(result.end_xy - np.asarray(gt.end_xy)))
        m["end_err_m"] = end_err
        m["end_err_pct"] = 100 * end_err / gt.distance_m if gt.distance_m else nan
    else:
        m["end_err_m"] = m["end_err_pct"] = nan

    if gt.net_turn_deg is not None:
        # 좌/우 방향은 경로마다 다르므로 크기만 비교
        m["turn_gt"] = gt.net_turn_deg
        m["turn_est"] = result.net_turn_deg
        m["turn_err_deg"] = abs(result.net_turn_deg) - abs(gt.net_turn_deg)
    else:
        m["turn_gt"] = m["turn_est"] = m["turn_err_deg"] = nan
    return m


def _pct(est, gt):
    if gt is None or not np.isfinite(gt) or gt == 0:
        return float("nan")
    return 100.0 * (est - gt) / gt
