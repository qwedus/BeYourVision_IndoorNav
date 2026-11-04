"""걸음 사이 회전 보정 (비전 코어, I1 연결안 3절 4-2, 11-04): 촬영 순간 방향과 PF에 마지막으로 들어간 걸음 방향의 차이.

PF 방향은 걸음이 끝날 때만 바뀌는데 사진은 초당 5장이라, 걸음 사이에 돈 만큼 비교 기준이 틀어진다.
    Δ = 촬영 시각 PDR 방향 - PF에 마지막으로 반영된 걸음의 PDR 방향   (샘플별 방향, 반시계 +, rad)
PF 상태에는 더하지 않는다(다음 걸음에서 같은 회전이 두 번 들어감). 관측을 비교할 때만 쓴다:
    - 'theta'  : 파티클 나침반 방향 - deg(Δ)로 관측 모델에 넘김 (모든 파티클 방향에 Δ를 더한 것과 정확히 같음, 화각 검사까지 맞음)
    - 'bearing': 관측 방위 - deg(Δ)로 바꿈 (obs_model만 있으면 되지만 화각 검사는 옛 방향 기준으로 남음)
numpy만 쓴다(vision-integration 브랜치 vision/에 그대로 복사됨 — 원본은 localization_design).
"""
import numpy as np


def rotation_since_step(t, t_series, h_series, step_t, step_h, h0=0.0):
    """시각 t들에서 Δ [rad]. t_series·h_series: 샘플별 방향(이어진 값), step_t·step_h: 걸음 시각·그 걸음의 방향(시각 순).
    같은 시각이면 걸음이 먼저 반영된 것으로 본다. 첫 걸음 전 기준은 h0(녹화 시작 방향)."""
    t = np.asarray(t, float)
    now = np.interp(t, t_series, h_series)
    idx = np.searchsorted(step_t, t, side="right") - 1
    return now - np.where(idx >= 0, np.asarray(step_h, float)[np.maximum(idx, 0)], h0)


def fix_bearing(obs, dpsi_deg):
    """관측 한 프레임의 방위를 Δ[도, 반시계 +]만큼 고친 사본 (오른쪽 + 방위라 빼기)."""
    return [dict(o, bearing=o["bearing"] - dpsi_deg) for o in obs]
