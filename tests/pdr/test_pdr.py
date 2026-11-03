from pathlib import Path

import numpy as np
import pytest

from indoornav.io import describe, load_session
from indoornav.pdr import PDRParams, run_pdr
from indoornav.pdr.step_length import calibrate_k

ROOT_CFG = Path(__file__).resolve().parents[2] / "configs" / "pdr_default.toml"


def test_default_config_matches_code():
    """TOML 기본값과 코드 기본값이 어긋나지 않는지."""
    assert PDRParams.from_toml(ROOT_CFG) == PDRParams()


@pytest.mark.parametrize("unit_g", [True, False])
def test_loader_units(walk, unit_g):
    d, *_ = walk(unit_g=unit_g)
    s = load_session(d)
    assert abs(np.median(np.linalg.norm(s.gravity, axis=1)) - 9.80665) < 1e-3
    assert np.median(np.linalg.norm(s.acc_user, axis=1)) < 3.0
    assert s.iphone_steps is not None
    assert {"Accelerometer.csv", "Gyroscope.csv"} <= set(describe(d)["file"])


def test_step_count_straight(walk):
    d, n, dist, _ = walk(legs=((30, 0.0),))
    r = run_pdr(load_session(d))
    assert abs(r.n_steps - n) <= 1


def test_weinberg_calibration_recovers_distance(walk):
    d, n, dist, _ = walk(legs=((30, 0.0),))
    s = load_session(d)
    r = run_pdr(s)
    k = calibrate_k([r.steps["feature"].to_numpy()], [dist])
    r2 = run_pdr(s, PDRParams().with_(**{"length.k": k}))
    assert abs(r2.distance - dist) / dist < 0.02


def test_square_loop_closes(walk):
    legs = ((20, 0.0), (20, 90.0), (20, 90.0), (20, 90.0))
    d, n, dist, turn = walk(legs=legs)
    s = load_session(d)
    r = run_pdr(s, PDRParams().with_(**{"length.model": "constant", "length.constant_m": 0.6}))
    assert abs(r.net_turn_deg - turn) < 3.0
    # 정사각형 4변(회전 3번) → 출발점 복귀. 폐루프 오차 2% 이내
    assert np.linalg.norm(r.end_xy) < 0.02 * dist


def test_zaru_reduces_bias_drift(walk):
    d, *_ = walk(legs=((20, 0.0),), gyro_bias=np.radians(0.2), stand_s=20.0)
    s = load_session(d)
    off = run_pdr(s, PDRParams())
    on = run_pdr(s, PDRParams().with_(**{"heading.zaru": True}))
    assert abs(on.net_turn_deg) < abs(off.net_turn_deg)


def test_walking_bounds_trims_standing(walk):
    from indoornav.pdr.motion import walking_bounds
    d, n, *_ = walk(legs=((30, 0.0),), cadence_hz=1.8, stand_s=5.0)
    wb = walking_bounds(load_session(d))
    assert abs(wb.lead_still_s - 5.0) < 0.8
    assert abs(wb.tail_still_s - 5.0) < 0.8
    assert abs(wb.duration_s - 30 / 1.8) < 1.5


def test_delayed_filter_matches_offline(walk):
    """0.5초 지연 필터(앱 방식)는 오프라인 필터와 거의 같은 걸음 수를 내야 한다."""
    d, n, *_ = walk(legs=((30, 0.0),))
    s = load_session(d)
    off = run_pdr(s, PDRParams())
    dly = run_pdr(s, PDRParams().with_(**{"preprocess.mode": "delayed"}))
    assert abs(off.n_steps - dly.n_steps) <= 1
    assert np.corrcoef(off.signal, dly.signal)[0, 1] > 0.98


def test_legacy_causal_key():
    p = PDRParams.from_dict({"preprocess": {"causal": True}})
    assert p.preprocess.mode == "causal"


def test_weinberg_freq_runs(walk):
    d, *_ = walk(legs=((20, 0.0),))
    r = run_pdr(load_session(d), PDRParams().with_(**{"length.model": "weinberg_freq",
                                                        "length.k_freq": 0.1}))
    assert r.n_steps > 0 and "freq" in r.steps


def test_adaptive_counts_and_ignores_standing(walk):
    """적응형 걸음 검출: 보통 걸음을 세고, 앞뒤 정지 구간에서는 걸음을 내지 않는다."""
    d, n, *_ = walk(legs=((30, 0.0),), stand_s=8.0)
    s = load_session(d)
    r = run_pdr(s, PDRParams().with_(**{"step.method": "adaptive"}))
    assert abs(r.n_steps - n) <= 2
    t = r.steps["t"].to_numpy()
    assert t.min() > 7.0 and t.max() < s.duration - 7.0     # 정지 8초 구간에 걸음 없음


def test_online_calibration_converges_and_rejects():
    from indoornav.pdr.online_calib import StrideCalibrator
    true_k, f = 0.38, 1.6                      # feature 1.6 → 실제 보폭 0.608 m
    cal = StrideCalibrator(k=0.25)
    t = 0.0
    for _ in range(6):                         # 직선 10 m 구간을 6번
        t_start = t
        for _ in range(16):
            t += 0.8
            cal.add_step(t, f, heading=0.0)
        cal.correct(t_start, t, distance_m=true_k * f * 16, uncertainty_m=0.3)
    assert abs(cal.k - true_k) / true_k < 0.05
    k = cal.k
    t_start = t                                # 회전이 있는 구간은 반영하지 않음
    for i in range(16):
        t += 0.8
        cal.add_step(t, f, heading=np.radians(90) * i / 15)
    assert not cal.correct(t_start, t, 9.7, 0.3).applied and cal.k == k
    t_start = t                                # 랜드마크 오인식 같은 이상값은 반영하지 않음
    for _ in range(16):
        t += 0.8
        cal.add_step(t, f, heading=0.0)
    assert not cal.correct(t_start, t, 30.0, 0.3).applied and cal.k == k


def test_bout_mask_zeroes_short_bouts():
    import numpy as np
    from indoornav.pdr.step_length import bout_mask
    fs = 100
    # 2걸음 고립 묶음 + 6걸음 연속 걷기 + 1걸음 고립
    steps = np.array([0, 60, 1000, 1060, 1120, 1180, 1240, 1300, 2000])
    m = bout_mask(steps, fs, min_bout_steps=4, bout_gap_s=3.0)
    assert m.tolist() == [False, False] + [True] * 6 + [False]
    assert bout_mask(steps, fs, 0, 3.0).all()
