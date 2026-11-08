"""과신 대책 시험 스크립트(integration/exp_overconf_fix.py) 단위 시험.

녹화 재생은 하지 않는다. 작은 가짜 1층 지도와 파티클 몇 개로 옵션 해석·track 가중·후보 비율·update 분기만 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402,F401  (vision/·indoornav 경로 준비)
import exp_overconf_fix as ovf  # noqa: E402
from vision_filter import VisionFilter  # noqa: E402

FLOOR = -1


# ---------------------------------------------------------------- 가짜 지도·파티클
def _fmap():
    """10×10 칸(1 m), 회전·이동 없음. 랜드마크 0 = exit_sign (5, 9): x < 5 칸에서만 보임(거리 5).
    랜드마크 1 = info_monitor (5, 0): 어디서도 안 보임. computed는 전부 True."""
    H = W = 10
    dist = np.full((2, H, W), np.nan)
    dist[0, :, :5] = 5.0
    fm = dict(x0=0.0, y0=0.0, cell=1.0, R=np.eye(2), T=np.zeros(2), dist=dist,
              computed=np.ones((H, W), bool), lm_xy=np.array([[5.0, 9.0], [5.0, 0.0]]),
              lm_dz=np.zeros(2), lm_type=np.array(["exit_sign", "info_monitor"]))
    return {FLOOR: fm}


# 파티클: (4,5) 북쪽 → 후보 / (6,5) 북쪽 → 안 보이는 칸 / (4,5) 남쪽 → 화각 밖 / (4.5,5) 북쪽 → 후보 / (-1,5) → 격자 밖
PX = np.array([4.0, 6.0, 4.0, 4.5, -1.0])
PY = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
PC = np.array([0.0, 0.0, 180.0, 0.0, 0.0])
PW = np.array([0.1, 0.2, 0.3, 0.25, 0.15])


def _ob(det, track=None, typ="exit_sign", bearing=10.0, distance=4.0, score=0.9):
    return dict(detection_id=det, track_id=track, type=typ, bearing=bearing, distance=distance, score=score)


class _FakePF:
    """indoornav ParticleFilter 대역: logw, _normalize(같은 식), _maybe_resample(부른 횟수만 셈)"""

    def __init__(self, w):
        self.logw = np.log(np.asarray(w, float))
        self.n_resample = 0

    def _normalize(self):
        m = np.max(self.logw)
        self.logw = self.logw - (m + np.log(np.sum(np.exp(self.logw - m))))

    def _maybe_resample(self):
        self.n_resample += 1


def _filter(cls, fix=None, obs_params=None):
    """무거운 __init__(지도·PF 생성) 없이 update에 필요한 속성만 채움"""
    F = object.__new__(cls)
    F.pf = _FakePF(PW)
    F.enu_compass = lambda: (PX.copy(), PY.copy(), PC.copy())
    F.fmap, F.floor, F.obs_params = _fmap(), FLOOR, obs_params
    F.diag = dict(t_predict=0.0, t_update=0.0, n_update=0, miss_sum=0.0, miss_max=0.0, snap_redraw=0)
    if cls is ovf.FixFilter:
        F.fix, F.track_n, F.log = fix or {}, {}, dict(skipped=[], weights=[])
    return F


# ---------------------------------------------------------------- 1. parse
def test_parse_base_all_none():
    assert ovf.parse("base") == dict(eps=None, gate=None, cap=None, skip=None, link=None, wt=None)


def test_parse_combo():
    assert ovf.parse("e3_g30_c5_s20") == dict(eps=0.3, gate=0.3, cap=5.0, skip=0.2, link=None, wt=None)


def test_parse_all_combos_known():
    for c in ovf.COMBOS:
        ovf.parse(c)          # 조각 이름이 PIECES에 다 있음


# ---------------------------------------------------------------- 2. cap_weight
@pytest.mark.parametrize("W", [3.0, 5.0])
def test_cap_weight_sum_bounded(W):
    w = np.array([ovf.cap_weight(k, W) for k in range(1, 1001)])
    assert w[0] == 1.0
    assert np.all(np.diff(w) < 0)
    assert w[1] == pytest.approx(np.exp(-1 / W))
    assert w.sum() <= W / (1 - np.exp(-1 / W)) + 1e-9
    assert w.sum() < W + 1


@pytest.mark.parametrize("W", [None, 0])
def test_cap_weight_off(W):
    assert ovf.cap_weight(1, W) == 1.0
    assert ovf.cap_weight(7, W) == 1.0


# ---------------------------------------------------------------- 3. cand_fraction
def test_cand_fraction_weighted():
    f = ovf.cand_fraction(PX, PY, PC, FLOOR, _ob(0), _fmap(), None, PW)
    assert f == pytest.approx(0.1 + 0.25)


def test_cand_fraction_range_limit():
    # 표 거리 5 m > max_range 4 m → 후보 없음
    assert ovf.cand_fraction(PX, PY, PC, FLOOR, _ob(0), _fmap(), dict(max_range=4.0), PW) == 0.0


def test_cand_fraction_no_kind_rows():
    assert ovf.cand_fraction(PX, PY, PC, FLOOR, _ob(0, typ="station_map"), _fmap(), None, PW) == 0.0


def test_cand_fraction_kind_not_visible_anywhere():
    assert ovf.cand_fraction(PX, PY, PC, FLOOR, _ob(0, typ="info_monitor"), _fmap(), None, PW) == 0.0


def test_cand_fraction_no_floor_map():
    assert ovf.cand_fraction(PX, PY, PC, -2, _ob(0), _fmap(), None, PW) == 0.0


# ---------------------------------------------------------------- 4. FixFilter.update
OBS = [_ob(0, track=1), _ob(1, track=1, bearing=12.0), _ob(2, typ="station_map")]


@pytest.mark.parametrize("fix", [None, {}, dict(eps=0.3, gate=None, cap=None, skip=None)])
def test_update_no_option_equals_parent(fix):
    a = _filter(VisionFilter, obs_params={"eps": 0.3} if fix else None)
    b = _filter(ovf.FixFilter, fix=fix, obs_params={"eps": 0.3} if fix else None)
    VisionFilter.update(a, OBS, 2.0)
    b.update(OBS, 2.0, t=1.0)
    assert b.pf.logw.tobytes() == a.pf.logw.tobytes()
    assert b.diag["n_update"] == a.diag["n_update"] == 1
    assert b.log == dict(skipped=[], weights=[])


def test_update_cap_without_track_matches_parent():
    """가중 1(track 없음)·건너뛰기 없음·한계 20 m: 관측별 계산 합 = 한 번 계산 (정규화 뒤 상수 차이 사라짐)"""
    obs = [{**o, "track_id": None} for o in OBS]
    a = _filter(VisionFilter)
    b = _filter(ovf.FixFilter, fix=dict(cap=5.0))
    VisionFilter.update(a, obs, 0.0)
    b.update(obs, 0.0, t=1.0)
    assert b.pf.logw == pytest.approx(a.pf.logw, abs=1e-9)
    assert b.log["weights"] == []


def test_update_skip_all():
    F = _filter(ovf.FixFilter, fix=dict(skip=1.01))
    before = F.pf.logw.copy()
    F.update(OBS, 0.0, t=3.5)
    assert F.pf.logw == pytest.approx(before, abs=1e-12)
    assert F.log["skipped"] == [(3.5, 0), (3.5, 1), (3.5, 2)]
    assert F.track_n == {}          # 건너뛴 관측은 track 수에 안 셈


def test_update_skip_partial():
    """비율 0.35인 exit_sign 관측은 문턱 0.2를 넘고, 후보 없는 station_map 관측만 건너뜀"""
    F = _filter(ovf.FixFilter, fix=dict(skip=0.2))
    F.update(OBS, 0.0, t=2.0)
    assert F.log["skipped"] == [(2.0, 2)]


def test_update_cap_second_weight():
    W = 5.0
    F = _filter(ovf.FixFilter, fix=dict(cap=W))
    F.update(OBS[:2], 0.0, t=1.0)
    assert F.log["weights"] == pytest.approx([1.0, np.exp(-1 / W)])
    assert F.track_n == {1: 2}
    F.update([_ob(5, track=1)], 0.0, t=2.0)        # 다음 프레임에도 이어서 셈
    assert F.log["weights"][-1] == pytest.approx(np.exp(-2 / W))


def test_update_cap_logw_is_weighted_sum():
    W = 5.0
    F = _filter(ovf.FixFilter, fix=dict(cap=W))
    F.update(OBS[:2], 0.0, t=1.0)
    l0 = np.log(np.maximum(ovf.obs_likelihood(PX, PY, PC, FLOOR, [OBS[0]], _fmap(), None), 1e-300))
    l1 = np.log(np.maximum(ovf.obs_likelihood(PX, PY, PC, FLOOR, [OBS[1]], _fmap(), None), 1e-300))
    exp = np.log(PW) + l0 + np.exp(-1 / W) * l1
    exp = exp - (exp.max() + np.log(np.exp(exp - exp.max()).sum()))
    assert F.pf.logw == pytest.approx(exp, abs=1e-9)
