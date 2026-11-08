"""과신 대책 w1·w2·w3(같은 track 관측 우도 L^alpha) 단위 시험.

녹화 재생은 하지 않는다. test_overconf_fix.py의 가짜 1층 지도·파티클·필터 대역을 그대로 쓴다.
"""
import random

import numpy as np
import pytest

from test_overconf_fix import FLOOR, PC, PW, PX, PY, _filter, _fmap, _ob, ovf  # 가짜 지도·파티클·필터 대역
from vision_filter import VisionFilter

import exp_cause  # noqa: E402  (test_overconf_fix가 integration/을 sys.path에 넣은 뒤)

AR1 = 0.07 / 1.93


def _norm(logw):
    return logw - (logw.max() + np.log(np.exp(logw - logw.max()).sum()))


def _logl(ob, params=None, lik=None, fmap=None):
    """관측 하나의 로그 우도 (FixFilter와 같은 바닥 1e-300)"""
    lik = lik or ovf.obs_likelihood
    return np.log(np.maximum(lik(PX, PY, PC, FLOOR, [ob], fmap or _fmap(), params), 1e-300))


@pytest.fixture
def gated(monkeypatch):
    """exp_cause.gated_likelihood(실제 함수)를 가짜 지도에 맞춰 연결: lm_json 치수, far_gate 0.3"""
    fm = _fmap()
    fm[FLOOR]["lm_json"] = [dict(size=[0.4, 0.3]), dict(size=[1.0, 0.5])]
    monkeypatch.setattr(exp_cause, "_C", {"far_gate": 0.3})
    monkeypatch.setitem(ovf._W, "gated", exp_cause.gated_likelihood)
    return fm


# ---------------------------------------------------------------- 1. parse
def test_parse_w1():
    assert ovf.parse("w1") == dict(eps=None, gate=None, cap=None, skip=None, link=None, wt="sqrt")


def test_parse_e3_g30_w2():
    assert ovf.parse("e3_g30_w2") == dict(eps=0.3, gate=0.3, cap=None, skip=None, link=None, wt="ar1")


def test_parse_e3_g30_w3():
    assert ovf.parse("e3_g30_w3") == dict(eps=0.3, gate=0.3, cap=None, skip=None, link=None, wt="sat3")


def test_new_combos_listed_and_parse():
    new = ("w1", "e3_g30_w1", "e3_g30_w2", "e3_g30_w3")
    kinds = {"w1": "sqrt", "w2": "ar1", "w3": "sat3"}
    for c in new:
        assert c in ovf.COMBOS
        p = ovf.parse(c)
        assert p["wt"] == kinds[c.split("_")[-1]]
        assert p["cap"] is None          # cap과 wt는 함께 안 씀 (update에서 cap이 우선)
    assert len(set(ovf.COMBOS)) == len(ovf.COMBOS)


def test_w2_r_value():
    assert ovf.W2_R == 0.93


# ---------------------------------------------------------------- 2. wt_alpha
@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_alpha_first_is_one(kind):
    assert ovf.wt_alpha(1, kind) == 1.0


def test_alpha_sqrt():
    assert ovf.wt_alpha(4, "sqrt") == pytest.approx(0.5)
    assert ovf.wt_alpha(9, "sqrt") == pytest.approx(1 / 3)
    a = [ovf.wt_alpha(k, "sqrt") for k in range(1, 30)]
    assert np.all(np.diff(a) < 0)


@pytest.mark.parametrize("k", [2, 3, 10, 1000])
def test_alpha_ar1_constant_after_first(k):
    assert ovf.wt_alpha(k, "ar1") == pytest.approx(AR1, rel=1e-12)


def test_alpha_sat3_values():
    assert [ovf.wt_alpha(k, "sat3") for k in (1, 2, 3, 4)] == pytest.approx([1.0, 0.5, 0.3, 0.2])


def test_alpha_sat3_partial_sums():
    s = np.cumsum([ovf.wt_alpha(k, "sat3") for k in range(1, 51)])
    k = np.arange(1, 51)
    assert s == pytest.approx(3 * k / (k + 2), rel=1e-12)
    assert np.all(s < 3)
    assert np.all(np.diff(s) > 0)


@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_alpha_depends_only_on_k(kind):
    ks = list(range(1, 40))
    ref = {k: ovf.wt_alpha(k, kind) for k in ks}
    sh = ks[:]
    random.Random(0).shuffle(sh)
    for k in sh + sh[::-1]:          # 부르는 순서·반복과 상관없음 (상태 없음)
        assert ovf.wt_alpha(k, kind) == ref[k]


@pytest.mark.parametrize("kind", ["cap", "", None, "SQRT"])
def test_alpha_unknown_kind(kind):
    with pytest.raises(ValueError):
        ovf.wt_alpha(2, kind)


# ---------------------------------------------------------------- 3. FixFilter track 셈·기록
@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_interleaved_tracks_count_separately(kind):
    F = _filter(ovf.FixFilter, fix=dict(wt=kind))
    obs = [_ob(0, track=1), _ob(1, track=2), _ob(2, track=1), _ob(3, track=2), _ob(4, track=1)]
    F.update(obs, 0.0, t=1.0)
    assert F.track_n == {1: 3, 2: 2}
    a = lambda k: ovf.wt_alpha(k, kind)  # noqa: E731
    assert F.log["weights"] == [a(1), a(1), a(2), a(2), a(3)]


def test_counts_continue_across_updates_and_new_track_restarts():
    F = _filter(ovf.FixFilter, fix=dict(wt="sat3"))
    F.update([_ob(0, track=1), _ob(1, track=1)], 0.0, t=1.0)
    F.update([_ob(2, track=1), _ob(3, track=7)], 0.0, t=2.0)          # track 1은 이어서 3번째, 새 track 7은 1부터
    assert F.track_n == {1: 3, 7: 1}
    assert F.log["weights"] == pytest.approx([1.0, 0.5, 0.3, 1.0])


def test_no_track_alpha_one_and_no_counter():
    F = _filter(ovf.FixFilter, fix=dict(wt="sqrt"))
    obs = [_ob(0, track=1), _ob(1), _ob(2, track=1), _ob(3)]
    F.update(obs, 0.0, t=1.0)
    assert F.track_n == {1: 2}                        # track 없는 관측은 셈에 안 들어감
    assert F.log["weights"] == pytest.approx([1.0, 1 / np.sqrt(2)])          # track 없는 관측은 기록 안 함
    # 우도에는 track 없는 관측이 가중 1로 들어감
    exp = np.log(PW) + _logl(obs[0]) + _logl(obs[1]) + _logl(obs[2]) / np.sqrt(2) + _logl(obs[3])
    assert F.pf.logw == pytest.approx(_norm(exp), abs=1e-9)


def test_weights_real_time_prefix():
    """k번째 관측 가중은 뒤에 올 관측과 상관없음: 앞부분만 넣은 기록 = 전체 기록의 앞부분"""
    obs = [_ob(i, track=1 + i % 2) for i in range(8)]
    a = _filter(ovf.FixFilter, fix=dict(wt="sat3"))
    b = _filter(ovf.FixFilter, fix=dict(wt="sat3"))
    a.update(obs[:3], 0.0, t=1.0)
    b.update(obs, 0.0, t=1.0)
    assert b.log["weights"][:3] == a.log["weights"]


# ---------------------------------------------------------------- 4. 우도 = alpha × 로그 우도
@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_logw_is_alpha_weighted_sum(kind):
    obs = [_ob(0, track=1), _ob(1, track=1, bearing=12.0), _ob(2, track=1, bearing=-5.0), _ob(3, typ="station_map")]
    F = _filter(ovf.FixFilter, fix=dict(wt=kind))
    F.update(obs, 0.0, t=1.0)
    al = [ovf.wt_alpha(k, kind) for k in (1, 2, 3)] + [1.0]
    exp = np.log(PW) + sum(a * _logl(o) for a, o in zip(al, obs))
    assert F.pf.logw == pytest.approx(_norm(exp), abs=1e-9)
    assert F.pf.n_resample == 1 and F.diag["n_update"] == 1


def test_logw_eps_floor_before_power():
    """eps 바닥은 관측 하나 점수 안에서 먼저 걸리고, 그 로그에 alpha를 곱함"""
    params = {"eps": 0.3}
    obs = [_ob(0, track=4), _ob(1, track=4, bearing=40.0)]
    F = _filter(ovf.FixFilter, fix=dict(eps=0.3, wt="sqrt"), obs_params=params)
    F.update(obs, 0.0, t=1.0)
    exp = np.log(PW) + _logl(obs[0], params) + _logl(obs[1], params) / np.sqrt(2)
    assert F.pf.logw == pytest.approx(_norm(exp), abs=1e-9)
    # 바닥이 실제로 걸리는 파티클(후보 없음)이 있어 점수가 0이 아님
    assert np.all(np.isfinite(_logl(obs[1], params)))


def test_logw_gated_alpha(gated):
    obs = [_ob(0, track=1), _ob(1, track=1, bearing=12.0), _ob(2, track=1, bearing=-3.0)]
    obs = [dict(o, f_over_h=10.0) for o in obs]
    F = _filter(ovf.FixFilter, fix=dict(gate=0.3, wt="sat3"))
    F.fmap = gated
    F.update(obs, 1.5, t=1.0)
    p = dict(max_range=ovf.GATE_RANGE)
    lg = lambda o: np.log(np.maximum(exp_cause.gated_likelihood(PX, PY, PC - 1.5, FLOOR, [o], gated, p), 1e-300))  # noqa: E731
    exp = np.log(PW) + lg(obs[0]) + 0.5 * lg(obs[1]) + 0.3 * lg(obs[2])
    assert F.pf.logw == pytest.approx(_norm(exp), abs=1e-9)


@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_all_alpha_one_equals_gate_only_exactly(gated, kind):
    """track마다 관측 하나뿐(모두 alpha 1) + track 없는 관측: wt 켠 결과 = gate만 켠 관측별 경로와 비트까지 같음"""
    obs = [_ob(0, track=1), _ob(1, track=2, bearing=12.0), _ob(2), _ob(3, track=3, typ="station_map")]
    obs = [dict(o, f_over_h=10.0) for o in obs]
    a = _filter(ovf.FixFilter, fix=dict(gate=0.3))
    b = _filter(ovf.FixFilter, fix=dict(gate=0.3, wt=kind))
    a.fmap = b.fmap = gated
    a.update(obs, 2.0, t=1.0)
    b.update(obs, 2.0, t=1.0)
    assert b.pf.logw.tobytes() == a.pf.logw.tobytes()
    assert b.log["weights"] == [1.0, 1.0, 1.0]
    assert a.log["weights"] == []


@pytest.mark.parametrize("kind", ["sqrt", "ar1", "sat3"])
def test_all_alpha_one_matches_parent(kind):
    """한계 20 m·가중 1이면 관측별 경로 = 부모 한 번 계산 (정규화 뒤 상수 차이 사라짐)"""
    obs = [_ob(0, track=1), _ob(1, track=2, bearing=12.0), _ob(2, typ="station_map")]
    a = _filter(VisionFilter)
    b = _filter(ovf.FixFilter, fix=dict(wt=kind))
    VisionFilter.update(a, obs, 0.0)
    b.update(obs, 0.0, t=1.0)
    assert b.pf.logw == pytest.approx(a.pf.logw, abs=1e-9)


# ---------------------------------------------------------------- 5. 기본 꺼짐
def test_base_wt_none():
    assert ovf.parse("base")["wt"] is None


def test_base_calls_parent_update(monkeypatch):
    calls = []
    monkeypatch.setattr(VisionFilter, "update", lambda self, obs, d=0.0: calls.append((obs, d)))
    F = _filter(ovf.FixFilter, fix=ovf.parse("base"))
    obs = [_ob(0, track=1)]
    F.update(obs, 1.0, t=1.0)
    assert calls == [(obs, 1.0)]
    assert F.track_n == {} and F.log == dict(skipped=[], weights=[])


def test_base_bitwise_equals_parent():
    obs = [_ob(0, track=1), _ob(1, track=1, bearing=12.0), _ob(2, typ="station_map")]
    a = _filter(VisionFilter)
    b = _filter(ovf.FixFilter, fix=ovf.parse("base"))
    VisionFilter.update(a, obs, 2.0)
    b.update(obs, 2.0, t=1.0)
    assert b.pf.logw.tobytes() == a.pf.logw.tobytes()


def test_cap_and_wt_together_rejected():
    with pytest.raises(ValueError):
        ovf.parse("e3_g30_c3_w1")
