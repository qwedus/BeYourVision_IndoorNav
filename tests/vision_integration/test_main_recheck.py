"""main 건의 재확인 실험(integration/exp_main_recheck.py)의 PF 바꾸기 단위 시험. 녹화 재생은 하지 않는다.

가짜 격자 지도(가운데 세로 벽)에서 기본 값이 indoornav predict와 비트까지 같은지, 벌점·ESS 리샘플·미끄러짐 분기를 본다.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402,F401
import exp_main_recheck as E  # noqa: E402
from indoornav.pf.filter import ParticleFilter, PFParams  # noqa: E402
from indoornav.pf.gridmap import GridMap  # noqa: E402


def _map():
    """20 m × 20 m, 0.1 m 칸. x = 10 m에 세로 벽(두께 0.2 m)."""
    walk = np.ones((200, 200), bool)
    walk[:, 99:101] = False
    return GridMap(walk, 0.1, np.array([0.0, 20.0]))


def _pf(cls, seed=3, **kw):
    P = replace(PFParams(), n=400, sigma_len=0.15, sigma_k=0.02)
    pf = cls(P, _map(), seed=seed, **kw)
    pf.init(np.array([8.0, 10.0]), 0.0)          # 동쪽(벽 쪽)으로 향함
    return pf


STEPS = [(0.7, 0.0, True)] * 4 + [(0.6, 0.3, True), (0.0, 0.0, False), (0.7, -0.2, True)]


def test_default_same_as_indoornav():
    a, b = _pf(ParticleFilter), _pf(E.PFX)
    for s in STEPS:
        a.predict(*s)
        b.predict(*s)
        for f in ("xy", "th", "k", "logw"):
            assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
    assert a.stats == b.stats
    assert len(b.slog) == len(STEPS)
    assert any(x[5] > 0 for x in b.slog)          # 벽에 막힌 걸음이 있었음


def test_penalty_keeps_particles_alive():
    b = _pf(E.PFX, wall="pen", pen=0.1)
    for s in STEPS:
        b.predict(*s)
        assert np.isfinite(b.logw).all()
    assert b.stats["wall_kills"] > 0
    assert np.isclose(np.exp(b.logw).sum(), 1.0)


def test_ess_resample_only_below_threshold():
    b = _pf(E.PFX, res="ess")
    for s in STEPS:
        b.predict(*s)
    for dk_rw, dk_wall, dk_res, ess, res, nh, moved in b.slog:
        if res:
            assert ess < b.p.ess_frac * b.p.n
        if moved and not res:
            assert dk_res == 0.0
    assert any(x[5] > 0 and not x[4] for x in b.slog)   # 막혔지만 리샘플 안 한 걸음


def test_slide_fewer_kills_and_no_wall_crossing():
    a, b = _pf(E.PFX), _pf(E.PFX, wall="slide")
    for s in STEPS:
        a.predict(*s)
        b.predict(*s)
    assert b.stats["wall_kills"] < a.stats["wall_kills"]
    assert b.slides > 0


def test_slide_moves_do_not_cross_walls():
    b = _pf(E.PFX, wall="slide")
    m = _map()
    for s in STEPS:
        prev = b.xy.copy()
        n0 = b.stats["resamples"]
        b.predict(*s)
        if b.stats["resamples"] == n0:            # 리샘플이 없으면 파티클 순서가 그대로라 이동을 짝지어 볼 수 있음
            assert not m.crosses(prev, b.xy).any()
    assert (b.xy[:, 0] < 9.9).all()               # 출발점 쪽(서쪽)에 남음


def _tiny(**kw):
    """파티클 4개: 0번은 점수 0, 1~3번은 벽 바로 앞(동쪽 향함), 0번은 벽에서 멂."""
    P = replace(PFParams(), n=4, sigma_len=0.0, sigma_k=0.0, sigma_h0=0.0, sigma_h1=0.0, rough_xy=0.0, rough_h=0.0, rough_k=0.0)
    pf = E.PFX(P, _map(), seed=0, **kw)
    pf.init(np.array([9.5, 10.0]), 0.0)
    pf.xy = np.array([[2.0, 10.0], [9.7, 10.0], [9.7, 12.0], [9.7, 8.0]])
    pf.th = np.zeros(4)
    pf.k = np.ones(4)
    pf.logw = np.array([-np.inf, *np.full(3, -np.log(3))])
    return pf


def test_ess_dead_particle_not_revived():
    pf = _tiny(res="ess")
    pf._recover = lambda L: (pf.xy.copy(), np.ones(4, bool))     # 흩어도 모두 막히는 경우
    pf.predict(0.6, 0.0)
    assert np.isneginf(pf.logw[0])                                # 점수 0 파티클은 그대로 0
    assert (pf.xy[1:, 0] < 9.9).all()                             # 산 파티클은 안 옮김
    assert pf.stats["wall_kills"] == 0


def test_slide_tried_before_recover():
    pf = _tiny(wall="slide")
    pf.logw = np.full(4, -np.log(4))
    pf.xy = np.array([[9.7, 10.0], [9.7, 12.0], [9.7, 8.0], [9.7, 14.0]])
    called = []
    pf._recover = lambda L: called.append(1) or (pf.xy.copy(), np.zeros(4, bool))
    pf.predict(0.6, 0.0)
    assert not called                                             # 미끄러짐으로 살아남아 흩기 안 함
    assert pf.slides > 0
