"""PF 버그 잡기용 합성 데이터 테스트 (정답을 완벽히 아는 조건)."""

from pathlib import Path

import numpy as np

from indoornav.pf.filter import PFParams, ParticleFilter
from indoornav.pf.map import WallMap

MAP = Path(__file__).resolve().parents[2] / "maps" / "home_loop" / "map.json"


def synth_loop(wm: WallMap, step=0.6):
    """START → NE → NW → SW → SE → START 를 정확한 보폭·헤딩으로 걷는 걸음 목록."""
    cp = wm.checkpoints
    pts = [cp[k] for k in ("START", "NE", "NW", "SW", "SE", "START")]
    steps, th_prev = [], np.pi / 2
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        th = np.arctan2(d[1], d[0])
        n = max(1, int(round(np.linalg.norm(d) / step)))
        L = np.linalg.norm(d) / n
        for i in range(n):
            dh = np.angle(np.exp(1j * (th - th_prev))) if i == 0 else 0.0
            steps.append((L, dh))
            th_prev = th
    return steps


def _run(params, steps, seed=0, scale=1.0):
    wm = WallMap.load(MAP)
    pf = ParticleFilter(params, wm, seed=seed)
    pf.init(wm.checkpoints["START"], np.pi / 2)
    for L, dh in steps:
        pf.predict(L * scale, dh)
    return pf, wm


def test_noise_free_loop_returns_to_start():
    wm = WallMap.load(MAP)
    p = PFParams(n=300, sigma_len=0.0, sigma_h0=0.0, sigma_h1=0.0, sigma_k=0.0,
                 k0_lo=1.0, k0_hi=1.0, init_pos_sigma=0.0, init_h_sigma=0.0)
    pf, _ = _run(p, synth_loop(wm))
    assert np.linalg.norm(pf.estimate()["xy"] - wm.checkpoints["START"]) < 0.05


def test_map_corrects_stride_bias():
    """보폭이 10% 길게 나와도 지도(모서리) 때문에 k가 0.9 근처로 수렴하고 위치가 맞는다."""
    wm = WallMap.load(MAP)
    pf, _ = _run(PFParams(n=1000), synth_loop(wm), scale=1.10)
    est = pf.estimate()
    assert np.linalg.norm(est["xy"] - wm.checkpoints["START"]) < 1.0
    assert abs(est["k"] - 1 / 1.10) < 0.07
    assert pf.stats["all_dead"] == 0


def test_no_particle_inside_walls():
    wm = WallMap.load(MAP)
    pf, _ = _run(PFParams(n=500), synth_loop(wm), scale=1.05)
    # 중정(벽 안쪽)에 들어간 파티클이 없어야 함
    ix, iy = 27 * 0.6 / 2, 28 * 0.6 / 2
    inside = (np.abs(pf.xy[:, 0]) < ix) & (np.abs(pf.xy[:, 1]) < iy)
    assert not inside.any()


def test_seed_is_deterministic():
    wm = WallMap.load(MAP)
    a, _ = _run(PFParams(n=200), synth_loop(wm), seed=7)
    b, _ = _run(PFParams(n=200), synth_loop(wm), seed=7)
    assert np.array_equal(a.xy, b.xy)


def test_nonmoving_steps_widen_instead_of_freezing():
    """PDR 이 보폭 0 을 준 걸음이 이어지면 PF 는 멈춰 있지 않고 불확실하게 퍼지며, 'PDR 신뢰 낮음'을 알린다."""
    wm = WallMap.load(MAP)
    pf = ParticleFilter(PFParams(n=500), wm, seed=0)
    pf.init(wm.checkpoints["START"], np.pi / 2)
    r0 = pf.estimate()["unc95"]
    for _ in range(20):
        pf.predict(0.0, 0.0, moving=False)
    est = pf.estimate()
    assert est["xy"][1] > wm.checkpoints["START"][1] + 1.0   # 북쪽으로 일부 이동
    assert est["unc95"] > r0
    assert est["pdr_suspect"]


def test_nonmoving_off_keeps_old_behavior():
    wm = WallMap.load(MAP)
    pf = ParticleFilter(PFParams(n=200, nonmoving_p=0.0), wm, seed=0)
    pf.init(wm.checkpoints["START"], np.pi / 2)
    xy0 = pf.xy.copy()
    for _ in range(5):
        pf.predict(0.0, 0.0, moving=False)
    assert np.allclose(pf.xy, xy0)
