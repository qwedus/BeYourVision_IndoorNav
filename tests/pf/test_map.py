import json
from pathlib import Path

import numpy as np

from indoornav.pf.map import WallMap

MAP = Path(__file__).resolve().parents[2] / "maps" / "home_loop" / "map.json"


def _route(name):
    m = json.loads(MAP.read_text(encoding="utf-8"))
    return np.array([m["checkpoints"][k] for k in m["routes"][name]])


def test_routes_do_not_cross_walls():
    wm = WallMap.load(MAP)
    for name in ("straight", "loop_ccw"):
        p = _route(name)
        assert not wm.crosses(p[:-1], p[1:]).any(), name


def test_cannot_enter_atrium_or_exit_outward():
    wm = WallMap.load(MAP)
    s = wm.checkpoints["START"]
    p0 = np.array([s, s])
    p1 = np.array([s - [2.0, 0], s + [2.0, 0]])  # 중정 안쪽 / 동쪽 바깥 벽 너머
    assert wm.crosses(p0, p1).all()


def test_straight_length():
    p = _route("straight")
    assert abs(np.linalg.norm(p[1] - p[0]) - 16.8) < 1e-6
