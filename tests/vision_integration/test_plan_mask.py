"""정식 도면에서 만드는 보행 마스크(integration/plan_mask.py) 시험.

앞쪽은 작은 가짜 배열로 규칙을 하나씩 본다(벽이 자름, 이어지지 않은 조각 버림, 넓힘은 벽까지만, 개찰구 줄은 엶).
뒤쪽은 실제 자료로 만든 마스크: PF가 쓰는 GridMap.crosses가 벽 선을 건너는 걸음을 하나도 놓치지 않는지 등.
개찰구 줄 파일(vision/inputs/graph/gates.json)이 아직 복사되지 않았으면 실제 자료 시험은 건너뛴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import plan_mask as pm  # noqa: E402
from indoornav.pf.gridmap import GridMap  # noqa: E402
from station_pkg import enu_to_aligned, load_station  # noqa: E402

MPP = 0.05


def _line(shape, p0, p1):
    return pm.wall_pixels(shape, [np.array([p0, p1], float)])


# ---------------------------------------------------------------- 가짜 배열
def test_wall_cuts_and_other_side_is_dropped():
    walk = np.ones((60, 100), bool)
    wall = _line(walk.shape, (50, 0), (50, 59))                 # 세로 벽 x = 50
    mask, rep = pm.build_mask(walk, wall, [(30, 20)], MPP, thick_m=0.2, grow_m=0.0)
    assert mask[30, 20] and mask[:, :47].all()
    assert not mask[:, 49:52].any()                              # 벽 띠
    assert not mask[:, 53:].any()                                # 건너편은 씨앗과 안 이어져 버림
    assert rep["pieces"] == 2 and rep["kept_pieces"] == 1 and rep["dropped_m2"] > 0
    both, _ = pm.build_mask(walk, wall, [(30, 20), (30, 80)], MPP, thick_m=0.2, grow_m=0.0)
    assert both[:, 53:].all()                                    # 양쪽에 씨앗이 있으면 둘 다 남음


def test_band_is_thick_enough_for_the_step_check():
    walk = np.ones((60, 100), bool)
    wall = _line(walk.shape, (50, 0), (50, 59))
    mask, _ = pm.build_mask(walk, wall, [(30, 20), (30, 80)], MPP, thick_m=0.2, grow_m=0.0)
    width = (~mask[30]).sum() * MPP
    assert 0.2 <= width <= 0.2 + 2 * MPP                         # 선 픽셀 1칸 + 양쪽 thick/2


def test_grow_fills_the_gap_to_the_wall_but_not_beyond():
    walk = np.zeros((60, 100), bool)
    walk[:, :40] = True                                          # 보행칸이 벽(x = 50)보다 0.5 m 모자람
    wall = _line(walk.shape, (50, 0), (50, 59))
    plain, _ = pm.build_mask(walk, wall, [(30, 20)], MPP, thick_m=0.2, grow_m=0.0)
    grown, _ = pm.build_mask(walk, wall, [(30, 20)], MPP, thick_m=0.2, grow_m=0.6)
    assert not plain[30, 45] and grown[30, 45]                   # 틈이 메워짐
    assert not grown[30, 52:].any()                              # 벽 너머로 넘친 0.1 m 띠는 떨어져 나감


def test_closed_outline_becomes_a_hole():
    walk = np.ones((80, 80), bool)
    t = np.linspace(0, 2 * np.pi, 60)
    ring = np.c_[40 + 10 * np.cos(t), 40 + 10 * np.sin(t)]      # 지름 1 m 기둥 테두리
    mask, _ = pm.build_mask(walk, pm.wall_pixels(walk.shape, [ring]), [(5, 5)], MPP, thick_m=0.2, grow_m=0.5)
    assert not mask[40, 40] and mask[5, 5] and mask[40, 70]
    yy, xx = np.mgrid[0:80, 0:80]
    assert not mask[np.hypot(xx - 40, yy - 40) <= 10].any()      # 원 안은 모두 막힘


def test_open_pixels_win_over_the_wall_band():
    walk = np.ones((60, 100), bool)
    wall = _line(walk.shape, (50, 0), (50, 59))
    gate = pm.polygon_pixels(walk.shape, [np.array([(44, 20), (56, 20), (56, 30), (44, 30)], float)])
    mask, rep = pm.build_mask(walk, wall, [(30, 20)], MPP, thick_m=0.2, grow_m=0.0, open_px=gate)
    assert mask[25, 50] and not mask[40, 50]                     # 개찰구 줄 안은 열리고 밖의 벽은 그대로
    assert mask[:, 53:].all()                                    # 열린 줄을 거쳐 건너편이 이어짐
    assert rep["opened_m2"] > 0


def test_seed_on_a_blocked_pixel_uses_the_nearest_free_one():
    walk = np.ones((60, 100), bool)
    wall = _line(walk.shape, (50, 0), (50, 59))
    near, rep = pm.build_mask(walk, wall, [(30, 50)], MPP, thick_m=0.2, grow_m=0.0, seed_reach_m=0.5)
    assert rep["lost_seeds"] == [] and near.any()
    walk2 = walk.copy()
    walk2[:, 60:] = False
    _, rep2 = pm.build_mask(walk2, wall, [(30, 20), (30, 95)], MPP, thick_m=0.2, grow_m=0.0, seed_reach_m=0.5)
    assert rep2["lost_seeds"] == [1]                             # 보행 픽셀에서 0.5 m 넘게 떨어진 씨앗은 알림


def test_gate_row_corners_follow_the_row_axes():
    row = dict(centre=[10.0, 5.0], axis=[0.6, 0.8], normal=[0.8, -0.6], t0=-2.0, t1=3.0, n0=-1.0, n1=1.0)
    c = pm.gate_row_corners(row, reach_m=0.1)
    a, n = np.array(row["axis"]), np.array(row["normal"])
    t, m = (c - row["centre"]) @ a, (c - row["centre"]) @ n
    assert np.allclose(sorted(set(np.round(t, 6))), [-2.0, 3.0])            # 줄 방향으로는 안 넓힘
    assert np.allclose(sorted(set(np.round(m, 6))), [-1.1, 1.1])


# ---------------------------------------------------------------- 실제 자료
@pytest.fixture(scope="module")
def made(tmp_path_factory):
    if not pm.GATES.is_file():
        pytest.skip(f"{pm.GATES.relative_to(ROOT).as_posix()} 없음 (sync_to_branch.py로 복사한 뒤 시험)")
    out = tmp_path_factory.mktemp("plan_mask")
    return out, pm.make(out)


def test_same_grid_as_the_original_mask(made):
    out, rep = made
    for name in paths.FLOOR_NAME.values():
        a, b = GridMap.load(paths.ORIG_META, name), GridMap.load(out / pm.META_NAME, name)
        assert a.walk.shape == b.walk.shape and a.m_per_px == b.m_per_px and np.array_equal(a.offset, b.offset)
        assert rep[name]["lost_nodes"] == [] and rep[name]["kept_pieces"] >= 1
        assert 0.7 * rep[name]["walk_m2"] < rep[name]["mask_m2"] < 1.1 * rep[name]["walk_m2"]


def test_same_bytes_every_time(made, tmp_path):
    out, _ = made
    again = tmp_path / "again"
    pm.make(again)
    for name in paths.FLOOR_NAME.values():
        assert (out / f"{name}_walk_plan.png").read_bytes() == (again / f"{name}_walk_plan.png").read_bytes()


def test_mask_in_the_repository_is_current(made):
    """integration/maps에 둔 마스크가 지금 입력·기본 값으로 만든 것과 같다 (입력이 바뀌면 plan_mask.py를 다시 돌릴 것)"""
    out, _ = made
    for name in paths.FLOOR_NAME.values():
        kept = paths.MASK_DIR / f"{name}_walk_plan.png"
        assert kept.is_file(), "integration/plan_mask.py를 돌려 maps에 마스크를 만들 것"
        assert np.array_equal(GridMap.load(paths.MASK_DIR / pm.META_NAME, name).walk, GridMap.load(out / pm.META_NAME, name).walk)
    kept_meta = json.loads((paths.MASK_DIR / pm.META_NAME).read_text(encoding="utf-8"))["plan_mask"]
    assert (kept_meta["thick_m"], kept_meta["grow_m"]) == (pm.THICK_M, pm.GROW_M)


def test_gate_rows_are_open(made):
    out, _ = made
    gates = json.loads(pm.GATES.read_text(encoding="utf-8"))
    _, smeta = load_station(paths.INPUTS, with_meta=True)
    rot = smeta["aligned_from_enu_rot_deg"]
    n_rows = 0
    for f in gates["floors"]:
        m = GridMap.load(out / pm.META_NAME, paths.FLOOR_NAME[f["floor"]])
        for row in f["rows"]:
            n_rows += 1
            c, a, n = (np.array(row[k]) for k in ("centre", "axis", "normal"))
            ts, ns = np.meshgrid(np.linspace(row["t0"] + 0.05, row["t1"] - 0.05, 60), np.linspace(row["n0"], row["n1"], 30))
            pts = c + ts.reshape(-1, 1) * a + ns.reshape(-1, 1) * n
            assert m.free(enu_to_aligned(pts, rot)).all(), f"{row['key']} 줄 안에 막힌 곳이 있음"
    assert n_rows > 0


def test_no_step_crosses_a_wall_line_unnoticed(made):
    """벽 선을 사이에 둔 두 보행 점을 잇는 걸음은 GridMap.crosses가 모두 잡는다 (개찰구 줄 안은 일부러 연 곳이라 뺌)"""
    out, _ = made
    gates = json.loads(pm.GATES.read_text(encoding="utf-8"))
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    rot = smeta["aligned_from_enu_rot_deg"]
    rng = np.random.default_rng(0)
    for fl, name in paths.FLOOR_NAME.items():
        m = GridMap.load(out / pm.META_NAME, name)
        rows = [r for f in gates["floors"] if f["floor"] == fl for r in f["rows"]]
        with np.load(pm.PLAN_NPZ) as z:
            xy, off = z[f"{name}_walls__xy"], z[f"{name}_walls__off"]
        al = enu_to_aligned(xy, rot)
        last = np.zeros(len(al), bool)
        last[off[1:] - 1] = True                                 # 꺾은선의 마지막 점은 다음 점과 안 이어짐
        i = rng.choice(np.flatnonzero(~last), 40000)
        p, q = al[i], al[i + 1]
        seg = q - p
        length = np.hypot(*seg.T)
        ok = length > 1e-6
        p, seg, length = p[ok], seg[ok], length[ok]
        on = p + seg * rng.uniform(0.05, 0.95, (len(p), 1))      # 벽 선 위 한 점
        nrm = np.c_[-seg[:, 1], seg[:, 0]] / length[:, None]
        tan = seg / length[:, None]
        ang = rng.uniform(-np.radians(70), np.radians(70), len(p))     # 벽에 비스듬히 건너는 걸음까지
        d = nrm * np.cos(ang)[:, None] + tan * np.sin(ang)[:, None]
        a = on - d * rng.uniform(0.15, 0.7, (len(p), 1))
        b = on + d * rng.uniform(0.15, 0.7, (len(p), 1))
        both = m.free(a) & m.free(b)
        in_gate = np.zeros(len(p), bool)
        for row in rows:
            c, ax, nn = (np.array(row[k]) for k in ("centre", "axis", "normal"))
            e = enu_to_aligned(c, rot)[0]
            ax_al, nn_al = enu_to_aligned(ax, rot)[0], enu_to_aligned(nn, rot)[0]
            t, n = (on - e) @ ax_al, (on - e) @ nn_al
            in_gate |= (t > row["t0"] - 0.8) & (t < row["t1"] + 0.8) & (n > row["n0"] - 0.9) & (n < row["n1"] + 0.9)
        use = both & ~in_gate
        assert use.sum() > 200, f"{name}: 시험할 걸음이 너무 적음 ({use.sum()})"
        missed = ~m.crosses(a[use], b[use])
        assert not missed.any(), f"{name}: 벽 선을 건너는 걸음 {missed.sum()}/{use.sum()}개를 못 잡음"


def test_graph_edges_stay_on_the_mask(made):
    """그래프 간선이 막힌 곳을 지나는 길이는 지금 마스크(방 뺀 사본)보다 길지 않다"""
    out, _ = made
    _, smeta = load_station(paths.INPUTS, with_meta=True)
    rot = smeta["aligned_from_enu_rot_deg"]
    graph = json.loads(pm.GRAPH.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}

    def blocked_m(meta):
        tot = 0.0
        for fl, name in paths.FLOOR_NAME.items():
            m = GridMap.load(meta, name)
            for e in graph["edges"]:
                a, b = nodes[e["from"]], nodes[e["to"]]
                if a["floor"] != fl or b["floor"] != fl:
                    continue
                p = enu_to_aligned(np.array([[a["x"], a["y"]], [b["x"], b["y"]]]), rot)
                length = float(np.hypot(*(p[1] - p[0])))
                pts = p[0] + (p[1] - p[0]) * np.linspace(0, 1, max(2, int(length / 0.05) + 1))[:, None]
                tot += length * float(1 - m.free(pts).mean())
        return tot

    assert blocked_m(out / pm.META_NAME) <= blocked_m(paths.MASK_META) + 0.05
