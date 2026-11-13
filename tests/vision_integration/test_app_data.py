"""앱용 역 자료 묶음(integration/export_app_data.py) 시험.

묶음을 임시 폴더에 만들고, (1) 다시 읽은 값이 원본(npz·png)과 같은지, (2) 문서에 적은 바이트 약속(순서·칸 번호 식)대로
numpy 없이 읽어도 같은 값이 나오는지, (3) 그래프·목적지·개찰구 파일이 서로 맞는지 본다.
목적지·개찰구 파일이 아직 브랜치에 복사되지 않았으면 건너뛴다.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import export_app_data as ex  # noqa: E402
from indoornav.pf.gridmap import GridMap  # noqa: E402
from station_pkg import enu_to_aligned, load_station  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    missing = [f for f in ex.GRAPH_FILES if not (paths.INPUTS / "graph" / f).is_file()]
    missing += [] if (ex.UI_PLAN_DIR / ex.UI_PLAN_TRANSFORM).is_file() else [ex.UI_PLAN_TRANSFORM]
    if missing:
        pytest.skip(f"vision/inputs에 {missing} 없음 (sync_to_branch.py로 복사한 뒤 시험)")
    files, man = ex.build()
    folder = tmp_path_factory.mktemp("app_data") / man["station"]
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder, files, man


def test_same_bytes_every_time(bundle):
    _, files, _ = bundle
    again, _ = ex.build()
    assert again.keys() == files.keys()
    assert all(again[k] == files[k] for k in files)


def test_reads_back_equal_to_sources(bundle):
    folder, _, _ = bundle
    assert ex.verify(folder) == []


def test_manifest_lists_every_file(bundle):
    folder, files, man = bundle
    assert set(man["files"]) == set(files) - {"manifest.json"}
    for name, rec in man["files"].items():
        assert rec["bytes"] == len(files[name]) and rec["sha256"] == ex.sha(files[name])
        if "shape" in rec:
            assert rec["bytes"] == int(np.prod(rec["shape"])) * np.dtype(ex.DTYPE[rec["kind"]]).itemsize
    assert json.loads((folder / "manifest.json").read_text(encoding="utf-8")) == man


def test_plain_byte_reads_follow_the_written_rules(bundle):
    """numpy 없이 struct로: little-endian, 행 우선, (i, j) → i*W + j, (l, i, j) → (l*H + i)*W + j"""
    folder, files, man = bundle
    ref = load_station(paths.INPUTS)
    station = json.loads(files["station.json"].decode("utf-8"))
    rng = np.random.default_rng(0)
    for name, g in station["grids"].items():
        fm = ref[g["floor"]]
        h, w = g["shape"]
        n_lm = g["n_landmarks"]
        walk, dist = files[g["arrays"]["walk"]], files[g["arrays"]["dist"]]
        assert len(walk) == h * w and len(dist) == n_lm * h * w * 2
        seen = 0
        cells = np.argwhere(np.isfinite(fm["dist"]))            # 값이 있는 칸 위주로 + 아무 칸
        picks = [tuple(c) for c in cells[rng.choice(len(cells), 200)]] + \
                [(rng.integers(n_lm), rng.integers(h), rng.integers(w)) for _ in range(200)]
        for l, i, j in picks:
            assert walk[i * w + j] == int(fm["walk"][i, j])
            (v,) = struct.unpack_from("<e", dist, ((l * h + i) * w + j) * 2)
            want = float(fm["dist"][l, i, j])
            assert (np.isnan(v) and np.isnan(want)) or v == want
            seen += np.isfinite(want)
        assert seen >= 200
    for layer, rec in man["plan_lines"].items():
        off = struct.unpack(f"<{rec['n_parts'] + 1}i", files[rec["off"]])
        assert off[0] == 0 and off[-1] == rec["n_points"] and len(files[rec["xy"]]) == rec["n_points"] * 16
        assert all(b > a for a, b in zip(off, off[1:]))


def test_mask_lookup_matches_gridmap(bundle):
    """manifest에 적은 식으로 지도 좌표 → 마스크 칸을 찾으면 PF가 쓰는 GridMap.free와 같다"""
    _, files, man = bundle
    rng = np.random.default_rng(1)
    for name, m in man["masks"].items():
        g = GridMap.load(ex.APP_MASK_META, name)
        h, w = m["shape"]
        lo, hi = g.to_world(-20, h + 20), g.to_world(w + 20, -20)        # 그림 밖 조금까지
        pts = rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi), size=(5000, 2))
        want = g.free(pts)
        data = files[m["file"]]
        got = []
        for x, y in pts:
            px = int(np.round((x - m["offset_aligned"][0]) / m["m_per_px"]))
            py = int(np.round(-(y - m["offset_aligned"][1]) / m["m_per_px"]))
            got.append(0 <= px < w and 0 <= py < h and data[py * w + px] == 1)
        assert np.array_equal(np.array(got), want)
        assert 0 < want.mean() < 1


def test_graph_files_belong_together(bundle):
    _, files, man = bundle
    doc = {k: json.loads(files[v].decode("utf-8")) for k, v in man["graph"].items()}
    nodes = {n["id"]: n for n in doc["graph"]["nodes"]}
    plan_floors = {f["floor"] for f in doc["destinations"]["metadata"]["floors"] if f["has_plan"]}
    assert plan_floors == set(man["floors"].values())                     # 도면 있는 층 = 역사 패키지 층
    n_members = 0
    for group in doc["destinations"]["groups"]:
        for m in group["members"]:
            n_members += 1
            for rec in (m, m.get("plan_node")):
                if rec is not None:
                    n = nodes[rec["node"]]
                    assert (n.get("key"), n["floor"], n["x"], n["y"]) == (rec["key"], rec["floor"], rec["x"], rec["y"])
            assert ("plan_node" in m) == (m["floor"] not in plan_floors)
            if "plan_node" in m:
                assert m["plan_node"]["floor"] in plan_floors
    assert n_members > 0
    gate_nodes = {n["id"] for n in nodes.values() if n["type"] == "gate"}
    rows = [r for f in doc["gates"]["floors"] for r in f["rows"]]
    assert sorted(r["node"] for r in rows) == sorted(gate_nodes)          # 개찰구 노드마다 줄 하나, 그 밖의 줄은 없음
    assert all(len(ring) >= 4 and ring[0] == ring[-1] for r in rows for ring in r["bodies"])


def test_ui_plan_pixels_match_station_coordinates(bundle):
    """화면용 그림: 크기가 변환 파일과 같고, 동·북 → 픽셀이 역 좌표 → 픽셀과 같은 곳을 가리키며, 그래프 노드가 그림 안에 있다"""
    import io

    from PIL import Image

    folder, files, man = bundle
    got = ex.load_app_data(folder)
    graph = json.loads(files["graph.json"].decode("utf-8"))
    assert set(man["ui_plan"]) == set(man["floors"])
    for name, u in man["ui_plan"].items():
        with Image.open(io.BytesIO(files[u["file"]])) as im:
            assert im.size == (u["width"], u["height"])
        fm = got["fmap"][man["floors"][name]]
        xy = np.array([[n["x"], n["y"]] for n in graph["nodes"] if n["floor"] == man["floors"][name]])
        assert len(xy) > 0
        ones = np.ones((len(xy), 1))
        px = np.c_[xy, ones] @ np.array(u["en_to_px"]).T
        st = (xy - fm["T"]) @ fm["R"]
        px_st = np.c_[st, ones] @ np.array(u["station_to_px"]).T
        assert np.abs(px - px_st).max() < 0.2                             # 변환 값이 소수 6자리로 적혀 있어 생기는 차이만
        assert (px >= 0).all() and (px[:, 0] < u["width"]).all() and (px[:, 1] < u["height"]).all()
        back = np.c_[px, ones] @ np.array(u["px_to_en"]).T
        assert np.abs(back - xy).max() < 0.02                             # 왕복 2 cm 안
        assert np.isclose(np.hypot(*np.array(u["en_to_px"])[:, 0]), u["px_per_m"], atol=1e-3)


def test_destinations_stand_on_the_walk_grid(bundle):
    """출발·도착 위치로 쓰는 노드(도면 없는 층이면 plan_node)는 넓힌 보행 격자 위에 있다"""
    folder, files, man = bundle
    got = ex.load_app_data(folder)
    dest = json.loads(files["destinations.json"].decode("utf-8"))
    for group in dest["groups"]:
        for m in group["members"]:
            p = m.get("plan_node", m)
            fm = got["fmap"][p["floor"]]
            st = (np.array([p["x"], p["y"]]) - fm["T"]) @ fm["R"]
            j, i = int(np.floor((st[0] - fm["x0"]) / fm["cell"])), int(np.floor((st[1] - fm["y0"]) / fm["cell"]))
            assert fm["walk_pf"][i, j], f"{p['key']}가 보행 격자 밖"
    # 지도 좌표 변환이 station_pkg와 같은 약속인지 (에스컬레이터 끝: 패키지에 ENU와 aligned가 같이 있음)
    for e in got["station"]["escalators"]:
        for end in ("top", "bottom"):
            al = enu_to_aligned(e[end], man["aligned_from_enu_rot_deg"])[0]
            assert np.allclose(al, e[f"{end}_aligned"], atol=1e-3)


def test_changed_file_is_refused(bundle, tmp_path):
    folder, files, man = bundle
    name = next(n for n in files if n.endswith("_walk.u8"))
    data = bytearray(files[name])
    data[0] ^= 1
    (tmp_path / name).write_bytes(bytes(data))
    with pytest.raises(ValueError, match="해시"):
        ex.read_array(tmp_path, name, man["files"][name])
