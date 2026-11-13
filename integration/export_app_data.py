"""역 자료를 앱(Swift)이 읽는 형식으로 내보내기: npz·png → JSON + 맨 바이트 배열.

    .venv/Scripts/python integration/export_app_data.py            → integration/out/app_data/dongchun/
    .venv/Scripts/python integration/export_app_data.py --check    # 쓰지 않고, 있는 묶음이 지금 입력과 같은지만 확인

앱은 numpy(npz)·GeoPackage를 못 읽는다. 이 묶음은 JSON(Codable로 읽음)과 맨 바이트 파일(Data를 그대로 배열로 봄)만 쓴다.
입력은 이 브랜치 안 파일뿐이다(옆 저장소 필요 없음). 결과는 저장소에 넣지 않는다(다시 만들면 같은 바이트).

묶음 안
    manifest.json        형식·좌표 약속·파일 목록(종류·모양·sha256)·입력 해시. 앱은 이것부터 읽는다
    station.json         역사 패키지 글자 부분: 층, 카메라 높이, 좌표 변환, 에스컬레이터 끝, 층별 격자 값·랜드마크
    <층>_walk.u8 …       역사 패키지 배열(층별): walk·walk_pf·computed (H, W) uint8 0/1, dist (L, H, W) float16
    <층>_mask.u8         PF 벽 판정용 보행 마스크 (H, W) uint8 0/1 (정식 도면에서 만든 마스크, plan_mask.py), 축척·원점은 manifest의 masks
    <층>_<walls|tactile>_xy.f64 · _off.i32   도면 선: 이어 붙인 좌표 (N, 2)와 도형 시작 위치(도형 수 + 1)
    graph.json · graph_general.json · destinations.json · gates.json   경로 그래프·목적지 묶음·개찰구 몸체 (그대로 복사)
    plan_<층>.png        화면에 띄우는 평면도 그림(글자 없음, 벽·기둥·점자블록·개찰구). 픽셀 ↔ 좌표 변환은 manifest의 ui_plan

바이트 약속: 숫자는 모두 little-endian, 배열은 행 우선(C 순서). (H, W)면 칸 (i, j)는 i*W + j번째, (L, H, W)면 (l*H + i)*W + j번째.
float16의 nan은 그대로 nan(가시성 표: computed가 1인 칸의 nan = 안 보임, 0인 칸의 nan = 모름).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import paths
from plan_mask import META_NAME as PLAN_META_NAME
from station_pkg import ARRAYS, NAME, load_station

FORMAT = "station_app_bundle v1"
OUT = paths.OUT / "app_data"
GRAPH_FILES = ("graph.json", "graph_general.json", "destinations.json", "gates.json")
PLAN_NPZ = paths.MASK_DIR / "plan_lines.npz"
APP_MASK_META = paths.MASK_DIR / PLAN_META_NAME     # 앱에 넣는 보행 마스크: 정식 도면에서 만든 것 (11-13 채택)
UI_PLAN_DIR = paths.INPUTS / "plan"               # 화면용 평면도 그림 plan_<층>.png + 변환 (원본 localization_design)
UI_PLAN_TRANSFORM = "plan_transform.json"
GRID_BIN = {"walk": "u8", "walk_pf": "u8", "computed": "u8", "dist": "f16"}      # 큰 배열만 바이트 파일로, 나머지는 station.json
DTYPE = {"u8": "<u1", "f16": "<f2", "f64": "<f8", "i32": "<i4"}
NAMES = {"u8": "uint8", "f16": "float16", "f64": "float64", "i32": "int32"}
COORDS = {
    "enu": "동·북 m, 메시 원점. 그래프·랜드마크·개찰구·도면 선·에스컬레이터 끝",
    "grid": "역 좌표 st = (ENU - T) @ R. 격자 칸 j = floor((st_x - x0) / cell), i = floor((st_y - y0) / cell), 배열 [i][j]",
    "aligned": "지도 좌표 = Rot(aligned_from_enu_rot_deg) · ENU (원점 그대로). PF 상태와 보행 마스크가 쓰는 좌표",
    "mask_px": "px = (aligned_x - offset_x) / m_per_px, py = -(aligned_y - offset_y) / m_per_px, 반올림해서 배열 [py][px]",
    "compass": "나침반(북 0°, 시계 +) = wrap360(90 + aligned_from_enu_rot_deg - deg(th)), th = aligned 수학각(반시계 +)",
    "ui_plan_px": "화면용 그림 픽셀 [px, py] = en_to_px · [동, 북, 1] (2×3 행렬, 왼쪽 위가 0,0). 반대는 px_to_en. 역 좌표로는 station_to_px",
}


def lf(b):
    return b.replace(b"\r\n", b"\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def as_bytes(a, kind):
    return np.ascontiguousarray(a, dtype=DTYPE[kind]).tobytes()


def build(graph_dir=None, plan_dir=None):
    """→ (파일 이름 → 바이트, manifest dict). 디스크에 쓰지 않는다."""
    graph_dir = Path(graph_dir) if graph_dir else paths.INPUTS / "graph"
    plan_dir = Path(plan_dir) if plan_dir else UI_PLAN_DIR
    files, listing, inputs = {}, {}, {}

    def put(name, data, kind, shape=None, **more):
        files[name] = data
        rec = dict(kind=kind, bytes=len(data), sha256=sha(data))
        if shape is not None:
            rec.update(dtype=NAMES[kind], shape=list(shape))
        rec.update(more)
        listing[name] = rec

    def note_input(p, text=True):
        b = Path(p).read_bytes()
        inputs[Path(p).relative_to(paths.ROOT).as_posix() if Path(p).is_relative_to(paths.ROOT) else Path(p).name] = \
            sha(lf(b) if text else b)
        return b

    # ---------- 역사 패키지
    meta = json.loads(note_input(paths.INPUTS / f"{NAME}.json").decode("utf-8"))
    note_input(paths.INPUTS / f"{NAME}.npz", text=False)
    fmap = load_station(paths.INPUTS)
    station = {k: meta[k] for k in ("format", "floors", "cam_h", "aligned_from_enu_rot_deg", "computed_note", "escalators")}
    station["source"] = {k: meta["source"][k] for k in ("repo", "commit") if k in meta["source"]}
    station["grids"] = {}
    for fl in sorted(fmap, reverse=True):
        fm = fmap[fl]
        name = fm["name"]
        h, w = fm["walk"].shape
        g = dict(floor=int(fl), x0=fm["x0"], y0=fm["y0"], cell=fm["cell"], floor_z=fm["floor_z"], shape=[h, w],
                 R=fm["R"].tolist(), T=fm["T"].tolist(), n_landmarks=len(fm["lm_id"]),
                 lm_id=fm["lm_id"].tolist(), lm_type=fm["lm_type"].tolist(), lm_xy=fm["lm_xy"].tolist(),
                 lm_st=fm["lm_st"].tolist(), lm_dz=fm["lm_dz"].tolist(), landmarks=fm["lm_json"], arrays={})
        for key, kind in GRID_BIN.items():
            fname = f"{name}_{key}.{kind}"
            put(fname, as_bytes(fm[key], kind), kind, fm[key].shape, part="station", floor=name, array=key)
            g["arrays"][key] = fname
        station["grids"][name] = g
    assert set(ARRAYS) == set(GRID_BIN) | {"x0", "y0", "cell", "R", "T", "lm_id", "lm_type", "lm_xy", "lm_st", "lm_dz", "floor_z"}, \
        "역사 패키지 배열 목록이 바뀜 — 이 내보내기도 맞출 것"
    put("station.json", (json.dumps(station, ensure_ascii=False, indent=1) + "\n").encode("utf-8"), "json", part="station")

    # ---------- PF 벽 판정용 보행 마스크 (정식 도면에서 만든 마스크)
    mmeta = json.loads(note_input(APP_MASK_META).decode("utf-8"))
    assert mmeta["aligned_from_enu_rot_deg"] == meta["aligned_from_enu_rot_deg"], "마스크와 역사 패키지의 회전각이 다름"
    masks = {}
    for name, f in mmeta["floors"].items():
        png = APP_MASK_META.parent / f["walk_png"]
        note_input(png, text=False)
        walk = np.array(Image.open(png)) > 127              # indoornav GridMap.load와 같은 문턱
        assert walk.ndim == 2, f"마스크가 흑백 한 장이 아님: {png}"
        fname = f"{name}_mask.u8"
        put(fname, as_bytes(walk, "u8"), "u8", walk.shape, part="mask", floor=name)
        masks[name] = dict(file=fname, shape=list(walk.shape), m_per_px=f["m_per_px"], offset_aligned=f["offset_aligned"],
                           source=f["walk_png"], made={k: mmeta["plan_mask"][k] for k in ("thick_m", "grow_m", "gate_reach_m", "rule")})

    # ---------- 도면 선 (벽·점자블록, ENU)
    lines = {}
    note_input(PLAN_NPZ, text=False)
    with np.load(PLAN_NPZ) as z:
        for layer in sorted({k.split("__")[0] for k in z.files}):
            xy, off, kind = z[f"{layer}__xy"], z[f"{layer}__off"], z[f"{layer}__kind"]
            kinds = sorted({str(k) for k in kind})
            assert len(kinds) == 1 and off[0] == 0 and off[-1] == len(xy) and len(off) == len(kind) + 1, layer
            put(f"{layer}_xy.f64", as_bytes(xy, "f64"), "f64", xy.shape, part="plan_lines", layer=layer)
            put(f"{layer}_off.i32", as_bytes(off, "i32"), "i32", off.shape, part="plan_lines", layer=layer)
            lines[layer] = dict(xy=f"{layer}_xy.f64", off=f"{layer}_off.i32", geometry=kinds[0], n_parts=len(kind),
                                n_points=len(xy))

    # ---------- 경로 그래프·목적지·개찰구 (그대로, 줄 끝만 LF)
    graph = {}
    for fname in GRAPH_FILES:
        b = lf(note_input(graph_dir / fname))
        json.loads(b.decode("utf-8"))
        put(fname, b, "json", part="graph")
        graph[fname.split(".")[0]] = fname
    base = sha(files["graph.json"])
    for fname in ("graph_general.json", "destinations.json", "gates.json"):
        doc = json.loads(files[fname].decode("utf-8"))["metadata"]
        made_from = (doc.get("base") or doc.get("graph"))["sha256"]
        assert made_from == base, f"{fname}이 지금 graph.json에서 만든 것이 아님 (다시 만들어 복사할 것)"

    # ---------- 화면용 평면도 그림 (그대로 복사) + 픽셀 ↔ 좌표 변환
    transform = json.loads(note_input(plan_dir / UI_PLAN_TRANSFORM).decode("utf-8"))
    transform = {k: v for k, v in transform.items() if not k.startswith("_")}       # "_note" 같은 설명 칸은 층이 아님
    assert set(transform) == set(meta["floors"]), f"평면도 그림의 층 {sorted(transform)} ≠ 역사 패키지 층 {sorted(meta['floors'])}"
    ui_plan = {}
    for name in sorted(transform):
        fname = f"plan_{name}.png"
        b = note_input(plan_dir / fname, text=False)
        t = transform[name]
        with Image.open(plan_dir / fname) as im:
            assert im.size == (t["width"], t["height"]), f"{fname} 크기 {im.size}가 변환 파일과 다름"
        put(fname, b, "png", part="ui_plan", floor=name)
        ui_plan[name] = dict(file=fname, **t)

    station_name = json.loads(files["graph.json"].decode("utf-8"))["metadata"]["name"]      # 묶음 폴더 이름
    manifest = dict(format=FORMAT, station=station_name, byte_order="little", array_order="row-major (C)",
                    coords=COORDS, aligned_from_enu_rot_deg=meta["aligned_from_enu_rot_deg"], floors=meta["floors"],
                    masks=masks, plan_lines=lines, graph=graph, ui_plan=ui_plan, files=dict(sorted(listing.items())),
                    inputs=dict(sorted(inputs.items())))
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    return files, manifest


# ---------------------------------------------------------------- 읽기 (앱이 할 일을 Python으로 — 검산·참고 구현)
def read_array(folder, name, rec):
    b = (Path(folder) / name).read_bytes()
    if sha(b) != rec["sha256"]:
        raise ValueError(f"{name}: 해시가 manifest와 다름")
    kind = name.rsplit(".", 1)[1]
    return np.frombuffer(b, dtype=DTYPE[kind]).reshape(rec["shape"])


def load_app_data(folder):
    """묶음 → dict(manifest, station, fmap, masks, lines). fmap은 station_pkg.load_station과 같은 필드."""
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if man["format"] != FORMAT:
        raise ValueError(f"모르는 형식: {man['format']}")
    files = man["files"]
    station = json.loads((folder / "station.json").read_text(encoding="utf-8"))
    fmap = {}
    for name, g in station["grids"].items():
        fm = dict(x0=g["x0"], y0=g["y0"], cell=g["cell"], floor_z=g["floor_z"], R=np.array(g["R"]), T=np.array(g["T"]),
                  lm_id=np.array(g["lm_id"]), lm_type=np.array(g["lm_type"]), lm_xy=np.array(g["lm_xy"], float).reshape(-1, 2),
                  lm_st=np.array(g["lm_st"], float).reshape(-1, 2), lm_dz=np.array(g["lm_dz"], float),
                  lm_json=g["landmarks"], name=name)
        for key, fname in g["arrays"].items():
            a = read_array(folder, fname, files[fname])
            fm[key] = a.astype(bool) if key != "dist" else a
        fmap[g["floor"]] = fm
    masks = {name: dict(walk=read_array(folder, m["file"], files[m["file"]]).astype(bool), m_per_px=m["m_per_px"],
                        offset=np.array(m["offset_aligned"], float)) for name, m in man["masks"].items()}
    lines = {}
    for layer, rec in man["plan_lines"].items():
        xy = read_array(folder, rec["xy"], files[rec["xy"]])
        off = read_array(folder, rec["off"], files[rec["off"]])
        lines[layer] = [xy[off[i]:off[i + 1]] for i in range(rec["n_parts"])]
    return dict(manifest=man, station=station, fmap=fmap, masks=masks, lines=lines)


def verify(folder):
    """묶음을 다시 읽어 원본(npz·png)과 값이 똑같은지 확인 → 문제 목록 (없으면 [])"""
    from indoornav.pf.gridmap import GridMap

    bad = []
    got = load_app_data(folder)
    ref, meta = load_station(paths.INPUTS, with_meta=True)
    if set(got["fmap"]) != set(ref):
        bad.append(f"층이 다름: {sorted(got['fmap'])} ≠ {sorted(ref)}")
    for fl in set(got["fmap"]) & set(ref):
        a, b = got["fmap"][fl], ref[fl]
        for k in sorted(set(b) | set(a)):
            if k not in a or k not in b:
                bad.append(f"{fl} {k}: 한쪽에만 있음")
            elif isinstance(b[k], np.ndarray):
                same = a[k].shape == b[k].shape and (np.array_equal(a[k], b[k], equal_nan=True) if b[k].dtype.kind == "f"
                                                      else np.array_equal(a[k], b[k]))
                if not same or (k == "dist" and a[k].dtype != b[k].dtype):
                    bad.append(f"{fl} {k}: 값이 다름")
            elif a[k] != b[k]:
                bad.append(f"{fl} {k}: {a[k]!r} ≠ {b[k]!r}")
    for k in ("cam_h", "aligned_from_enu_rot_deg", "escalators", "floors"):
        if got["station"][k] != meta[k]:
            bad.append(f"station.json {k}가 원본과 다름")
    for name, m in got["masks"].items():
        g = GridMap.load(APP_MASK_META, name)
        if not (np.array_equal(m["walk"], g.walk) and m["m_per_px"] == g.m_per_px and np.array_equal(m["offset"], g.offset)):
            bad.append(f"보행 마스크 {name}: GridMap.load와 다름")
    with np.load(PLAN_NPZ) as z:
        for layer, parts in got["lines"].items():
            xy, off = z[f"{layer}__xy"], z[f"{layer}__off"]
            if len(parts) != len(off) - 1 or any(not np.array_equal(p, xy[off[i]:off[i + 1]]) for i, p in enumerate(parts)):
                bad.append(f"도면 선 {layer}: 원본과 다름")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT), help="묶음을 둘 상위 폴더 (그 아래 <역 이름>/)")
    ap.add_argument("--graph-dir", help="graph.json·destinations.json·gates.json이 있는 폴더 (기본 vision/inputs/graph)")
    ap.add_argument("--plan-dir", help="plan_<층>.png·plan_transform.json이 있는 폴더 (기본 vision/inputs/plan)")
    ap.add_argument("--check", action="store_true", help="쓰지 않고, 있는 묶음이 지금 입력으로 만든 것과 같은지 확인")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    files, man = build(a.graph_dir, a.plan_dir)
    folder = Path(a.out) / man["station"]
    if a.check:
        stale = [n for n, b in files.items() if not (folder / n).is_file() or (folder / n).read_bytes() != b]
        extra = sorted(p.name for p in folder.glob("*") if p.name not in files) if folder.is_dir() else []
        if stale or extra:
            sys.exit(f"묶음이 지금 입력과 다름 — 다시 만들 것. 다른 파일 {stale}, 남는 파일 {extra}")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*"):
            if old.is_file() and old.name not in files:
                old.unlink()
        for n, b in files.items():
            (folder / n).write_bytes(b)
    bad = verify(folder)
    for b in bad:
        print("✗", b)
    if bad:
        sys.exit(f"검산 실패 {len(bad)}건")
    total = sum(len(b) for b in files.values())
    print(f"{'확인' if a.check else '씀'}: {folder} — 파일 {len(files)}개, {total / 1e6:.1f} MB, 원본과 값 같음")


if __name__ == "__main__":
    main()
