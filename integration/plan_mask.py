"""정식 도면에서 만드는 보행 마스크: 보행 격자(0.5 m) + 벽 단면선. 평면도 그림(글자·지시선 있음)은 쓰지 않는다.

    .venv/Scripts/python integration/plan_mask.py                         → integration/maps/B0x_walk_plan.png + map_meta_plan.json
    .venv/Scripts/python integration/plan_mask.py --thick 0.2 --grow 0 --out <폴더>   # 비교용 변형 (기본 폴더는 안 건드림)

재료(모두 이 브랜치 안): 역사 패키지 보행칸 walk(메시 바닥에서 만든 0.5 m 격자, 방·기둥 자리는 이미 빠져 있음),
도면 벽 단면선 maps/plan_lines.npz(바닥 + 1.2 m에서 자른 선: 벽·기둥·키 큰 구조물), 경로 그래프 노드(이어진 영역 고르기).
픽셀 격자(축척·원점·크기)는 maps/dongchun/map_meta.json 그대로 → indoornav GridMap.load로 읽고 PF 코드·설정은 안 바뀐다.

만드는 순서 (층마다)
    1. 보행칸을 픽셀에 칠함. --grow(m)만큼 넓힘: 0.5 m 칸은 벽 선보다 반 칸쯤 모자라거나 넘치므로, 넉넉히 넓힌 뒤 벽 선으로 자른다
    2. 벽 선에서 --thick/2 안의 픽셀을 막음. PF는 한 걸음을 0.08 m 간격으로 검사하므로 띠가 그보다 충분히 두꺼워야 건너뛰지 않는다
    3. 개찰구 줄(첫 몸체 바깥 끝 ~ 마지막 몸체 바깥 끝, 몸체 길이만큼)을 통째로 엶. 보행칸은 몸체 자리가 빠져 있고 벽 선도 일부 지나가서,
       열지 않으면 몸체의 17~34%가 막힌다. 개찰구는 막지 않기로 함(칸 폭 0.4~1.0 m라 위치 오차가 있으면 파티클이 못 지나감, 11-13 결정)
    4. 그래프 노드와 이어진 영역만 남김(4방향 이어짐). 벽 밖으로 넘친 띠·승강장 안전문 바깥 띠가 떨어져 나가고 기둥은 원 모양이 됨
1.2 m보다 낮은 것(개찰구 울타리·벤치·낮은 난간)은 벽 선에 없어서 막히지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

import paths
from indoornav.pf.gridmap import GridMap
from station_pkg import enu_to_aligned, load_station
from walk_mask import station_walk_on_grid

THICK_M = 0.3       # 벽 띠 두께 (11-13 채택: exp_plan_mask에서 0.2 m보다 나음). (두께 - 0.08)/2 가 픽셀 반대각(B02 0.052 m)보다
                    # 커야 0.08 m 간격 검사가 띠를 건너뛰지 않으므로 0.2 m 아래로는 내리지 말 것
GROW_M = 0.5        # 보행칸 넓힘(한 칸). 0이면 보행칸 그대로
SEED_REACH_M = 0.5  # 노드 픽셀이 막혀 있으면 이 안의 가장 가까운 보행 픽셀을 씨앗으로
PLAN_NPZ = paths.MASK_DIR / "plan_lines.npz"
GATE_REACH_M = 0.1  # 개찰구 줄을 열 때 몸체 길이 방향 양 끝으로 더 여는 길이 (줄 방향으로는 안 넓힘: 줄 끝은 울타리가 붙는 곳)
GRAPH = paths.INPUTS / "graph" / "graph.json"
GATES = paths.INPUTS / "graph" / "gates.json"     # capstone_graph app_data가 만든 개찰구 줄(그래프 개찰구 노드가 있는 줄만)
META_NAME = "map_meta_plan.json"


def wall_pixels(shape, parts_px):
    """벽 선(픽셀 좌표 (n, 2) 꺾은선 목록) → 선이 지나는 픽셀 (H, W) bool, 1픽셀 굵기"""
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for p in parts_px:
        if len(p) >= 2:
            d.line([(float(x), float(y)) for x, y in p], fill=255, width=1)
    return np.array(im) > 0


def polygon_pixels(shape, polys_px):
    """다각형(픽셀 좌표 (n, 2)) 목록 → 안쪽 픽셀 (H, W) bool"""
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for p in polys_px:
        d.polygon([(float(x), float(y)) for x, y in p], fill=255)
    return np.array(im) > 0


def gate_row_corners(row, reach_m=GATE_REACH_M):
    """gates.json 줄 하나 → 줄 사각형 네 꼭짓점 (4, 2) 동·북. 줄 위 점 = centre + t·axis + n·normal"""
    c, a, n = (np.array(row[k], float) for k in ("centre", "axis", "normal"))
    return np.array([c + t * a + m * n for t, m in ((row["t0"], row["n0"] - reach_m), (row["t1"], row["n0"] - reach_m),
                                                    (row["t1"], row["n1"] + reach_m), (row["t0"], row["n1"] + reach_m))])


def build_mask(walk_px, wall_px, seeds_px, m_per_px, thick_m=THICK_M, grow_m=GROW_M, seed_reach_m=SEED_REACH_M, open_px=None):
    """보행칸 픽셀·벽 선 픽셀·씨앗 [(py, px)] → (마스크 (H, W) bool, 보고 dict).

    씨앗은 정수 픽셀 (행, 열). 씨앗과 이어지지 않은 조각은 버린다. 씨앗이 보행 픽셀을 못 찾으면 보고의 lost_seeds에 번호를 적는다.
    open_px: 벽 띠·보행칸과 상관없이 여는 픽셀(개찰구 줄)."""
    base = walk_px if grow_m <= 0 else distance_transform_edt(~walk_px) * m_per_px <= grow_m
    band = (distance_transform_edt(~wall_px) * m_per_px <= thick_m / 2) if wall_px.any() else np.zeros_like(wall_px)
    free = base & ~band
    if open_px is not None:
        free = free | open_px
    lab, n = label(free)                                       # 4방향 이어짐
    dist, (iy, ix) = distance_transform_edt(~free, return_indices=True)
    keep, lost = set(), []
    for k, (py, px) in enumerate(seeds_px):
        inside = 0 <= py < free.shape[0] and 0 <= px < free.shape[1]
        if not inside or dist[py, px] * m_per_px > seed_reach_m:
            lost.append(k)
            continue
        keep.add(int(lab[iy[py, px], ix[py, px]]))
    mask = np.isin(lab, sorted(keep)) if keep else np.zeros_like(free)
    a = m_per_px ** 2
    rep = dict(walk_m2=round(float(walk_px.sum() * a), 1), base_m2=round(float(base.sum() * a), 1),
               band_in_base_m2=round(float((base & band).sum() * a), 1), free_m2=round(float(free.sum() * a), 1),
               mask_m2=round(float(mask.sum() * a), 1), dropped_m2=round(float((free & ~mask).sum() * a), 1),
               opened_m2=round(float((open_px & ~(base & ~band)).sum() * a), 1) if open_px is not None else 0.0,
               pieces=int(n), kept_pieces=len(keep), lost_seeds=lost)
    return mask, rep


def floor_inputs(name, fl, fmap, rot, graph, gates=None):
    """층 하나의 (GridMap 원본 격자, 보행칸 픽셀, 벽 선 픽셀, 씨앗 픽셀, 씨앗 노드 id, 개찰구 줄 픽셀)"""
    g = GridMap.load(paths.ORIG_META, name)
    walk_px = station_walk_on_grid(g, fmap[fl], rot)
    with np.load(PLAN_NPZ) as z:
        xy, off = z[f"{name}_walls__xy"], z[f"{name}_walls__off"]
    al = enu_to_aligned(xy, rot)
    px, py = g.to_px(al)
    pts = np.c_[px, py]
    wall_px = wall_pixels(g.walk.shape, [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)])
    nodes = [n for n in graph["nodes"] if n["floor"] == fl]
    npx, npy = g.to_px(enu_to_aligned(np.array([[n["x"], n["y"]] for n in nodes]), rot))
    seeds = [(int(round(y)), int(round(x))) for x, y in zip(npx, npy)]
    rows = [r for f in (gates or {}).get("floors", []) if f["floor"] == fl for r in f["rows"]]
    open_px = polygon_pixels(g.walk.shape, [np.c_[g.to_px(enu_to_aligned(gate_row_corners(r), rot))] for r in rows])
    return g, walk_px, wall_px, seeds, [n["id"] for n in nodes], open_px, len(rows)


def make(out_dir, thick_m=THICK_M, grow_m=GROW_M, gates_path=GATES):
    """마스크 PNG와 map_meta 사본을 out_dir에 씀 → 보고 dict"""
    gates_path = Path(gates_path)
    if not gates_path.is_file():
        raise SystemExit(f"개찰구 줄 파일 없음: {gates_path} (sync_to_branch.py로 복사하거나 --gates로 지정)")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    meta = json.loads(paths.ORIG_META.read_text(encoding="utf-8"))
    rot = meta["aligned_from_enu_rot_deg"]
    assert rot == smeta["aligned_from_enu_rot_deg"], (rot, smeta["aligned_from_enu_rot_deg"])
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    report = {}
    inside = out_dir.resolve().is_relative_to(paths.ROOT)       # 저장소 밖 폴더면 원본 그림을 절대 경로로 가리킴
    rel = lambda p: (Path(*([".."] * len(out_dir.resolve().relative_to(paths.ROOT).parts)), p) if inside else paths.ROOT / p).as_posix()
    for fl, name in paths.FLOOR_NAME.items():
        g, walk_px, wall_px, seeds, ids, open_px, n_rows = floor_inputs(name, fl, fmap, rot, graph, gates)
        mask, rep = build_mask(walk_px, wall_px, seeds, g.m_per_px, thick_m, grow_m, open_px=open_px)
        rep["gate_rows"] = n_rows
        rep["lost_nodes"] = [ids[k] for k in rep.pop("lost_seeds")]
        if rep["lost_nodes"]:
            raise SystemExit(f"{name}: 그래프 노드 {rep['lost_nodes']} 가까이에 보행 픽셀이 없음 (보행칸·벽 선·그래프가 안 맞음)")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{name}_walk_plan.png")
        meta["floors"][name]["walk_png"] = f"{name}_walk_plan.png"
        meta["floors"][name]["drawn_path_png"] = rel(f"maps/dongchun/{Path(meta['floors'][name]['drawn_path_png']).name}")
        report[name] = rep
    meta["note"] = (f"integration/plan_mask.py가 만든 보행 마스크: 역사 패키지 보행칸(0.5 m)을 {grow_m} m 넓히고 도면 벽 단면선에서 "
                    f"{thick_m / 2} m 안을 막고 개찰구 줄을 연 뒤, 경로 그래프 노드와 이어진 영역만 남김. 좌표·축척은 maps/dongchun/map_meta.json과 같음")
    meta["plan_mask"] = dict(thick_m=thick_m, grow_m=grow_m, seed_reach_m=SEED_REACH_M, gate_reach_m=GATE_REACH_M,
                             rule="(grow(station walk) & ~band(wall lines)) | gate rows, pieces connected to graph nodes", report=report)
    (out_dir / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thick", type=float, default=THICK_M, help="벽 띠 두께 m")
    ap.add_argument("--grow", type=float, default=GROW_M, help="보행칸 넓힘 m (0 = 그대로)")
    ap.add_argument("--out", default=str(paths.MASK_DIR), help="결과 폴더 (기본 integration/maps)")
    ap.add_argument("--gates", default=str(GATES), help="개찰구 줄 파일 gates.json (기본 vision/inputs/graph)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(make(a.out, a.thick, a.grow, a.gates), ensure_ascii=False))


if __name__ == "__main__":
    main()
