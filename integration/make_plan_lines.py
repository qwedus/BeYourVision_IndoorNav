"""정식 도면(plan.gpkg)의 층별 벽선·점자블록 도형을 numpy 파일로 옮김(한 번만 돌림, geopandas 있는 파이썬).

    <geopandas 있는 파이썬> integration/make_plan_lines.py --plan <plan.gpkg>
→ integration/maps/plan_lines.npz, plan_lines.json(원본 출처)

그림 스크립트(fig_plan.py, fig_plan_rt.py)는 이 파일을 읽어서 geopandas·도면 원본 없이 돎(읽기: plan_lines.load_lines).
도형 푸는 방식은 fig_plan.lines_st와 같음: 멀티 도형은 부분마다, Polygon은 exterior, 나머지는 coords, 앞 2열.
좌표는 변환 전 원래 도면 좌표(ENU) 그대로 저장. 레이어마다 `<레이어>__xy`(부분 좌표를 이어 붙인 (N, 2)),
`<레이어>__off`(부분 시작 위치, 길이 부분 수 + 1), `<레이어>__kind`(부분 도형 종류 문자열).
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio

from plan_lines import NPZ, SRC_JSON

LAYER_RE = re.compile(r"^B\d+_(walls|tactile)$")


def parts(path, layer):
    out = []
    for g in gpd.read_file(path, layer=layer).geometry:
        for part in getattr(g, "geoms", [g]):
            xy = np.asarray(part.exterior.coords if part.geom_type == "Polygon" else part.coords)[:, :2]
            out.append((part.geom_type, np.ascontiguousarray(xy, dtype=np.float64)))
    return out


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True, help="원본 plan.gpkg 경로")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    plan = Path(a.plan).resolve()
    layers = sorted(str(n) for n, _ in pyogrio.list_layers(plan) if LAYER_RE.match(str(n)))
    if not layers:
        raise SystemExit(f"*_walls·*_tactile 레이어 없음: {plan.name}")
    arrs, info = {}, {}
    for layer in layers:
        ps = parts(plan, layer)
        off = np.cumsum([0] + [len(xy) for _, xy in ps]).astype(np.int64)
        arrs[f"{layer}__xy"] = np.concatenate([xy for _, xy in ps]) if ps else np.zeros((0, 2))
        arrs[f"{layer}__off"] = off
        arrs[f"{layer}__kind"] = np.array([k for k, _ in ps], dtype="U16")
        info[layer] = dict(n_parts=len(ps), n_points=int(off[-1]), kinds=sorted(set(k for k, _ in ps)))
    d = plan.parent
    top = Path(git(d, "rev-parse", "--show-toplevel"))      # 출처(git 저장소 안 도면)를 먼저 확인해 실패하면 아무것도 안 씀
    rel = plan.relative_to(top.resolve()).as_posix()
    src = dict(
        note="make_plan_lines.py가 정식 도면에서 층별 벽선·점자블록만 옮긴 것. 좌표는 변환 전 원래 도면 좌표(ENU), 그림 바탕용",
        repo=top.name,
        path=rel,
        repo_head=git(top, "rev-parse", "HEAD"),
        file_commit=git(top, "log", "-1", "--format=%H", "--", rel),
        file_dirty=bool(git(top, "status", "--porcelain", "--", rel)),
        sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        layers=info,
        format="<레이어>__xy (N,2) float64 이어 붙인 좌표, <레이어>__off 부분 시작 위치(부분 수 + 1), <레이어>__kind 부분 도형 종류",
    )
    NPZ.parent.mkdir(exist_ok=True)
    np.savez_compressed(NPZ, **arrs)
    SRC_JSON.write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
    print(NPZ, f"{NPZ.stat().st_size / 1024:.1f} KB")
    for k, v in info.items():
        print(f"  {k}: 부분 {v['n_parts']}, 점 {v['n_points']}, {v['kinds']}")


if __name__ == "__main__":
    main()
