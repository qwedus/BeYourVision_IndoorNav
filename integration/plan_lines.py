"""정식 도면 벽선·점자블록 읽기(numpy만): make_plan_lines.py가 만든 maps/plan_lines.npz에서 레이어 도형을 꺼내 역 좌표로 바꿈."""
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parent / "maps" / "plan_lines.npz"
SRC_JSON = NPZ.with_suffix(".json")


def load_lines(layer, R, T, path=None):
    """[(도형 종류, 역 좌표 (n, 2)), ...] — fig_plan.lines_st와 같은 순서·값. 좌표 변환 (xy - T) @ R"""
    with np.load(NPZ if path is None else path) as z:
        if f"{layer}__xy" not in z.files:
            raise KeyError(f"plan_lines에 레이어 없음: {layer}")
        xy, off, kind = z[f"{layer}__xy"], z[f"{layer}__off"], z[f"{layer}__kind"]
    return [(str(k), (xy[off[i]:off[i + 1]] - T) @ R) for i, k in enumerate(kind)]
