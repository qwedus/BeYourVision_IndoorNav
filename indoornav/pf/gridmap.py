"""격자(이미지 마스크) 지도: 평면도 이미지에서 만든 '걸을 수 있는 곳' 마스크로 벽 판정.

WallMap 과 같은 인터페이스(crosses)를 제공해 ParticleFilter 에 그대로 넣을 수 있다.
좌표: aligned 프레임 [m] (X = 이미지 오른쪽, Y = 이미지 위). 픽셀 ↔ 좌표 변환은 m_per_px, offset 으로.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class GridMap:
    walk: np.ndarray          # [H, W] bool, True = 걸을 수 있음
    m_per_px: float
    offset: np.ndarray        # [2] aligned 좌표에서 픽셀 (0,0) 의 위치
    step_m: float = 0.08      # 선분 위 검사 간격

    @classmethod
    def load(cls, meta_path: str | Path, floor: str) -> "GridMap":
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        f = meta["floors"][floor]
        walk = np.array(Image.open(meta_path.parent / f["walk_png"])) > 127
        return cls(walk, float(f["m_per_px"]), np.asarray(f["offset_aligned"], float))

    def to_px(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = (xy[..., 0] - self.offset[0]) / self.m_per_px
        py = -(xy[..., 1] - self.offset[1]) / self.m_per_px
        return px, py

    def to_world(self, px, py) -> np.ndarray:
        return np.stack([np.asarray(px) * self.m_per_px + self.offset[0],
                         -np.asarray(py) * self.m_per_px + self.offset[1]], axis=-1)

    def free(self, xy: np.ndarray) -> np.ndarray:
        px, py = self.to_px(xy)
        ix, iy = np.round(px).astype(int), np.round(py).astype(int)
        h, w = self.walk.shape
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = np.zeros(ix.shape, bool)
        out[ok] = self.walk[iy[ok], ix[ok]]
        return out

    def crosses(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """p0→p1 이동(각 [N,2]) 중 걸을 수 없는 칸을 지나면 True."""
        d = np.linalg.norm(p1 - p0, axis=1)
        k = int(np.clip(np.ceil(d.max() / self.step_m) if len(d) else 1, 1, 200)) + 1
        s = np.linspace(0, 1, k)[None, :, None]
        pts = p0[:, None, :] + (p1 - p0)[:, None, :] * s       # [N, k, 2]
        return ~self.free(pts).all(axis=1)
