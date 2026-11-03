"""PF용 지도: 벽 선분 로드 + 이동 선분의 벽 교차 판정 (numpy 벡터화)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WallMap:
    a: np.ndarray            # [M,2] 벽 시작점
    b: np.ndarray            # [M,2] 벽 끝점
    checkpoints: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "WallMap":
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        w = np.asarray(m["walls"], float)
        cps = {k: np.asarray(v, float) for k, v in m["checkpoints"].items()}
        return cls(a=w[:, 0], b=w[:, 1], checkpoints=cps)

    def crosses(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """p0→p1 이동(각 [N,2])이 벽을 하나라도 가로지르면 True. 반환 [N] bool."""
        p = p0[:, None, :]                 # [N,1,2]
        r = (p1 - p0)[:, None, :]          # [N,1,2]
        q = self.a[None, :, :]             # [1,M,2]
        s = (self.b - self.a)[None, :, :]  # [1,M,2]
        cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        den = cross(r, s)                  # [N,M]
        qp = q - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(qp, s) / den         # 이동 선분 위 위치
            u = cross(qp, r) / den         # 벽 선분 위 위치
        hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        return hit.any(axis=1)
