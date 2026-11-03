"""역사 패키지 읽기 (비전 코어, 11-04): export_station.py가 내보낸 npz + json → mapdata.load_map과 같은 fmap.

plan.py·build_landmarks.py·메시 없이 관측 모델 지도를 만든다(vision-integration 브랜치 vision/에 그대로 복사됨 — 원본은
localization_design). numpy만 쓴다.

반환 fmap {층 정수: dict} 필드 = mapdata.load_map과 같음
    x0, y0, cell, R, T, dist, walk, walk_pf, lm_id, lm_type, lm_xy, lm_st, lm_dz, lm_json, floor_z, name
    + computed (H, W) bool: 가시성 표를 실제로 계산한 칸. 그 밖은 dist가 nan이라도 '안 보임'이 아니라 '모름'
meta (두 번째 반환값, load_station(..., with_meta=True)): 층 이름·카메라 높이·좌표 변환·에스컬레이터 끝·출처
좌표 약속
    ENU (동·북 m, 메시 원점)     : 관측 모델·랜드마크·에스컬레이터 끝
    격자(역 좌표) st              : st = (ENU - T) @ R  (가시성 표 칸 찾기용)
    지도 aligned (maps/dongchun)  : aligned = Rot(meta['aligned_from_enu_rot_deg']) · ENU  (원점 그대로, T 안 뺌)
    나침반 (북 0°, 시계 +) ↔ aligned 수학각 th(반시계 +, rad): 나침반 = wrap360(90 + rot - deg(th))
"""
import json
from pathlib import Path

import numpy as np

NAME = "dongchun_station"
ARRAYS = ("x0", "y0", "cell", "R", "T", "dist", "walk", "walk_pf", "computed", "lm_id", "lm_type", "lm_xy",
          "lm_st", "lm_dz", "floor_z")


def load_station(folder, with_meta=False):
    folder = Path(folder)
    meta = json.loads((folder / f"{NAME}.json").read_text(encoding="utf-8"))
    z = np.load(folder / f"{NAME}.npz")
    fmap = {}
    for name, fl in meta["floors"].items():
        fm = {k: z[f"{name}/{k}"] for k in ARRAYS}
        for k in ("x0", "y0", "cell", "floor_z"):
            fm[k] = float(fm[k])
        fm["lm_id"], fm["lm_type"] = fm["lm_id"].astype(str), fm["lm_type"].astype(str)
        fm["lm_json"] = meta["lm_json"][name]
        fm["name"] = name
        fmap[int(fl)] = fm
    return (fmap, meta) if with_meta else fmap


def enu_to_aligned(xy, rot_deg):
    a = np.radians(rot_deg)
    c, s = np.cos(a), np.sin(a)
    xy = np.atleast_2d(np.asarray(xy, float))
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def aligned_to_enu(xy, rot_deg):
    return enu_to_aligned(xy, -rot_deg)


def th_to_compass(th, rot_deg):
    """aligned 수학각 [rad] → 나침반 [도]"""
    return (90.0 + rot_deg - np.degrees(th)) % 360


def compass_to_th(deg, rot_deg):
    """나침반 [도] → aligned 수학각 [rad] ((-π, π])"""
    a = np.radians(90.0 + rot_deg - np.asarray(deg, float))
    return np.arctan2(np.sin(a), np.cos(a))
