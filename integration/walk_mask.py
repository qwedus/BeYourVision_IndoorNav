"""방 뺀 보행 마스크 사본 (연결안 3절 2-1): 팀장 B01_walk.png·B02_walk.png에서 방·구조물 안쪽을 뺀 사본을 만든다.

    .venv/Scripts/python integration/walk_mask.py   → integration/maps/B0x_walk_noroom.png + map_meta_noroom.json

팀장 마스크는 평면도 이미지 색으로 만든 것이라 B01 직원실·매표실 등 방, B02 계단·에스컬레이터 구조물 안쪽, 그림 안
글자·지시선까지 '걸을 수 있음'으로 들어가 있다. 규칙: **팀장 마스크 칸 중 우리 보행칸(가시성 표 walk, 0.5 m 격자)에서
MARGIN_M(0.5 m) 넘게 떨어진 칸은 뺀다.** 우리 보행칸은 메시에서 바닥 높이·공공 구역으로 만든 것이라 방이 없다.
벽 가까이 팀장 경계는 그대로 둠(0.5 m 안). 결과는 팀장 GridMap.load(meta, floor)로 그대로 읽는다(map_meta 사본이
walk_png만 이 사본을 가리키고 m_per_px·offset은 팀장 값 그대로). 팀장 파일은 고치지 않는다.
"""
import json

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import paths
from indoornav.pf.gridmap import GridMap
from station_pkg import load_station, aligned_to_enu

MARGIN_M = 0.5


def ours_on_team_grid(g, fm, rot):
    """팀장 마스크 픽셀마다 우리 보행칸(walk) 위인가."""
    H, W = g.walk.shape
    ys, xs = np.mgrid[0:H, 0:W]
    en = aligned_to_enu(g.to_world(xs.ravel(), ys.ravel()), rot)
    st = (en - fm["T"]) @ fm["R"]
    j = np.floor((st[:, 0] - fm["x0"]) / fm["cell"]).astype(int)
    i = np.floor((st[:, 1] - fm["y0"]) / fm["cell"]).astype(int)
    h, w = fm["walk"].shape
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    out = np.zeros(len(st), bool)
    out[ok] = fm["walk"][i[ok], j[ok]]
    return out.reshape(H, W)


def main():
    fmap, smeta = load_station(paths.INPUTS, with_meta=True)
    meta = json.loads(paths.TEAM_META.read_text(encoding="utf-8"))
    rot = meta["aligned_from_enu_rot_deg"]
    assert rot == smeta["aligned_from_enu_rot_deg"], (rot, smeta["aligned_from_enu_rot_deg"])
    paths.MASK_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for fl, name in paths.FLOOR_NAME.items():
        g = GridMap.load(paths.TEAM_META, name)
        ours = ours_on_team_grid(g, fmap[fl], rot)
        d = distance_transform_edt(~ours) * g.m_per_px
        keep = g.walk & (d <= MARGIN_M)
        Image.fromarray((keep * 255).astype(np.uint8)).save(paths.MASK_DIR / f"{name}_walk_noroom.png")
        meta["floors"][name]["walk_png"] = f"{name}_walk_noroom.png"
        for k in ("drawn_path_png",):
            meta["floors"][name][k] = f"../../maps/dongchun/{meta['floors'][name][k]}"
        a = g.m_per_px ** 2
        report[name] = dict(team_m2=round(g.walk.sum() * a, 1), kept_m2=round(keep.sum() * a, 1),
                            removed_m2=round((g.walk & ~keep).sum() * a, 1),
                            team_in_ours=round(float((g.walk & ours).sum() / g.walk.sum()), 4))
    meta["note"] = (f"integration/walk_mask.py가 만든 사본: 팀장 maps/dongchun/map_meta.json과 같고 walk_png만 방 뺀 마스크"
                    f"(팀장 마스크 ∩ 우리 보행칸에서 {MARGIN_M} m 안). 원래 설명: " + meta["note"])
    meta["noroom"] = dict(margin_m=MARGIN_M, rule="team_walk & dist_to(our walk) <= margin", report=report)
    paths.MASK_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
