"""화면 끝 검출 처리 (비전 코어, replay_pf.py에서 분리 11-04): 판 일부만 보인 검출을 판 끝점과 비교한다.

박스가 화면 좌우 끝 EDGE(2%) 안에 걸리면 판 일부만 보여 박스 가운데 ≠ 판 가운데다. obs_model은 그대로 두고
지도에 점을 더 넣어 해결한다(가시성·높이는 그 판 것 그대로, 원래 행은 그대로라 화면 안 검출은 전과 같음).
    · 한쪽만 잘림(기본, mode='end'): 박스의 안쪽 끝은 판의 진짜 끝이다. 왼쪽이 잘렸으면 박스 오른쪽 끝 방위를
      판의 오른쪽 끝점('종류@오른끝')과, 오른쪽이 잘렸으면 왼쪽 끝 방위를 '종류@왼끝'과 비교한다.
      왼쪽·오른쪽은 글자 면을 마주 본 사람 기준(보이는 쪽은 늘 앞면이라 정해짐)
    · 양쪽 다 잘림, 또는 mode='seg': 판 면을 따라 step 간격 점('종류@면') 중 가장 가까운 점과 비교
      ('seg'만 쓰면 방위를 일부러 25° 틀어도 맞음이 비슷하게 늘어 느슨함 → 기본은 'end')
    · mode='0': 끔(비교용)
끝점 위치는 지도와 같은 동·북(ENU) 그대로 둔다(obs_model이 ENU에서 계산). 판 폭은 랜드마크 지도의 크기 값(size[0]).

numpy와 obs_model.kind_types만 쓴다(팀장 브랜치 vision/에 그대로 복사됨 — 원본은 localization_design).
"""
import numpy as np

from obs_model import kind_types

EDGE, FACE_STEP = 0.02, 0.25
SUF_SEG, SUF_L, SUF_R = "@면", "@왼끝", "@오른끝"
MODES = ("end", "seg", "0")


def add_face_points(fmap, step=FACE_STEP):
    """층마다 판 면(가로 폭 size[0])의 왼끝·오른끝 점과, step 간격 면 위 점을 '종류@…'로 덧붙인 지도.
    fmap: mapdata.load_map 형식(lm_json에 size·normal 필요). lm_st가 있으면 같이 늘린다(격자 좌표 = (ENU - T) @ R)."""
    out = {}
    for fl, fm in fmap.items():
        rows = []
        for k, d in enumerate(fm["lm_json"]):
            w = d["size"][0]
            nx, ny = np.array(d["normal"][:2]) / max(np.hypot(*d["normal"][:2]), 1e-9)
            right = np.array([-ny, nx])          # 글자 면을 마주 본 사람(-normal 방향을 봄)의 오른쪽
            c = fm["lm_xy"][k]
            rows += [(k, c - w / 2 * right, SUF_L), (k, c + w / 2 * right, SUF_R)]
            n = max(1, int(np.ceil(w / step)) + 1)
            rows += [(k, c + s * right, SUF_SEG) for s in (np.linspace(-w / 2, w / 2, n) if n > 1 else [0.0])]
        idx = np.array([k for k, _, _ in rows])
        xy = np.array([p for _, p, _ in rows])
        suf = np.array([q for _, _, q in rows])
        g = dict(fm)
        g["dist"] = np.concatenate([fm["dist"], fm["dist"][idx]])
        g["lm_xy"] = np.concatenate([fm["lm_xy"], xy])
        g["lm_dz"] = np.concatenate([fm["lm_dz"], fm["lm_dz"][idx]])
        g["lm_type"] = np.concatenate([fm["lm_type"], np.char.add(fm["lm_type"][idx].astype(str), suf)])
        g["lm_id"] = np.concatenate([fm["lm_id"], np.char.add(fm["lm_id"][idx].astype(str), suf)])
        if "lm_st" in fm:
            g["lm_st"] = np.concatenate([fm["lm_st"], (np.atleast_2d(xy)[:, :2] - fm["T"]) @ fm["R"]])
        g["lm_json"] = list(fm["lm_json"]) + [fm["lm_json"][k] for k in idx]
        out[fl] = g
    return out


def base_id(lm_id):
    """'B01_route_map_22@왼끝' → 'B01_route_map_22'"""
    return str(lm_id).split("@")[0]


def edge_obs(o, box, fi, mode="end"):
    """관측 하나 → (types, bearing, edge). 화면 끝에 걸렸으면 종류·방위를 위 규칙대로 바꾼다.
    o: cv_module observation, box: 그 검출 box_xyxy, fi: frame_info(image_width, intrinsics)."""
    W = fi["image_width"]
    cut_l, cut_r = box[0] < EDGE * W, box[2] > (1 - EDGE) * W
    edge = cut_l or cut_r
    if mode == "0" or not edge:
        return o["map_types"], o["bearing_deg"], edge
    kinds = kind_types({"types": o["map_types"], "type": o["map_types"][0]})
    if mode == "seg" or (cut_l and cut_r):
        return [t + SUF_SEG for t in kinds], o["bearing_deg"], edge
    I = fi["intrinsics"]
    ang = lambda u: np.degrees(np.arctan((u - I["cx"]) / I["fx"]))
    u = box[2] if cut_l else box[0]                     # 안쪽 끝 = 판의 진짜 끝
    b = o["bearing_deg"] + ang(u) - ang((box[0] + box[2]) / 2)   # 가운데 방위(roll 보정 포함)에서 옮김
    return [t + (SUF_R if cut_l else SUF_L) for t in kinds], float(b), edge


def frame_obs(f, t0, mode="end"):
    """cv_module FrameResult 하나 → (t, [obs dict]). t = capture_time_s - t0. 지도 종류가 없는 관측은 뺀다.
    obs dict: obs_model 입력 {types, type, bearing, distance, score} + label, edge(화면 끝 검출인가)."""
    det = {d["detection_id"]: d for d in f["detections"]}
    obs = []
    for o in f["observations"]:
        if not o["map_types"]:
            continue
        types, bearing, edge = edge_obs(o, det[o["detection_id"]]["box_xyxy"], f["frame_info"], mode)
        obs.append(dict(types=types, type=types[0], bearing=bearing, distance=o["distance_m"],
                        score=o["score"], label=o["label"], edge=edge))
    return f["frame_info"]["capture_time_s"] - t0, obs
