"""시간 필터 보조 잇기 옵션(vision/time_filter.py center_dist_max·pitch_comp)과 과신 대책 시험 연결(l05·l10·lp) 단위 시험.

합성 프레임으로 규칙을 보고, 녹화 s2_b_ex2_31이 있으면 실제 관측으로 PC 필터·앱 규칙(check_tf_app.app_rule)을 대조한다.
"""
import copy
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402,F401  (vision/·indoornav 경로 준비)
import exp_overconf_fix as ovf  # noqa: E402
import replay as rp  # noqa: E402
import run_i1  # noqa: E402
import time_filter as tf  # noqa: E402

FY = 1000.0


# ---------------------------------------------------------------- 합성 프레임
def _frame(k, dets, pitch_deg=None, fy=FY, t=None):
    """dets: [(label, box)] → cv_module FrameResult 비슷한 dict (detection_id = f{k}_{i})"""
    info = {"capture_time_s": k * 0.2 if t is None else t, "image_height": 1080}
    if pitch_deg is not None:
        info["pose"] = {"pitch_deg": pitch_deg}
    if fy is not None:
        info["intrinsics"] = {"fy": fy}
    detections, obs = [], []
    for i, (label, box) in enumerate(dets):
        did = f"f{k}_{i}"
        detections.append({"detection_id": did, "box_xyxy": list(box)})
        obs.append({"detection_id": did, "label": label, "score": 0.9})
    return {"frame_info": info, "detections": detections, "observations": obs, "status": "ok"}


def _kept(res):
    return [[o["detection_id"] for o in f["observations"]] for f in res]


def _tids(res):
    return {o["detection_id"]: o["track_id"] for f in res for o in f["observations"]}


def _old_apply(frames, window=5, min_hits=3, iou_min=0.3):
    """옵션 추가 전 time_filter.apply 반복문 그대로(참조 구현)"""
    tracks, out, next_id = [], [], 0
    for k, f in enumerate(frames):
        f = json.loads(json.dumps(f))
        boxes = {d["detection_id"]: d["box_xyxy"] for d in f.get("detections", [])}
        keep, drop, used = [], [], set()
        for ob in f.get("observations", []):
            box = boxes[ob["detection_id"]]
            best, bi = iou_min, None
            for ti, tr in enumerate(tracks):
                if ti in used or tr["label"] != ob["label"] or k - tr["hits"][-1] > window:
                    continue
                v = tf.iou(box, tr["box"])
                if v >= best:
                    best, bi = v, ti
            if bi is None:
                tracks.append({"id": next_id, "label": ob["label"], "box": box, "hits": []})
                next_id += 1
                bi = len(tracks) - 1
            used.add(bi)
            tr = tracks[bi]
            tr["box"] = box
            tr["hits"].append(k)
            ob["track_id"] = tr["id"]
            n = sum(1 for h in tr["hits"] if h > k - window)
            (keep if n >= min_hits else drop).append(ob)
        tracks = [t for t in tracks if k - t["hits"][-1] <= window]
        f["observations"] = keep
        f.setdefault("dropped", []).extend(
            {"detection_id": o["detection_id"], "label": o["label"], "score": o["score"], "reason": "time_filter"}
            for o in drop)
        if f["status"] != "failed":
            f["status"] = "ok" if keep else "empty"
        out.append(f)
    return out


def _random_seq(seed, n=40):
    """판 몇 개가 조금씩 움직이고, 가끔 빠지고, 같은 자리 겹친 판(IoU 동점)도 나오는 순서"""
    rng = random.Random(seed)
    labels = ["exit_sign", "info_monitor"]
    objs = [[rng.choice(labels), rng.uniform(0, 1500), rng.uniform(0, 900), rng.uniform(20, 120)] for _ in range(4)]
    frames = []
    for k in range(n):
        dets = []
        for o in objs:
            o[1] += rng.uniform(-15, 15)
            o[2] += rng.uniform(-15, 15)
            if rng.random() < 0.8:
                dets.append((o[0], [o[1], o[2], o[1] + o[3], o[2] + o[3] * 0.6]))
        if dets and rng.random() < 0.3:        # 같은 박스 중복 → 앞 track 둘과 IoU 동점
            dets.append(copy.deepcopy(dets[0]))
        if rng.random() < 0.1:
            dets = []
        rng.shuffle(dets)
        frames.append(_frame(k, dets, pitch_deg=rng.uniform(-5, 5)))
    return frames


# ---------------------------------------------------------------- 1. 옵션 끄면 원래 규칙
@pytest.mark.parametrize("seed", range(6))
def test_options_off_matches_old_rule(seed):
    fr = _random_seq(seed)
    assert tf.apply(fr) == _old_apply(fr)
    assert tf.apply(fr, window=3, min_hits=2, iou_min=0.1) == _old_apply(fr, 3, 2, 0.1)
    assert tf.apply(fr, center_dist_max=None, pitch_comp=False) == _old_apply(fr)


def test_iou_tie_goes_to_later_track_when_off():
    """같은 박스 두 track이 있으면 뒤 track(>=). 옵션 켜도(cd·pitch 쓸 일 없는 경우) 같은 동점 규칙"""
    b = [100, 100, 200, 150]
    fr = [_frame(0, [("exit_sign", b), ("exit_sign", b)]), _frame(1, [("exit_sign", b)])]
    for kw in ({}, {"center_dist_max": 0.5}, {"pitch_comp": True}):
        res = tf.apply(fr, min_hits=1, **kw)
        assert res[1]["observations"][0]["track_id"] == 1, kw
    assert tf.apply(fr, min_hits=1) == _old_apply(fr, min_hits=1)


# ---------------------------------------------------------------- 2. center_dist
def test_center_dist_values():
    a = [0, 0, 30, 40]                          # 대각선 50
    assert tf.center_dist(a, a) == 0.0
    assert tf.center_dist(a, [0, 30, 30, 70]) == pytest.approx(30 / 50)
    assert tf.center_dist(a, [30, 40, 60, 80]) == pytest.approx(50 / 50)
    big = [0, 0, 60, 80]                        # 대각선 100, 중심 (30, 40) / a 중심 (15, 20)
    assert tf.center_dist(a, big) == pytest.approx(25 / 100)
    assert tf.center_dist(big, a) == tf.center_dist(a, big)


def _moving(dy=30, n=5, label="exit_sign"):
    """높이 50 px 판이 프레임마다 세로로 dy씩 움직임 (IoU = 20/80 = 0.25 < 0.3)"""
    return [_frame(k, [(label, [100, 100 + dy * k, 180, 150 + dy * k])]) for k in range(n)]


def test_cd_links_moving_box_and_passes_filter():
    fr = _moving()
    b0, b1 = (fr[i]["detections"][0]["box_xyxy"] for i in (0, 1))
    assert tf.iou(b0, b1) < 0.3
    assert tf.center_dist(b0, b1) < 0.5
    off = tf.apply(fr)
    assert sum(map(len, _kept(off))) == 0
    assert len(set(_tids(tf.apply(fr, min_hits=1)).values())) == 5
    on = tf.apply(fr, center_dist_max=0.5)
    assert _kept(on) == [[], [], ["f2_0"], ["f3_0"], ["f4_0"]]
    assert set(_tids(on).values()) == {0}
    assert all(f["status"] == ("ok" if k >= 2 else "empty") for k, f in enumerate(on))
    assert [d["reason"] for f in on[:2] for d in f["dropped"]] == ["time_filter"] * 2


def test_cd_not_across_labels():
    fr = [_frame(k, [("exit_sign" if k % 2 == 0 else "info_monitor", [100, 100 + 30 * k, 180, 150 + 30 * k])])
          for k in range(5)]
    ids = _tids(tf.apply(fr, min_hits=1, center_dist_max=10.0))
    assert all(ids[f"f{k}_0"] != ids[f"f{k + 1}_0"] for k in range(4))       # 이웃 프레임(다른 종류)끼리 안 이어짐
    assert ids["f0_0"] == ids["f2_0"] == ids["f4_0"]                          # 같은 종류는 한 장 건너도 이어짐
    same = [_frame(k, [("exit_sign", [100, 100 + 30 * k, 180, 150 + 30 * k])]) for k in range(5)]
    assert len(set(_tids(tf.apply(same, min_hits=1, center_dist_max=10.0)).values())) == 1


def test_cd_not_when_too_far():
    fr = _moving(dy=60)                         # dn = 60 / 94.3 = 0.64
    dn = tf.center_dist(fr[0]["detections"][0]["box_xyxy"], fr[1]["detections"][0]["box_xyxy"])
    assert 0.5 < dn < 1.0
    assert len(set(_tids(tf.apply(fr, min_hits=1, center_dist_max=0.5)).values())) == 5
    assert len(set(_tids(tf.apply(fr, min_hits=1, center_dist_max=1.0)).values())) == 1
    assert len(set(_tids(tf.apply(fr, min_hits=1, center_dist_max=dn)).values())) == 1     # dn == cd도 통과


def test_iou_track_preferred_over_cd_track():
    """앞 프레임: track 0 = IoU 통과할 박스, track 1 = 중심 거리 0인데 IoU 미만(크기 다른) 박스.
    track 1이 뒤라서 동점이면 이기지만 IoU 점수(1+IoU)가 중심 거리 점수(≤ 1+1e-6)보다 커서 track 0."""
    cur = [100, 100, 200, 150]
    near = [105, 100, 205, 150]                 # IoU 0.905
    same_center = [140, 115, 160, 135]          # 중심 같음, IoU 0.08
    assert tf.iou(cur, near) >= 0.3 and tf.iou(cur, same_center) < 0.3
    assert tf.center_dist(same_center, cur) == 0.0
    fr = [_frame(0, [("exit_sign", near), ("exit_sign", same_center)]), _frame(1, [("exit_sign", cur)])]
    res = tf.apply(fr, min_hits=1, center_dist_max=0.5)
    assert res[1]["observations"][0]["track_id"] == 0


def test_cd_higher_score_closer_track():
    """IoU 둘 다 미만이면 중심 거리가 가까운 track(점수 1 - dn/cd)이 이김. 앞 track이어도"""
    cur = [100, 160, 180, 210]
    fr = [_frame(0, [("exit_sign", [100, 130, 180, 180]), ("exit_sign", [100, 115, 180, 165])]),
          _frame(1, [("exit_sign", cur)])]
    d0 = tf.center_dist(fr[0]["detections"][0]["box_xyxy"], cur)
    d1 = tf.center_dist(fr[0]["detections"][1]["box_xyxy"], cur)
    assert d0 < d1 < 0.5
    assert all(tf.iou(d["box_xyxy"], cur) < 0.3 for d in fr[0]["detections"])
    assert tf.apply(fr, min_hits=1, center_dist_max=0.5)[1]["observations"][0]["track_id"] == 0


# ---------------------------------------------------------------- 3. pitch_comp
def _pitched(dy_px, dpitch_deg, n=5, pitch0=0.0, pitch=True):
    """박스가 프레임마다 dy_px 내려가고, 자세 pitch는 dpitch_deg씩 바뀜"""
    return [_frame(k, [("exit_sign", [100, 100 + dy_px * k, 180, 150 + dy_px * k])],
                   pitch_deg=(pitch0 + dpitch_deg * k) if pitch else None) for k in range(n)]


def test_pitch_fy_reads_frame():
    f = _frame(0, [], pitch_deg=2.0, fy=1362.0)
    p, fy = tf._pitch_fy(f)
    assert p == pytest.approx(math.radians(2.0)) and fy == 1362.0
    assert tf._pitch_fy(_frame(0, [], pitch_deg=None)) is None
    assert tf._pitch_fy(_frame(0, [], pitch_deg=1.0, fy=None)) is None
    assert tf._pitch_fy({"frame_info": None}) is None and tf._pitch_fy({}) is None


def test_pitch_comp_exact_shift_links():
    dpd = 3.0
    dy = FY * math.radians(dpd)                 # 52.4 px: IoU 0, 중심 거리 0.55
    fr = _pitched(dy, dpd)
    assert tf.iou(fr[0]["detections"][0]["box_xyxy"], fr[1]["detections"][0]["box_xyxy"]) == 0.0
    assert sum(map(len, _kept(tf.apply(fr)))) == 0
    on = tf.apply(fr, pitch_comp=True)
    assert _kept(on) == [[], [], ["f2_0"], ["f3_0"], ["f4_0"]]
    assert set(_tids(on).values()) == {0}
    # 옮긴 박스와 지금 박스 IoU 1 (식 그대로 계산)
    b0, b1 = fr[0]["detections"][0]["box_xyxy"], fr[1]["detections"][0]["box_xyxy"]
    p0, p1 = tf._pitch_fy(fr[0]), tf._pitch_fy(fr[1])
    s = p1[1] * (p1[0] - p0[0])
    assert tf.iou([b0[0], b0[1] + s, b0[2], b0[3] + s], b1) == pytest.approx(1.0)


def test_pitch_comp_sign():
    """pitch가 커지면 앞 박스가 아래(+y)로 옮겨짐. 반대 방향으로 움직인 박스는 안 이어짐"""
    dpd = 3.0
    dy = FY * math.radians(dpd)
    assert len(set(_tids(tf.apply(_pitched(dy, dpd), min_hits=1, pitch_comp=True)).values())) == 1
    assert len(set(_tids(tf.apply(_pitched(-dy, -dpd), min_hits=1, pitch_comp=True)).values())) == 1
    assert len(set(_tids(tf.apply(_pitched(-dy, dpd), min_hits=1, pitch_comp=True)).values())) == 5


def test_pitch_comp_missing_pitch_is_off():
    dpd = 3.0
    dy = FY * math.radians(dpd)
    fr = _pitched(dy, dpd, pitch=False)
    assert tf.apply(fr, pitch_comp=True) == tf.apply(fr)
    # 한 프레임만 pitch 없음: 그 프레임은 안 옮김 → 끊김
    fr = _pitched(dy, dpd)
    del fr[2]["frame_info"]["pose"]
    ids = _tids(tf.apply(fr, min_hits=1, pitch_comp=True))
    assert ids["f0_0"] == ids["f1_0"] and ids["f2_0"] != ids["f1_0"]
    # fy 없음도 같음
    fr = [_frame(k, [("exit_sign", [100, 100 + dy * k, 180, 150 + dy * k])], pitch_deg=dpd * k, fy=None)
          for k in range(5)]
    assert tf.apply(fr, pitch_comp=True) == tf.apply(fr)


def test_pitch_comp_no_change_equals_off():
    fr = _random_seq(3)
    for f in fr:
        f["frame_info"]["pose"]["pitch_deg"] = 1.5
    assert tf.apply(fr, pitch_comp=True) == tf.apply(fr)


def test_pitch_comp_cd_uses_unshifted_box():
    """pitch 보정이 엉뚱하게 옮겨도(박스는 그대로, pitch만 크게 바뀜) 중심 거리는 안 옮긴 박스로 → 이어짐"""
    fr = [_frame(k, [("exit_sign", [100, 100, 180, 150])], pitch_deg=10.0 * k) for k in range(5)]
    assert len(set(_tids(tf.apply(fr, min_hits=1, pitch_comp=True)).values())) == 5
    assert len(set(_tids(tf.apply(fr, min_hits=1, pitch_comp=True, center_dist_max=0.5)).values())) == 1


def test_pitch_fy_failed_frame_is_none():
    f = _frame(0, [], pitch_deg=2.0)
    assert tf._pitch_fy(f) is not None
    f["status"] = "failed"
    assert tf._pitch_fy(f) is None
    # 실패 프레임 pitch가 수가 아니어도 apply가 안 멈춤
    dpd = 3.0
    dy = FY * math.radians(dpd)
    fr = _pitched(dy, dpd)
    fr[2]["status"] = "failed"
    fr[2]["frame_info"]["pose"]["pitch_deg"] = "bad"
    assert tf._pitch_fy(fr[2]) is None
    res = tf.apply(fr, pitch_comp=True)
    assert res[2]["status"] == "failed"
    ids = _tids(tf.apply(fr, min_hits=1, pitch_comp=True))
    assert ids["f0_0"] == ids["f1_0"] and ids["f2_0"] != ids["f1_0"]


def test_pitch_fy_non_device_source_is_none():
    f = _frame(0, [], pitch_deg=10.0)
    assert tf._pitch_fy(f) is not None                          # source 없음 = device
    f["frame_info"]["pose"]["source"] = "device"
    assert tf._pitch_fy(f) == pytest.approx((math.radians(10.0), FY))
    for src in ("assumed", "default", None):
        g = _frame(0, [], pitch_deg=0.0)
        g["frame_info"]["pose"]["source"] = src
        assert tf._pitch_fy(g) is None, src


def test_pitch_comp_assumed_pose_no_spurious_shift():
    """프레임 0만 실제 pitch 10°, 나머지는 가정값(0.0, source=assumed). 박스는 그대로 → 한 track"""
    box = [100, 100, 180, 150]
    fr = [_frame(k, [("exit_sign", box)], pitch_deg=10.0 if k == 0 else 0.0) for k in range(5)]
    for f in fr[1:]:
        f["frame_info"]["pose"]["source"] = "assumed"
    on = tf.apply(fr, pitch_comp=True)
    assert set(_tids(on).values()) == {0}
    assert _kept(on) == [[], [], ["f2_0"], ["f3_0"], ["f4_0"]]
    # source를 device로 두면(가정값을 실제처럼 읽으면) 10° 차이만큼 옮겨져 끊김
    for f in fr[1:]:
        f["frame_info"]["pose"]["source"] = "device"
    ids = _tids(tf.apply(fr, min_hits=1, pitch_comp=True))
    assert ids["f0_0"] != ids["f1_0"]


def test_pitch_fy_scales_fy_to_image_height():
    f = _frame(0, [], pitch_deg=2.0)
    f["frame_info"].update(image_width=480, image_height=270)
    f["frame_info"]["intrinsics"] = {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0,
                                     "width": 1920, "height": 1080}
    p, fy = tf._pitch_fy(f)
    assert p == pytest.approx(math.radians(2.0)) and fy == pytest.approx(250.0)
    f["frame_info"]["intrinsics"]["height"] = 270                # 같은 크기면 그대로
    assert tf._pitch_fy(f)[1] == pytest.approx(1000.0)


def test_pitch_comp_scaled_fy_links():
    dpd = 12.0
    dy = 250.0 * math.radians(dpd)              # 52.4 px: IoU 0. fy 1000 그대로면 209 px 옮겨져 안 이어짐
    fr = _pitched(dy, dpd)
    for f in fr:
        f["frame_info"].update(image_width=480, image_height=270)
        f["frame_info"]["intrinsics"] = {"fx": 1000.0, "fy": 1000.0, "width": 1920, "height": 1080}
    assert tf.iou(fr[0]["detections"][0]["box_xyxy"], fr[1]["detections"][0]["box_xyxy"]) == 0.0
    assert sum(map(len, _kept(tf.apply(fr)))) == 0
    on = tf.apply(fr, pitch_comp=True)
    assert _kept(on) == [[], [], ["f2_0"], ["f3_0"], ["f4_0"]]
    assert set(_tids(on).values()) == {0}
    # 비례 안 맞춘 경우(K 높이 = 이미지 높이로 둠, fy 1000)엔 끊김
    for f in fr:
        f["frame_info"]["intrinsics"]["height"] = 270
    assert len(set(_tids(tf.apply(fr, min_hits=1, pitch_comp=True)).values())) == 5


@pytest.mark.parametrize("pitch,fy", [(float("nan"), FY), (float("inf"), FY), (float("-inf"), FY),
                                      (1.0, float("nan")), (1.0, float("inf")), (1.0, 0.0), (1.0, -500.0),
                                      ("nan", FY), (1.0, "abc")])
def test_pitch_fy_non_finite_or_bad_fy_is_none(pitch, fy):
    f = _frame(0, [], pitch_deg=1.0)
    f["frame_info"]["pose"]["pitch_deg"] = pitch
    f["frame_info"]["intrinsics"]["fy"] = fy
    assert tf._pitch_fy(f) is None


# ---------------------------------------------------------------- 4. track_ids
@pytest.mark.parametrize("kw", [{}, {"center_dist_max": 0.5}, {"center_dist_max": 1.0}, {"pitch_comp": True},
                                {"center_dist_max": 0.5, "pitch_comp": True}])
def test_track_ids_same_as_apply_min_hits_1(kw):
    for seed in range(3):
        fr = _random_seq(seed)
        ids = tf.track_ids(fr, **kw)
        assert ids == _tids(tf.apply(fr, min_hits=1, **kw))
        assert len(ids) == sum(len(f["observations"]) for f in fr)
        # 필터 뒤 관측의 track 번호도 같은 추적
        for f in tf.apply(fr, **kw):
            for o in f["observations"]:
                assert o["track_id"] == ids[o["detection_id"]]


# ---------------------------------------------------------------- 5. exp_overconf_fix 연결
def test_parse_link():
    assert ovf.parse("e3_g30_l05") == dict(eps=0.3, gate=0.3, cap=None, skip=None, link={"center_dist_max": 0.5}, wt=None)
    assert ovf.parse("e3_g30_l10")["link"] == {"center_dist_max": 1.0}
    assert ovf.parse("e3_g30_lp")["link"] == {"pitch_comp": True}
    assert ovf.parse("l05") == dict(eps=None, gate=None, cap=None, skip=None, link={"center_dist_max": 0.5}, wt=None)
    assert ovf.parse("base")["link"] is None
    assert ovf.parse("e3_g30_c5")["link"] is None


def test_all_combos_parse():
    for c in ovf.COMBOS:
        opt = ovf.parse(c)
        assert set(opt) == {"eps", "gate", "cap", "skip", "link", "wt"}
    for c in ("l05", "e3_g30_l05", "e3_g30_l10", "e3_g30_lp"):
        assert c in ovf.COMBOS
        assert ovf.parse(c)["link"] is not None
    # 연결 옵션은 apply 인자 이름과 맞아야 함
    for k, v in ovf.PIECES.items():
        if v[0] == "link":
            tf.apply([], **v[1])


@pytest.mark.parametrize("bad", [{"text_check": True}, {"cls_th": {"exit_sign": 0.5}}, {"bearing_fix": "bearing"}])
def test_link_obs_rejects_incompatible(bad):
    with pytest.raises(SystemExit):
        ovf.link_obs(dict(track_limit=1.0, **bad), {"center_dist_max": 0.5})


def test_link_obs_synthetic(tmp_path, monkeypatch):
    """obs_nofilter에 옵션 건 필터 → frames_obs, track_limit면 track 번호 붙임"""
    fr = _moving()
    for f in fr:
        f["frame_info"].update(image_width=1920, intrinsics={"fx": FY, "fy": FY, "cx": 960, "cy": 540})
        for o in f["observations"]:
            o.update(map_types=["exit_sign"], bearing_deg=0.0, elevation_deg=0.0, distance_m=None, point_px=[140.0, 125.0])
    try:
        rp.frames_obs(fr, 0.0)
    except Exception as e:                      # 합성 프레임이 frame_obs 형식에 모자라면 이 시험만 건너뜀
        pytest.skip(f"합성 프레임이 frame_obs에 안 맞음: {e!r}")
    (tmp_path / "obs_nofilter.jsonl").write_text("\n".join(json.dumps(f) for f in fr), encoding="utf-8")
    monkeypatch.setattr(ovf.paths, "I1", tmp_path)
    monkeypatch.setitem(run_i1._G, "t0", 0.0)
    off = ovf.link_obs({}, {})
    on = ovf.link_obs({}, {"center_dist_max": 0.5})
    assert sum(len(o) for _, o in off) == 0
    assert sum(len(o) for _, o in on) == 3
    on_t = ovf.link_obs({"track_limit": 1.0}, {"center_dist_max": 0.5})
    assert [o["track_id"] for _, obs in on_t for o in obs] == [0, 0, 0]


# ---------------------------------------------------------------- 6. 실제 녹화 s2_b_ex2_31
REC_NAME = "s2_b_ex2_31"


def _rec():
    rc_path = ROOT / "integration" / "recordings" / f"{REC_NAME}.json"
    if not rc_path.exists():
        pytest.skip("녹화 설정 없음")
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    inputs, rec = ROOT / rc["inputs"], paths.rec_dir(rc, strict=False)
    if not ((inputs / "obs_nofilter.jsonl").exists() and (inputs / "obs_timefilter.jsonl").exists()):
        pytest.skip("녹화 관측 파일 없음")
    return inputs, rec


@pytest.fixture(scope="module")
def real():
    inputs, rec = _rec()
    raw = tf.load_frames(inputs / "obs_nofilter.jsonl")[1]
    saved = tf.load_frames(inputs / "obs_timefilter.jsonl")[1]
    return inputs, rec, raw, saved


def test_real_off_matches_saved(real):
    _, _, raw, saved = real
    res = tf.apply(raw)
    assert _kept(res) == _kept(saved)
    assert res == _old_apply(raw)


def test_real_cd_keeps_more_and_close_to_app_rule(real):
    from check_tf_app import app_rule
    _, rec, raw, _ = real
    off = sum(len(f["observations"]) for f in tf.apply(raw))
    pc = {o["detection_id"] for f in tf.apply(raw, center_dist_max=0.5) for o in f["observations"]}
    assert len(pc) > off
    assert (off, len(pc)) == (760, 1349)
    cfg_path = rec / "config.json"
    if not cfg_path.exists():
        pytest.skip("녹화 앱 설정 없음")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["time_filter"]["match"]["center_dist_max"] = 0.5
    rule, _ = app_rule(raw, cfg)
    assert len(rule) == 1378
    # 남는 차이: 끊김 기준(PC 5장 / 앱 1 s), 동점 순서(PC 뒤 / 앱 앞) — check_tf_app.py 설명
    assert len(pc ^ rule) <= 40


def test_real_link_obs(real, monkeypatch):
    inputs, _, raw, saved = real
    monkeypatch.setattr(ovf.paths, "I1", inputs)
    monkeypatch.setitem(run_i1._G, "t0", 0.0)
    assert ovf.link_obs({}, {}) == rp.frames_obs(saved, 0.0)
    obs = ovf.link_obs({"track_limit": 1.0}, {"center_dist_max": 0.5})
    assert sum(len(o) for _, o in obs) == 1349
    ids = tf.track_ids(raw, center_dist_max=0.5)
    assert all(o["track_id"] == ids[o["detection_id"]] for _, os_ in obs for o in os_)
