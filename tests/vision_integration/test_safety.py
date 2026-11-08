"""평가 안전장치 시험: 조건 비교, 입력 기록, 옛 파일 경고, 조용한 실패, 지표 이름, PC·앱 시간 필터 대조.

녹화 자료(integration/out/rec, 저장소 밖)가 있어야 하는 시험은 없으면 건너뛴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import check_tf_app as tf  # noqa: E402
import compare_runs as cr  # noqa: E402
import provenance as record  # noqa: E402
import replay as rp  # noqa: E402
import truth_build  # noqa: E402
from bearing_fix import rotation_since_step  # noqa: E402
from text_check import apply_decisions, link_stats  # noqa: E402
from time_filter import apply  # noqa: E402


# ---------------------------------------------------------------- 걸음 0개
def test_rotation_no_steps_uses_h0():
    d = rotation_since_step([0.5, 1.5], [0.0, 1.0, 2.0], [0.0, 0.2, 0.4], [], [], h0=0.1)
    assert d == pytest.approx([0.0, 0.2])


def test_rotation_with_steps_unchanged():
    d = rotation_since_step([0.5, 1.5], [0.0, 1.0, 2.0], [0.0, 0.2, 0.4], [1.0], [0.2], h0=0.0)
    assert d == pytest.approx([0.1, 0.1])


def test_rotation_length_mismatch_raises():
    with pytest.raises(ValueError, match="안 맞음"):
        rotation_since_step([0.5], [0.0, 1.0], [0.0, 0.2], [1.0], [])


# ---------------------------------------------------------------- 정답 기준점 경고
def _wp(**kw):
    return dict(dict(name="기준점", t=0.0, st=[1.0, 2.0], floor=-1), **kw)


def test_truth_warns_on_ref_and_basis():
    _, warn = truth_build.resolve([_wp(ref="모니터 아래"), _wp(basis="27번 판이 화면 위로 빠짐", t=1.0), _wp(ref="기둥", t=2.0)], None)
    assert len(warn) == 2
    assert "ref" in warn[0] and "basis" in warn[1]


def test_truth_no_false_warning_on_decision_word():
    _, warn = truth_build.resolve([_wp(basis="영상으로 판단한 기둥 모서리")], None)
    assert warn == []


def test_route2_sign27_waypoint_is_caught():
    wps = json.loads((paths.RECORDINGS / "route2_1030.json").read_text(encoding="utf-8"))["waypoints"]
    _, warn = truth_build.resolve(wps, None)
    assert any("27번 판" in w for w in warn)


# ---------------------------------------------------------------- 글자 확인 연결률
def _frames(ids):
    return [dict(observations=[dict(detection_id=i, label="exit", score=0.9)]) for i in ids]


def _dec(ids):
    return {i: dict(detection_id=i, decision="keep", text_status="confirmed") for i in ids}


def test_text_check_unlinked_file_fails():
    with pytest.raises(ValueError, match="안 이어짐"):
        apply_decisions(_frames(["a", "b"]), _dec(["x", "y"]))
    with pytest.raises(ValueError, match="안 이어짐"):
        apply_decisions(_frames(["a"]), {})


def test_text_check_subset_both_ways_passes():
    # 전체 관측용 결정 → 시간 필터 뒤 관측(작은 쪽이 다 이어짐)
    assert len(apply_decisions(_frames(["a"]), _dec(["a", "b", "c", "d"]))) == 1
    # 시간 필터 뒤 결정 → 전체 관측: 연결률을 알리고 그대로 넣음
    out = apply_decisions(_frames(["a", "b", "c", "d"]), _dec(["a"]))
    assert [o["detection_id"] for f in out for o in f["observations"]] == ["a", "b", "c", "d"]
    st = link_stats(_frames(["a", "b", "c", "d"]), _dec(["a"]))
    assert st["obs_rate"] == 0.25 and st["overlap"] == 1.0


def test_text_check_low_overlap_fails_high_passes():
    ids = [f"o{i}" for i in range(10)]
    with pytest.raises(ValueError):
        apply_decisions(_frames(ids), _dec(ids[:4] + [f"x{i}" for i in range(6)]))
    apply_decisions(_frames(ids), _dec(ids[:6] + [f"x{i}" for i in range(4)]))


def test_text_check_real_file_links_on_first_walk():
    from text_check import load_decisions
    from time_filter import load_frames
    dec = load_decisions(paths.INPUTS / "i1/text_check_easyocr.jsonl")[0]
    for name in ("obs_nofilter.jsonl", "obs_timefilter.jsonl"):
        assert link_stats(load_frames(paths.INPUTS / "i1" / name)[1], dec)["overlap"] == 1.0


# ---------------------------------------------------------------- 조건 비교
def _run(err, unc=None, n=100, dt=0.2):
    """걷는 프레임만 있는 재생 기록"""
    r = np.zeros(n, dtype=rp.REC_FIELDS)
    r["t"] = 3.0 + np.arange(n) * dt
    r["err"] = err
    r["unc"] = 1.0 if unc is None else unc
    r["unc95"] = 2.0
    r["floor"] = r["efloor"] = -1
    return r


@pytest.fixture
def plain_recording(monkeypatch):
    monkeypatch.setattr(paths, "RC", dict(exclude=[], end_start=1e9))


def _save(path, runs, cond="known_tf_기본"):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "runs.npz", **{f"{cond}_{s}": r for s, r in runs.items()})


def test_paired_direction_and_range():
    r = cr.paired([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 3.5])
    assert r["med"] == pytest.approx(1.0) and r["dir"] == "나빠짐" and r["agree"] == 3
    assert (r["lo"], r["hi"]) == (-0.5, 1.0)
    assert cr.paired([1.0, 1.0], [1.0, 1.0])["dir"] == "변화 없음"
    assert cr.paired([0.8, 0.8], [0.9, 0.9], better=1)["dir"] == "좋아짐"


def test_wrong_seconds(plain_recording):
    err = np.ones(100)
    err[10:20] = 5.0       # 10프레임 × 0.2 s = 2 s, 불확실도 1 m
    m = cr.seed_metrics(_run(err))
    assert m["wrong_s"] == pytest.approx(2.0)
    assert cr.seed_metrics(_run(err, unc=3.0))["wrong_s"] == 0.0


def test_compare_flags_worse_segment_even_if_overall_better(tmp_path, monkeypatch, plain_recording):
    monkeypatch.setattr(cr, "use_recording", lambda name: None)
    t = 3.0 + np.arange(100) * 0.2
    a, b = {}, {}
    for s in range(10):
        ea, eb = np.full(100, 3.0), np.full(100, 1.0)      # 전체는 좋아짐
        eb[(t >= 10) & (t < 15)] = 6.0                     # 10~15 s만 나빠짐
        a[s], b[s] = _run(ea + 0.01 * s), _run(eb + 0.01 * s)
    _save(tmp_path / "A", a)
    _save(tmp_path / "B", b)
    res = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"), seg=5.0, ranges=[(10.0, 15.0)])
    t90 = {x["key"]: x for x in res["table"]}
    assert t90["med_all"]["dir"] == "좋아짐" and t90["med_all"]["agree"] == 10
    worse = [r for r in res["bins"] if r["flag"] == "나빠짐"]
    assert [(r["t0"], r["t1"]) for r in worse] == [(10.0, 15.0)]
    assert res["picked"][0]["flag"] == "나빠짐"
    assert "나빠짐" in "\n".join(cr.render(res))


def test_compare_seed_split_flag(plain_recording):
    A = {s: _run(np.full(100, 2.0)) for s in range(10)}
    B = {s: _run(np.full(100, 3.0 if s < 6 else 1.5)) for s in range(10)}
    r = cr.segment_row(A, B, list(range(10)), 0.0, 30.0, 0.5)
    assert r["flag"] == "seed 갈림"


def test_compare_notes_seed_mismatch_and_missing_condition(tmp_path, monkeypatch, plain_recording):
    monkeypatch.setattr(cr, "use_recording", lambda name: None)
    _save(tmp_path / "A", {0: _run(np.ones(100)), 1: _run(np.ones(100))})
    _save(tmp_path / "B", {1: _run(np.ones(100))})
    res = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"))
    assert res["seeds"] == [1] and "seed가 달라" in res["notes"][0]
    with pytest.raises(SystemExit, match="조건"):
        cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"), cond="없는_조건")


def test_compare_reports_input_difference(tmp_path, monkeypatch, plain_recording):
    monkeypatch.setattr(cr, "use_recording", lambda name: None)
    for name, h in (("A", "1"), ("B", "2")):
        _save(tmp_path / name, {0: _run(np.ones(100))})
        info = dict(name=name, max_range=None if name == "A" else 30, provenance=dict(files={"지도": h, "걸음": "0"}, source_digest="s"))
        (tmp_path / name / "summary.json").write_text(json.dumps(dict(info=info)), encoding="utf-8")
    inp = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"))["inputs"]
    assert inp["files"] == ["지도"] and inp["config"] == {"max_range": (None, 30)} and inp["sources"] is True


# ---------------------------------------------------------------- 입력 기록
def test_provenance_has_all_inputs():
    pr = record.run_record(paths.I1 / "steps_simple.csv", {"n": 1}, 2, {"text_check": "text_check_easyocr.jsonl"})
    names = set(pr["files"])
    for want in ("heading_indoornav.npz", "dongchun_station.json", "dongchun_station.npz", "vision_options.json",
                 "recordings/route2_1030.json", "text_check_easyocr.jsonl", "pf_home_v3.toml", "pdr_tuned_v4.toml"):
        assert any(k.endswith(want) for k in names), want
    assert "integration/run_i1.py" in pr["sources"] and "vision/obs_model.py" in pr["sources"]
    assert any(k.startswith("indoornav/pf/") for k in pr["sources"])
    assert pr["seeds"] == [0, 1] and len(pr["source_digest"]) == 16


def test_provenance_outside_paths_hide_pc_folder(tmp_path):
    f = tmp_path / "밖" / "a.json"
    assert record.rel(f) == "(저장소 밖)/밖/a.json"
    assert record.file_hashes([f]) == {"(저장소 밖)/밖/a.json": None}


def test_sha_ignores_line_endings(tmp_path):
    (tmp_path / "a").write_bytes(b"x\r\ny\r\n")
    (tmp_path / "b").write_bytes(b"x\ny\n")
    assert record.sha(tmp_path / "a") == record.sha(tmp_path / "b")


def test_run_i1_exp_cause_use_same_record():
    src = (ROOT / "integration/run_i1.py").read_text(encoding="utf-8")
    assert "record.run_record" in src
    assert "provenance=record.run_record" in (ROOT / "integration/exp_cause.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 옛 파일 재사용 경고
def test_cache_warning_cases(tmp_path):
    now = dict(model="m1", config="c1", how="앱 검출")
    assert "입력 기록 없음" in record.cache_warning(tmp_path, "cv_raw", now, "관측")
    record.write_record(tmp_path, "cv_raw", now)
    assert record.cache_warning(tmp_path, "cv_raw", now, "관측") is None
    w = record.cache_warning(tmp_path, "cv_raw", dict(now, model="m2"), "관측")
    assert "달라진 입력 model" in w and "--force" in w
    w = record.cache_warning(tmp_path, "cv_raw", dict(now, config=None), "관측")
    assert "비교할 수 없는 입력 config" in w
    record.write_record(tmp_path, "steps_simple", dict(motion="a"))
    assert record.cache_warning(tmp_path, "cv_raw", now, "관측") is None
    assert record.cache_warning(tmp_path, "steps_simple", dict(motion="b"), "걸음") is not None


def test_run_recording_warns_and_reports(tmp_path, capsys):
    import run_recording as rr
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "motion.csv").write_text("t\n1\n")
    (rec / "baro.csv").write_text("t\n1\n")
    (rec / "meta.json").write_text("{}")
    inputs = tmp_path / "inputs"
    now = rr.steps_indoornav_inputs(rec)
    record.write_record(inputs, "steps_indoornav", now)
    rr.WARNINGS.clear()
    rr.warn_cache(inputs, "steps_indoornav", now, "걸음")
    assert rr.WARNINGS == []
    (rec / "motion.csv").write_text("t\n2\n")
    rr.warn_cache(inputs, "steps_indoornav", rr.steps_indoornav_inputs(rec), "걸음")
    assert len(rr.WARNINGS) == 1 and "motion" in rr.WARNINGS[0]
    assert "경고" in capsys.readouterr().out
    rr.WARNINGS.clear()


def test_obs_inputs_track_model_for_video(tmp_path):
    import run_recording as rr
    (tmp_path / "video.mov").write_bytes(b"123")
    a = rr.obs_inputs(tmp_path, dict(detections=False, video="video.mov"))
    assert a["how"] == "PC 검출" and a["video"] and "model" in a
    b = rr.obs_inputs(tmp_path, dict(detections=True, video=None))
    assert b["how"] == "앱 검출" and "detections" in b


# ---------------------------------------------------------------- PC·앱 시간 필터
def _frame(k, t, dets):
    """dets: [(id, 종류, 박스)]"""
    return dict(status="ok", frame_info=dict(capture_time_s=t, image_height=1080),
                detections=[dict(detection_id=i, box_xyxy=b) for i, _, b in dets],
                observations=[dict(detection_id=i, label=lb, score=0.9) for i, lb, _ in dets])


BOX = [100.0, 100.0, 200.0, 200.0]


def _ids(frames):
    return {o["detection_id"] for f in apply(frames) for o in f["observations"]}


def test_app_rule_gap_seconds_vs_frames():
    # 5장 안이지만 1.0 s를 넘는 끊김: PC는 잇고(3번째에 통과) 앱은 새 판으로 봄
    fr = [_frame(0, 0.0, [("a", "exit", BOX)]), _frame(1, 0.2, [("b", "exit", BOX)]), _frame(2, 1.5, [("c", "exit", BOX)])]
    assert "c" in _ids(fr)
    assert "c" not in tf.app_rule(fr)[0]


def test_app_rule_count_by_seconds():
    # 초 단위로 세면 5장 넘게 떨어져도 0.6 s 안 3번이면 통과
    fr = [_frame(k, k * 0.05, [] if 1 <= k <= 5 or 7 <= k <= 11 else [(f"d{k}", "exit", BOX)]) for k in range(13)]
    cfg = json.loads(json.dumps(tf.APP_DEFAULT))
    assert tf.app_rule(fr, cfg)[0] == set()
    cfg["time_filter"]["count_by"] = "seconds"
    assert tf.app_rule(fr, cfg)[0] == {"d12"}


def test_app_rule_tie_goes_to_first_track_pc_to_last():
    # 동점: 앱은 앞 track(>), PC는 뒤 track(>=). 앞 track에만 hit가 쌓인 경우 앱만 통과
    a, b = [0.0, 0.0, 100.0, 100.0], [50.0, 0.0, 150.0, 100.0]
    mid = [25.0, 0.0, 125.0, 100.0]
    fr = [_frame(0, 0.0, [("a0", "exit", a)]), _frame(1, 0.2, [("a1", "exit", a), ("b1", "exit", b)]),
          _frame(2, 0.4, [("m", "exit", mid)])]
    assert "m" in tf.app_rule(fr)[0]
    assert "m" not in _ids(fr)


def test_app_rule_size_limit_and_same_input_pass():
    tall = [0.0, 0.0, 100.0, 800.0]
    fr = [_frame(k, k * 0.2, [(f"x{k}", "area_map", tall)]) for k in range(4)]
    on = dict(tf.APP_DEFAULT, size_limits=dict(enabled=True, max_height_ratio=dict(area_map=0.7)))
    passed, big = tf.app_rule(fr, on)
    assert passed == set() and len(big) == 4
    assert len(_ids(fr)) == 2
    assert tf.app_rule(fr, dict(tf.APP_DEFAULT, size_limits=dict(enabled=False)))[0] == {"x2", "x3"}


def test_app_rule_matches_real_decisions():
    """이식한 규칙이 앱이 실제로 낸 판정과 같아야 대조가 뜻을 가짐. 참고값 8/685"""
    rec = "s2_a_ex6_ex3"
    rc = json.loads((paths.RECORDINGS / f"{rec}.json").read_text(encoding="utf-8"))
    if not (paths.rec_dir(rc, strict=False) / "decisions.jsonl").exists():
        pytest.skip("녹화 자료 없음")
    r = tf.check(rec)
    assert r["rule_vs_app"] == 0
    assert (r["pc_vs_app"], r["pc"]) == (8, 685)


# ---------------------------------------------------------------- 지표 이름
@pytest.mark.parametrize("rec,old,new,n", [("route2_1030", 400, 342, 506), ("s2_b_emart_ex5", 596, 516, 940)])
def test_map_bearing_vs_pf_candidate(rec, old, new, n, tmp_path):
    import subprocess
    code = f"""
import json, numpy as np
import paths, replay as rp, analyze_recording as ar
from edge_faces import add_face_points
from station_pkg import load_station
t0 = paths.rec_t0(); truth = rp.load_truth(paths.I1 / "truth.csv")
fmap = add_face_points(load_station(paths.INPUTS))
h = np.load(paths.REC_OUT / "heading_indoornav.npz"); ty, yaw = h["t_host"] - t0, -np.degrees(np.unwrap(h["heading"]))
off = ar.camera_offset(truth, paths.RC["waypoints"], ty, yaw)
m = ar.obs_match(paths.I1 / "obs_timefilter.jsonl", fmap, truth, t0, ty, yaw, off)
print(json.dumps({{k: m[k] for k in ("n", "map_bearing_hits", "pf_cand_hits", "match", "map_bearing_match")}}))
"""
    rc = json.loads((paths.RECORDINGS / f"{rec}.json").read_text(encoding="utf-8"))
    if not (paths.ROOT / rc["inputs"] / "truth.csv").exists() or not (paths.ROOT / rc["out"] / "heading_indoornav.npz").exists():
        pytest.skip("녹화 자료 없음")
    import os
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=ROOT / "integration",
                       env={**os.environ, "I1_RECORDING": rec, "PYTHONPATH": str(ROOT / "integration"), "PYTHONIOENCODING": "utf-8"})
    assert p.returncode == 0, p.stderr
    m = json.loads(p.stdout.splitlines()[-1])
    assert (m["map_bearing_hits"], m["pf_cand_hits"], m["n"]) == (old, new, n)
    assert m["match"] == m["map_bearing_match"]
