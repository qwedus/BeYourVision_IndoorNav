"""독립 검토에서 나온 안전장치 결함 시험. 번호는 검토 항목 번호.

재생이 들어가는 시험은 저장소 안 1차 녹화(route2_1030)로 돌린다. 저장소 밖 재생 결과가 필요한 시험은 없으면 건너뛴다.
"""
import json
import os
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import check_tf_app as tf  # noqa: E402
import compare_runs as cr  # noqa: E402
import provenance as record  # noqa: E402
import regress_replay as reg  # noqa: E402
import replay as rp  # noqa: E402
import run_recording as rr  # noqa: E402

KEY = "known_tf_기본_0"
PY = sys.executable


def _run(err=1.0, times=None):
    times = np.arange(3.0, 6.0) if times is None else np.asarray(times, float)
    r = np.zeros(len(times), dtype=rp.REC_FIELDS)
    r["t"], r["err"], r["unc"], r["unc95"] = times, err, 1.0, 2.0
    r["ex"], r["ey"] = 3.0, 4.0
    r["floor"] = r["efloor"] = -1
    return r


def _cli(args, rec="route2_1030", timeout=90):
    """명령을 돌려 (종료 코드, 출력). 시간 안에 안 끝나면 프로세스 묶음을 지우고 코드 None (Pool 작업자가 끝없이 다시 뜨는 경우)"""
    env = {**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.Popen([PY, *map(str, args)], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        out, _ = p.communicate()
        return None, out


# ---------------------------------------------------------------- 1. 저장 경로 = 기준 경로
def _one_args(**kw):
    return Namespace(**{**dict(n=3000, seeds=1, procs=1, save=None, ref=None, ref_dir=None), **kw})


def test_1_save_same_as_ref_dir_refused(tmp_path, monkeypatch):
    f = tmp_path / f"{paths.RECORDING}.npz"
    np.savez(f, **{KEY: _run(1.0)})
    monkeypatch.setattr(reg, "replay", lambda *_: {KEY: _run(9.0)})
    with pytest.raises(SystemExit, match="기준"):
        reg.one(_one_args(save=str(tmp_path), ref_dir=str(tmp_path)))
    assert float(np.load(f)[KEY]["err"][0]) == 1.0        # 기준이 그대로


def test_1_save_same_as_ref_folder_refused(tmp_path, monkeypatch):
    folder = tmp_path / "X"
    folder.mkdir()
    np.savez(folder / "runs.npz", **{KEY: _run(1.0)})
    monkeypatch.setattr(paths, "REC_OUT", tmp_path)
    monkeypatch.setattr(paths, "RECORDING", "runs")
    monkeypatch.setattr(reg, "replay", lambda *_: {KEY: _run(9.0)})
    with pytest.raises(SystemExit, match="기준"):
        reg.one(_one_args(save=str(folder), ref="X"))
    assert float(np.load(folder / "runs.npz")[KEY]["err"][0]) == 1.0


def test_1_different_paths_still_compare_against_old(tmp_path, monkeypatch, capsys):
    (tmp_path / "ref").mkdir()
    np.savez(tmp_path / "ref" / f"{paths.RECORDING}.npz", **{KEY: _run(1.0)})
    monkeypatch.setattr(reg, "replay", lambda *_: {KEY: _run(9.0)})
    reg.one(_one_args(save=str(tmp_path / "new"), ref_dir=str(tmp_path / "ref")))
    res = json.loads(capsys.readouterr().out.split("RESULT ", 1)[1])
    assert res["checks"]["ref_dir"][KEY]["same"] is False
    assert float(np.load(tmp_path / "new" / f"{paths.RECORDING}.npz")[KEY]["err"][0]) == 9.0


# ---------------------------------------------------------------- 2. 전체 필드 비교
def test_2_same_err_unc_but_moved_is_different(tmp_path):
    old = _run(5.0)
    new = old.copy()
    new["ex"] = -3.0           # 정답에서 거리는 같고 반대쪽
    new["efloor"] = -2
    new["t"] += 100
    np.savez(tmp_path / "a.npz", **{KEY: old})
    d = reg.diff({KEY: new}, tmp_path / "a.npz")[KEY]
    assert d["same"] is False
    assert set(d["fields"]) == {"ex", "efloor", "t"}


def test_2_identical_including_nan_is_same(tmp_path):
    old = _run(5.0)
    old["m2x"] = np.nan
    np.savez(tmp_path / "a.npz", **{KEY: old})
    d = reg.diff({KEY: old.copy()}, tmp_path / "a.npz")[KEY]
    assert d["same"] is True and d["fields"] == []


def test_2_different_dtype_or_length_is_different(tmp_path):
    old = _run(5.0)
    np.savez(tmp_path / "a.npz", **{KEY: old})
    assert reg.diff({KEY: _run(5.0, np.arange(3.0, 9.0))}, tmp_path / "a.npz")[KEY]["same"] is False
    other = np.zeros(3, dtype=[("t", "f8"), ("err", "f4"), ("unc", "f4")])
    assert reg.diff({KEY: other}, tmp_path / "a.npz")[KEY]["same"] is False


# ---------------------------------------------------------------- 3. 자신 있게 틀린 시간과 채점 제외
def test_3_wrong_seconds_skips_excluded_time(monkeypatch):
    monkeypatch.setattr(paths, "RC", dict(exclude=[[6, 20]], end_start=1e9))
    assert cr.seed_metrics(_run(5.0, [3, 4, 5, 20, 21, 22]))["wrong_s"] == pytest.approx(5.0)


def test_3_wrong_seconds_no_exclusion_unchanged(monkeypatch):
    monkeypatch.setattr(paths, "RC", dict(exclude=[], end_start=1e9))
    assert cr.seed_metrics(_run(5.0, [3, 4, 5, 20, 21, 22]))["wrong_s"] == pytest.approx(19.0)


def test_3_wrong_seconds_overlapping_exclusions_count_once(monkeypatch):
    monkeypatch.setattr(paths, "RC", dict(exclude=[[6, 12], [10, 20]], end_start=1e9))
    assert cr.seed_metrics(_run(5.0, [3, 4, 5, 20, 21, 22]))["wrong_s"] == pytest.approx(5.0)


# ---------------------------------------------------------------- 4. PC 영상 경로 입력
@pytest.fixture
def video_rec(tmp_path):
    (tmp_path / "video.mov").write_bytes(b"old video")
    (tmp_path / "frames.jsonl").write_text('{"K":[[100,0,50]]}\n', encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    return tmp_path


INFO = dict(detections=False, video="video.mov")


@pytest.mark.parametrize("name,new,key", [("frames.jsonl", b'{"K":[[200,0,50]]}\n', "frames"), ("meta.json", b'{"a":1}', "meta"), ("video.mov", b"new video", "video")])
def test_4_video_path_detects_input_change(video_rec, name, new, key):
    old = rr.obs_inputs(video_rec, INFO)
    record.write_record(video_rec, "cv_raw", old)
    assert record.cache_warning(video_rec, "cv_raw", rr.obs_inputs(video_rec, INFO), "관측") is None
    (video_rec / name).write_bytes(new)
    w = record.cache_warning(video_rec, "cv_raw", rr.obs_inputs(video_rec, INFO), "관측")
    assert w is not None and key in w


def test_4_big_file_signature_streams_binary(tmp_path):
    (tmp_path / "a").write_bytes(b"x\r\ny")
    (tmp_path / "b").write_bytes(b"x\ny")
    assert record.binary_sig(tmp_path / "a") != record.binary_sig(tmp_path / "b")
    assert record.binary_sig(tmp_path / "없음") is None


# ---------------------------------------------------------------- 5. 시간 원점 입력
def _record():
    return record.run_record(paths.I1 / "steps_simple.csv", {"n": 1}, 1)


def test_5_meta_change_shows_in_record(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"recording": {"first_t": 100}}), encoding="utf-8")
    monkeypatch.setattr(paths, "REC", tmp_path)
    a = _record()
    (tmp_path / "meta.json").write_text(json.dumps({"recording": {"first_t": 101}}), encoding="utf-8")
    b = _record()
    assert a["files"] != b["files"]
    assert (a["rec_t0"], b["rec_t0"]) == (100.0, 101.0)
    assert any(k.endswith("meta.json") for k in a["files"])


def test_5_frames_fallback_shows_in_record(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "frames.jsonl").write_text('{"t": 50}\n', encoding="utf-8")
    monkeypatch.setattr(paths, "REC", tmp_path)
    a = _record()
    (tmp_path / "frames.jsonl").write_text('{"t": 51}\n', encoding="utf-8")
    b = _record()
    assert a["files"] != b["files"] and (a["rec_t0"], b["rec_t0"]) == (50.0, 51.0)


def test_5_missing_time_origin_does_not_stop_record(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REC", tmp_path)
    assert _record()["rec_t0"] is None


# ---------------------------------------------------------------- 6. 설정 비교
def _result(path, info):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "runs.npz", **{KEY: _run()})
    (path / "summary.json").write_text(json.dumps(dict(info=info)), encoding="utf-8")


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(paths, "RC", dict(exclude=[], end_start=1e9))
    monkeypatch.setattr(cr, "use_recording", lambda name: None)


PROV = dict(files={"지도": "같음"}, source_digest="같음")


def test_6_one_sided_setting_shown(tmp_path, plain):
    _result(tmp_path / "A", dict(max_range=30, provenance=PROV))
    _result(tmp_path / "B", dict(provenance=PROV))
    res = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"))
    assert res["inputs"]["only_a"] == {"max_range": 30} and res["inputs"]["only_b"] == {}
    assert "max_range" in "\n".join(cr.render(res))


def test_6_pf_params_difference_shown(tmp_path, plain):
    _result(tmp_path / "A", dict(provenance=dict(PROV, pf_params=dict(sigma_h0=0.1, n=3000))))
    _result(tmp_path / "B", dict(provenance=dict(PROV, pf_params=dict(sigma_h0=0.5, n=3000))))
    res = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"))
    assert res["inputs"]["config"] == {"pf_params.sigma_h0": (0.1, 0.5)}
    assert "pf_params.sigma_h0" in "\n".join(cr.render(res))


def test_6_same_settings_show_no_difference(tmp_path, plain):
    info = dict(max_range=30, provenance=dict(PROV, pf_params=dict(n=3000)))
    _result(tmp_path / "A", info)
    _result(tmp_path / "B", info)
    inp = cr.compare("x", str(tmp_path / "A"), str(tmp_path / "B"))["inputs"]
    assert inp["config"] == {} and inp["only_a"] == {} and inp["only_b"] == {}


# ---------------------------------------------------------------- 7. Pool 초기화 오류
def _boom():
    raise ValueError("초기화 시험 오류")


def _ok():
    pass


def _double(x):
    return 2 * x


def test_7_run_pool_returns_results():
    from pool_guard import run_pool
    assert run_pool(2, _ok, (), _double, [1, 2, 3]) == [2, 4, 6]


def test_7_run_pool_raises_init_error():
    from pool_guard import run_pool
    with pytest.raises(RuntimeError, match="초기화 시험 오류"):
        run_pool(2, _boom, (), _double, [1, 2, 3])


def test_7_run_i1_bad_decision_file_ends_with_error(tmp_path):
    bad = tmp_path / "wrong.jsonl"
    bad.write_text('{"engine": {"name": "시험"}}\n{"detection_id": "없는 관측", "decision": "reject", "text_status": "mismatch"}\n', encoding="utf-8")
    code, out = _cli(["integration/run_i1.py", "--name", "_test_bad", "--text-check", bad, "--seeds", "1", "--n", "100", "--procs", "2"])
    assert code not in (None, 0), f"코드 {code}: 끝나지 않거나 성공으로 끝남"
    assert "안 이어짐" in out
    assert not (paths.REC_OUT / "_test_bad").exists()


def test_7_exp_cause_bad_station_ends_with_error(tmp_path):
    code, out = _cli(["integration/exp_cause.py", "--name", "_test_bad", "--station", tmp_path / "없는 폴더", "--seeds", "1", "--procs", "2"])
    try:
        assert code not in (None, 0), f"코드 {code}: 끝나지 않거나 성공으로 끝남"
    finally:
        import shutil
        shutil.rmtree(paths.REC_OUT / "_test_bad", ignore_errors=True)


def test_7_no_runner_uses_pool_directly():
    bad = [f.name for f in (ROOT / "integration").glob("*.py") if f.name != "pool_guard.py" and re.search(r"\bPool\(", f.read_text(encoding="utf-8"))]
    assert bad == []


# ---------------------------------------------------------------- 8. 앱 대체 설정
APP_CONFIG = ROOT.parent / "ocr_test_app/App/Config/filter_config.json"


def test_8_fallback_matches_app_size_limit():
    assert tf.APP_DEFAULT["size_limits"]["enabled"] is False
    tall = [0.0, 0.0, 100.0, 800.0]
    frames = [dict(frame_info=dict(capture_time_s=k * 0.2, image_height=1080), detections=[dict(detection_id=f"t{k}", box_xyxy=tall)],
                   observations=[dict(detection_id=f"t{k}", label="area_map")]) for k in range(4)]
    assert tf.app_rule(frames)[0] == {"t2", "t3"}


def test_8_fallback_equals_app_file():
    if not APP_CONFIG.exists():
        pytest.skip("앱 저장소 없음")
    app = json.loads(APP_CONFIG.read_text(encoding="utf-8"))
    tfc, d = app["time_filter"], tf.APP_DEFAULT["time_filter"]
    for k in ("enabled", "count_by", "window_frames", "window_s", "min_hits", "track_max_gap_s"):
        assert tfc[k] == d[k], k
    for k in ("same_label", "iou_min", "center_dist_max"):
        assert tfc["match"][k] == d["match"][k], k
    assert app["size_limits"]["enabled"] == tf.APP_DEFAULT["size_limits"]["enabled"]
    assert app["size_limits"]["max_height_ratio"] == tf.APP_DEFAULT["size_limits"]["max_height_ratio"]


def test_8_first_walk_value_is_marked_fallback(capsys, monkeypatch):
    r = tf.check("route2_1030")
    assert (r["pc"], r["rule"], r["pc_vs_rule"], r["config"]) == (599, 601, 2, "대체 설정")
    monkeypatch.setattr(sys, "argv", ["check_tf_app.py", "route2_1030"])
    tf.main()
    assert "대체 설정" in capsys.readouterr().out


# ---------------------------------------------------------------- 9. PC 경로
def test_9_no_pc_path_in_code():
    pat = re.compile(r"[A-Za-z]:[\\/]+Users|[\\/]Users[\\/]+\w+[\\/]")
    bad = [str(f.relative_to(ROOT)) for d in ("integration", "vision", "tests/vision_integration") for f in (ROOT / d).glob("*.py") if pat.search(f.read_text(encoding="utf-8"))]
    assert bad == []


def test_9_vision_python_default_same_place_and_env_wins():
    want = str(Path.home() / "AppData/Local/Programs/Python/Python312/python.exe")
    env = {**{k: v for k, v in os.environ.items() if k != "I1_VISION_PY"}, "PYTHONIOENCODING": "utf-8"}
    code = "import sys; sys.path.insert(0, 'integration'); import paths, run_recording as r; print(paths.VISION_PY); print(r.VISION_PY)"
    out = subprocess.run([PY, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    assert Path(out[0]) == Path(want) and Path(out[1]) == Path(want)
    out = subprocess.run([PY, "-c", code], cwd=ROOT, env={**env, "I1_VISION_PY": "다른파이썬"}, capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    assert out[:2] == ["다른파이썬", "다른파이썬"]


# ---------------------------------------------------------------- 10. 음성 대조
def _obs(i, tid):
    return dict(detection_id=i, track_id=tid)


def test_10_random_pick_is_repeatable_and_needs_no_truth():
    import run_negative as neg
    pool = {f"d{i}" for i in range(100)}
    a = neg.pick_random(pool, 10, 12345)
    assert a == neg.pick_random(pool, 10, 12345) and len(a) == 10 and a <= pool
    assert a != neg.pick_random(pool, 10, 1)
    assert neg.pick_random({"a"}, 10, 1) == {"a"}


def test_10_add_only_keeps_base_and_limits_repeats():
    import run_negative as neg
    lim = neg.AddOnly({"new1", "new2", "new3"}, sched={1.0: {"b"}}, times={7: [1.0]}, limit=1.0)
    # 기존 관측은 그대로, 같은 track 기존 입력과 1초 안이면 안 넣음
    assert [o["detection_id"] for o in lim(1.0, [_obs("b", 7), _obs("new1", 7)])] == ["b"]
    # 다른 track 후보는 넣고, 같은 track은 1초에 한 번
    assert [o["detection_id"] for o in lim(5.0, [_obs("new1", 8)])] == ["new1"]
    assert lim(5.5, [_obs("new2", 8)]) == []
    assert [o["detection_id"] for o in lim(6.0, [_obs("new3", 8)])] == ["new3"]
    assert lim.add == 2
    # 후보가 아니면 안 넣음
    assert lim(9.0, [_obs("없음", 9)]) == []


def test_10_negative_results_go_into_compare(tmp_path):
    code, out = _cli(["integration/run_negative.py", "--cond", "base", "random", "--seeds", "2", "--procs", "2", "--out-dir", tmp_path], timeout=400)
    assert code == 0, out
    for c in ("base", "random"):
        s = json.loads((tmp_path / f"neg_{c}" / "summary.json").read_text(encoding="utf-8"))
        assert s["info"]["negative"] == c and s["info"]["provenance"]["source_digest"]
    assert s["info"]["added_median"] > 0
    z = np.load(tmp_path / "neg_base" / "runs.npz")
    assert sorted(z.files) == ["known_tf_기본_0", "known_tf_기본_1"]
    ref = paths.REC_OUT / "exp_d58_r20/runs.npz"
    if ref.exists():    # 대조 없는 기본 경로는 기존 재생과 비트까지 같아야 함
        assert all(reg.diff({k: z[k]}, ref)[k]["same"] for k in z.files)
    code, out = _cli(["integration/compare_runs.py", "--rec", "route2_1030", "--a", tmp_path / "neg_base", "--b", tmp_path / "neg_random"])
    assert code == 0 and "neg_random" in out
