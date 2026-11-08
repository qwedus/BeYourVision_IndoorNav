"""실시간 조건 재생(rt_pdr·rt_replay·run_rt) 시험.

녹화마다 paths가 불러올 때 I1_RECORDING을 읽으므로 실제 검사는 rt_check_helper.py를 기록마다 새 프로세스로 돌리고
(동시에 몇 개씩), 그 결과(JSON)를 여기서 확인한다. 녹화 데이터(저장소 밖 integration/out/rec)가 없으면 건너뛴다.
PF는 입자 500개·seed 1개로 작게 돌린다.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))
import paths  # noqa: E402  (녹화 폴더 찾기 rec_dir)
HELPER = Path(__file__).with_name("rt_check_helper.py")
PY = sys.executable
RECS = ("route2_1030", "s2_a_71_ex2", "s2_a_ex3_ex6", "s2_a_ex6_ex3", "s2_a_platform", "s2_b_elev_pillar",
        "s2_b_emart_ex5", "s2_b_ex2_31", "s2_b_pillar_map")
RIDE = "s2_a_71_ex2"            # 에스컬레이터 탑승 약 51~73 s, 층 바뀜
NORIDE = "s2_b_pillar_map"      # 탑승 없음, 거짓 탑승 의심 여러 번
T_EDGE = 13.52680275            # 격자 값이 T 뒤 원본 표본에 기대는 시각(받은 시각 없이 자르면 w ≤ T 출력 62 대 60행)
T_RIDE = (T_EDGE, 60.0, 76.0, 150.0)    # 걷는 중(경계) / 탑승 중 / 탑승 끝 직후 / 걷는 중
T_VAR = (T_EDGE, 60.0)          # early·최대 기다림 3 s 단계도 자르기 비교
T_REC = (T_EDGE, 60.0, 76.0)    # 권장 단계 rec(early·봉우리 뒤 0.5 s·의심 무시·최대 기다림 5 s)도 자르기 비교

# (검사, 기록, 추가 인자). 모두 한 번에 띄워 결과를 모아 둔다
JOBS = [("oracle", RIDE, ()),
        ("trunc", RIDE, ("--T", ",".join(map(str, T_RIDE)), "--var-T", ",".join(map(str, T_VAR)),
                              "--rec-T", ",".join(map(str, T_REC)))),
        ("trunc", NORIDE, ("--auto-suspect",)),
        ("filter", RIDE, ("--T", "33.333,60,76,150")),
        ("filter", NORIDE, ("--T", "26.25,47.01")),
        ("steps", RIDE, ("--sig",))] + \
       [("steps", r, ()) for r in RECS if r != RIDE] + [("baro", r, ()) for r in RECS] + \
       [("prep_short", NORIDE, ()), ("prep_heading", NORIDE, ("--n", "50"))]
PF_CHECKS = {"oracle", "trunc", "prep_heading"}


def _rc(rec):
    return json.loads((ROOT / "integration/recordings" / f"{rec}.json").read_text(encoding="utf-8"))


def _have(rec, check):
    """그 검사에 필요한 녹화 파일이 있는가"""
    rc = _rc(rec)
    d = paths.rec_dir(rc, strict=False)          # 녹화 폴더: I1_DATA_ROOT 아래 session 이름, 없으면 설정 rec
    need = [d / "motion.csv", d / "baro.csv", d / "meta.json",
            ROOT / rc["out"] / "heading_indoornav.npz"]
    if check in PF_CHECKS:
        need += [ROOT / rc["inputs"] / "obs_timefilter.jsonl", ROOT / rc["inputs"] / "truth.csv",
                 ROOT / rc["out"] / "steps_indoornav.csv"]
    return all(p.exists() for p in need)


def _call(job, tmp):
    check, rec, extra = job
    env = {**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    out = tmp / f"{check}_{rec}"
    p = subprocess.run([PY, str(HELPER), check, "--tmp", str(out), *extra], cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=900)
    line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
    if p.returncode or line is None:
        return dict(error=f"코드 {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
    return json.loads(line[7:])


@pytest.fixture(scope="session")
def results(tmp_path_factory):
    """모든 검사를 동시에 4개씩 돌린 결과 {(검사, 기록): dict}. 데이터 없는 것은 None"""
    tmp = tmp_path_factory.mktemp("rt")
    todo = [j for j in JOBS if _have(j[1], j[0])]
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip([(c, r) for c, r, _ in todo], ex.map(lambda j: _call(j, tmp), todo)))
    return {(c, r): res.get((c, r)) for c, r, _ in JOBS}


def _get(results, check, rec):
    r = results[(check, rec)]
    if r is None:
        pytest.skip(f"{rec} 녹화 데이터 없음")
    assert "error" not in r, r.get("error")
    return r


# ---------------------------------------------------------------- 1. 기준 지식 = run_one
def test_1_oracle_equals_run_one(results):
    """OracleSteps+OracleBaro로 돌린 run_rt가 run_one과 REC_FIELDS 모든 필드 바이트까지 같고, 출력 지연 0(w = 프레임 시각),
    새 필드 t64 = 프레임 시각(f8), w = t64"""
    r = _get(results, "oracle", RIDE)
    assert r["n1"] == r["n2"] == r["n_frames"] > 0
    assert r["bad"] == []
    assert r["w_eq_t"], r["w_minus_t_max"]
    assert r["t64_eq_frames"] and r["w_eq_t64"]
    assert {"w", "ec", "riding_rt", "t64", "href"} <= set(r["fields"])
    assert r["floors"] and r["n_obs_used"] > 0            # 탑승·층 바뀜·관측이 실제로 들어간 기록


# ---------------------------------------------------------------- 2. 자르기: 미래 값 안 씀
def _trunc_rows(results, rec):
    r = _get(results, "trunc", rec)
    assert r["prepare_same"], "시험이 만든 전체 걸음·갱신·방향 받은 시각 입력이 run_rt.prepare 결과와 다름"
    assert r["T"]
    assert {"t64", "href"} <= set(r["fields"])
    assert 0 < r["avail_lag_max"] < 0.05                  # 격자 값은 바로 뒤 원본 표본(약 10 ms 안)을 받아야 정해짐
    return r


@pytest.mark.parametrize("rec", [RIDE, NORIDE])
def test_2_raw_cut_rebuilds_same_prefix(results, rec):
    """원본 motion.csv(호스트 t ≤ T)·baro.csv(t_recv ≤ T)를 잘라 load_app_session으로 다시 만든 Session·받은 시각·방향·갱신이
    전체의 앞부분과 같고, 자른 입력의 받은 시각·갱신 시각이 모두 T 안"""
    r = _trunc_rows(results, rec)
    for d in r["T"]:
        assert d["t0_same"] and d["session_prefix"] and d["baro_rows_same"], d["T"]
        assert d["avail_prefix"] and d["heading_prefix"] and d["ticks_same"], d["T"]
        assert d["avail_max"] <= d["T"] + 1e-9, d["T"]


@pytest.mark.parametrize("rec", [RIDE, NORIDE])
def test_2_truncated_inputs_same_output_until_T(results, rec):
    """모든 입력을 벽시계 T에서 자르고 다시 만든 실시간 입력으로 돌려도 w ≤ T 출력이 행 수·모든 필드까지 같다 (hold·ignore)"""
    r = _trunc_rows(results, rec)
    for d in r["T"]:
        for m in ("hold", "ignore"):
            x = d[m]
            assert x["n_full"] == x["n_cut"] > 0, (d["T"], m, x)
            assert x["bad"] == [], (d["T"], m, x)
            assert x["n_full_after"] > 0                   # 자른 뒤에도 출력이 남는 시각(끝이 아님)
            # step_rows 그대로(판정 대기 걸음은 t_known = nan으로 남음)도 같아야 함
            assert x["drop_n"] == x["n_full"] and x["drop_bad"] == [], (d["T"], m, x)


def test_2_truncated_variants_and_edge(results):
    """early 걸음 행(묶음 판정 안 기다림)·최대 기다림 3 s(hold·ignore)도 자른 입력으로 w ≤ T 출력이 같다. 그 단계들은 기본 단계와
    실제로 다른 출력을 낸다(비교가 의미 있음). 경계 시각 T_EDGE에서 받은 시각 없이(격자 시각으로) 자르면 행 수가 달라진다(음성 대조)"""
    r = _trunc_rows(results, RIDE)
    by_t = {d["T"]: d for d in r["T"]}
    for T in T_VAR:
        v = by_t[T]["var"]
        assert set(v) == {"early_hold", "early_ignore", "w3_hold", "w3_ignore"}
        for nm, x in v.items():
            assert x["n_full"] == x["n_cut"] > 0 and x["bad"] == [], (T, nm, x)
            assert x["n_full_after"] > 0
            assert x["differs_from_base"], (T, nm)
        neg = by_t[T]["neg_avail"]
        if T == T_EDGE:
            assert neg["bad"] and neg["n_full"] != neg["n_cut"] and neg["ticks_extra"] > 0, neg


def test_2_truncated_recommended_stage(results):
    """권장 단계 rec(run_rt.STAGES["rec"]: early 걸음 행·봉우리 뒤 0.5 s·탑승 의심 무시·최대 기다림 5 s)도 자른 입력으로
    w ≤ T 출력이 행 수·모든 필드 바이트까지 같다(걷는 중 경계·탑승 중·층 바뀜 직후). run_rt.prepare(["rec"])가 만든
    steps_early_l050.csv·ticks_l050.npz = 시험 입력, 그 출력은 기본 단계(online 0.75 s·의심 무시)·early_ignore(0.75 s)와 실제로 다름"""
    r = _trunc_rows(results, RIDE)
    assert r["rec_stage"] == dict(steps="early", look=0.5, baro="ignore", max_wait=5.0)
    p = r["rec_prepare"]
    assert p["files"] == ["steps_early_l050.csv", "steps_online_l050.csv", "ticks_l050.npz"], p
    assert p["rows_same"] and p["ticks_same"], p
    assert p["differs_from_l075"]                          # 봉우리 뒤 0.5 s 걸음 행이 0.75 s와 다름
    by_t = {d["T"]: d for d in r["T"]}
    assert set(T_REC) <= set(by_t)
    for T in T_REC:
        x = by_t[T]["rec"]
        assert x["n_full"] == x["n_cut"] > 0, (T, x)
        assert x["bad"] == [], (T, x)
        assert x["n_full_after"] > 0
        assert x["differs_from_base"], T
        assert x["differs_from_early_ignore"], T


@pytest.mark.parametrize("rec", [RIDE, NORIDE])
def test_2_truncated_steps_and_baro_knowledge(results, rec):
    """걸음: t_confirm ≤ T 걸음의 idx·t·t_confirm·length·heading·dheading, t_known ≤ T 걸음의 moving·t_known이 같다.
    기압: 후보마다 w ≤ T에 보이는 지식(의심·확정·끝·층)이 같다"""
    r = _trunc_rows(results, rec)
    for d in r["T"]:
        assert d["steps_confirm_same"] and d["n_confirm"] > 0, d["T"]
        assert d["steps_known_same"] and d["n_known"] > 0, d["T"]
        assert d["baro_same"], d["T"]


def test_2_truncation_points_cover_ride_cases(results):
    """자른 시각이 뜻한 상황인가: 60 s 탑승 확정 전, 76 s 층 바뀜 안 직후, 150 s 걷는 중 / 기둥 기록은 거짓 의심 중"""
    r = _trunc_rows(results, RIDE)
    by_t = {d["T"]: d for d in r["T"]}
    v60 = by_t[60.0]["baro_views"]
    assert len(v60) == 1 and v60[0]["k_confirm"] is None and "k_end" not in v60[0]        # 의심만 앎
    v76 = by_t[76.0]["baro_views"]
    assert v76[0]["ride"] is True and v76[0]["delta"] == 1 and v76[0]["k_end"] <= 76.0      # 층 바뀜 방금 앎
    assert by_t[150.0]["n_pending_cut"] == 0
    q = _trunc_rows(results, NORIDE)["T"]
    assert len(q) == 1
    T, v = q[0]["T"], q[0]["baro_views"]
    assert 24.0 <= T <= 27.0
    assert any(c["k_suspect"] <= T and "k_end" not in c for c in v)                         # 의심 중(아직 취소 안 앎)
    assert all(c.get("ride") is not True for c in v)


# ---------------------------------------------------------------- 3. 음성 대조: 자르기 시험이 새는 것을 잡는가
def test_3_offline_path_leaks_are_detected(results):
    """같은 자르기 비교를 지금 방식(전체 filtfilt 걸음·탑승 끝난 뒤 정하고 2 s 앞당긴 기압)에 하면 T 전에 차이가 난다"""
    r = _trunc_rows(results, RIDE)
    by_t = {d["T"]: d["neg"] for d in r["T"]}
    for T, n in by_t.items():
        assert n["off_steps_diff"] > 0, T                  # 오프라인 걸음 값이 뒤 데이터에 따라 바뀜
        assert n["pf_bad"], T                              # 그 걸음·기압으로 돌린 PF 출력도 T 전에 달라짐
    n60 = by_t[60.0]
    assert n60["baro_diff"] and n60["rides_full"] and not n60["rides_cut"]      # 전체 기록만 60 s에 탑승 구간을 앎
    big = by_t[76.0]["off_big"]                            # 이동 묶음 판정이 76 s 뒤 걸음에 따라 바뀜(보폭 0 ↔ 정상)
    assert any(b["moving"][0] != b["moving"][1] for b in big), big


# ---------------------------------------------------------------- 4. 실시간 걸음 판정 속
@pytest.mark.parametrize("rec", RECS)
def test_4_online_steps_bout_heading_early(results, rec):
    """이동 판정 = indoornav bout_mask, 방향 = heading_indoornav.npz, "early" 행은 모두 이동·t_known = t_confirm"""
    r = _get(results, "steps", rec)
    assert r["n"] > 0
    assert r["check_bout"] == 0
    assert r["heading_same"] and r["t_host_same"]
    assert r["early_n"] == r["early_n_confirmed"]
    assert r["early_all_moving"] and r["early_known_eq"]


def test_4_offline_signal_gives_team_steps(results):
    """오프라인 신호를 넣고 봉우리 뒤 3 s·창 무한대로 돌리면 봉우리 = indoornav run_pdr 걸음(판정이 끝난 범위), moving도 같다"""
    r = _get(results, "steps", RIDE)
    assert r["sig_idx_same"], r
    assert r["n_online"] == r["n_team_in"] > 0
    assert r["team_after"] <= 3                            # 녹화 끝 3.5 s 안 걸음만 판정할 갱신이 없음
    assert r["moving_bad"] == 0 and r["n_moving_decided"] > 0


# ---------------------------------------------------------------- 5. 지연 필터 인과성
@pytest.mark.parametrize("rec", [RIDE, NORIDE])
def test_5_delayed_filter_causal(results, rec):
    """원본을 잘라 다시 만든 Session의 지연 필터 신호가 마지막 갱신(now ≤ 자른 길이) 기준 [0, now - 50) 표본에서 전체 신호와 같다"""
    r = _get(results, "filter", rec)
    assert r["d"] == 50 and r["hop"] == 10
    for c in r["cuts"]:
        assert c["e"] > 0 and c["same"] and c["prefix"], c
    assert any(c["tail_diff"] for c in r["cuts"])         # 그 뒤 표본은 실제로 달라짐(비교가 의미 있음)


# ---------------------------------------------------------------- 6. 실시간 기압 = 오프라인 규칙
@pytest.mark.parametrize("rec", RECS)
def test_6_online_baro_matches_offline_rule(results, rec):
    """ride=True 후보의 끝·층 수 = replay.baro_events, 첫 탑승 표본은 오프라인 시작(2 s 앞당김)보다 0.9 s 이상 뒤, 앎은 받은 뒤"""
    r = _get(results, "baro", rec)
    assert r["causal_order"]
    rides = [c for c in r["cands"] if c["ride"]]
    ended = [(a, b) for a, b in r["rides"] if b < r["t_last"]]         # 녹화 끝까지 안 끝난 탑승은 오프라인만 셈
    if not r["floors"]:                                   # 탑승 없는 기록: 거짓 의심은 모두 취소·층 바뀜 없음
        assert not rides and all(c["delta"] == 0 for c in r["cands"])
        return
    assert len(rides) == len(ended)
    for c, (a, b), (tf, df) in zip(rides, ended, r["floors"]):
        assert c["t_end"] == b == tf
        assert c["delta"] == df
        assert c["t_first"] >= a + 0.9
        assert c["t_first"] <= a + 2.0
        assert c["k_suspect"] >= c["t_first"] and c["k_end"] >= c["t_end"]
        assert c["k_confirm"] is not None and c["k_suspect"] <= c["k_confirm"] <= c["k_end"]


def test_6_ride_recordings_present(results):
    """탑승이 있는 기록(71_ex2·route2_1030)으로 위 비교가 실제로 돌았는가"""
    for rec in (RIDE, "route2_1030"):
        r = _get(results, "baro", rec)
        assert any(c["ride"] for c in r["cands"]), rec


# ---------------------------------------------------------------- 7. run_rt.prepare 짧은 입력·실시간 방향 파일
def test_7_prepare_short_session_no_decided_steps(results):
    """앞 2 s(걸음 없음)·첫 걸음 판정 직후(판정 1개, 이동 판정 전) Session으로 run_rt.prepare가 멈추지 않고,
    이동 판정 통계(known_med·p90·max)는 None, 걸음이 없으면 판정 통계도 None"""
    r = _get(results, "prep_short", NORIDE)
    assert "error_in_prepare" not in r, r.get("error_in_prepare")
    c2, c1 = r["cases"]
    assert c2["sec"] == 2.0 and c2["chk"]["n"] == 0 and c2["chk"]["confirm_med"] is None
    assert c1["chk"]["n"] > 0 and c1["chk"]["undecided"] == c1["chk"]["n"] and c1["chk"]["confirm_med"] is not None
    for c in (c2, c1):
        assert c["chk"]["known_med"] is None and c["chk"]["known_p90"] is None and c["chk"]["known_max"] is None, c


def test_7_heading_rt_is_session_heading(results):
    """heading_indoornav.npz와 방향이 다른 Session(z 자이로 치우침)으로 prepare: heading_rt.npz 방향·시각·받은 시각 = 그 Session의
    online_steps 값, run_rt._init 뒤 실시간 단계(both_ignore) heading_for = 그 파일 (3개), 미리 아는 단계(off) = heading_indoornav (2개)"""
    r = _get(results, "prep_heading", NORIDE)
    assert r["differs_from_indoornav"]                     # 두 파일을 구별할 수 있는 입력
    assert r["file_eq_online"] and r["file_t_eq"] and r["file_avail_eq"], r
    assert r["rt_len"] == 3 and r["rt_eq"], r
    assert r["off_len"] == 2 and r["off_eq"], r
