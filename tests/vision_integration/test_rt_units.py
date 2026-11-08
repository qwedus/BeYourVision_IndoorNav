"""실시간 조건 재생의 작은 단위 시험 (같은 프로세스, 손으로 만든 입력·가짜 PF).

    절대 고도 인과성(read_baro_abs_rt·online_baro, 원본 파일을 받은 시각으로 자름 — 녹화 데이터 없으면 건너뜀)
    now_errors 앞당김: 화면 시각 뒤에 받은 방향 표본을 안 씀, PF 방향 기준 href로 회전
    run_rt: 탑승으로 모은 회전을 관측 전에 넣음, 관측 회전 보정 = 관측 시각 방향 − 마지막으로 꺼낸 걸음 방향(늦은 관측 포함)
    refuse_overwrite: 비교 기준·다른 실행기 결과를 덮어쓰지 않음
    online_baro: 끝 표본에서 처음 3 m를 넘으면 확정 = 끝 수신
    G(회귀): t64로 지연·화면 위치, 판정 시각 전체 자릿수, cut_t 반올림 맞춤, 받은 방향 표본만, 앞당김 시작 = ts,
             summary.json만 있어도 덮어쓰기 막기, 실시간 아닌 PDR 설정 거부
"""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration"))

import paths  # noqa: E402
import rt_pdr  # noqa: E402
import rt_replay as rr  # noqa: E402

TRUTH_DT = [("t", "f8"), ("east", "f8"), ("north", "f8"), ("heading_deg", "f8"), ("on_esc", "?"), ("floor", "i1")]


def _truth(t, east=None, north=None):
    t = np.asarray(t, float)
    a = np.zeros(len(t), dtype=TRUTH_DT)
    a["t"] = t
    a["east"] = 0.0 if east is None else east
    a["north"] = 0.0 if north is None else north
    a["floor"] = -1
    return a


# ---------------------------------------------------------------- B. 절대 고도 인과성
ABS_RECS = ("s2_b_emart_ex5", "s2_a_71_ex2")


def _rec_dir(rec):
    rc = json.loads((ROOT / "integration/recordings" / f"{rec}.json").read_text(encoding="utf-8"))
    return paths.rec_dir(rc, strict=False)       # I1_DATA_ROOT 아래 session 이름, 없으면 설정 rec


def _first_t(d):
    return float(json.loads((d / "meta.json").read_text(encoding="utf-8"))["recording"]["first_t"])


def _cut_abs(src, dst, Th):
    """baro_abs.csv 원본 줄을 글자 그대로, 받은 시각 t_recv ≤ Th(호스트)인 줄만"""
    with open(src, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    head = next(csv.reader([lines[0]]))
    k = head.index("t_recv")
    keep = [ln for ln in lines[1:] if ln.strip() and float(next(csv.reader([ln]))[k]) <= Th]
    dst.write_text(lines[0] + "".join(keep), encoding="utf-8", newline="")
    return len(keep)


def _baro_view(c, T):
    """w ≤ T에 보이는 후보 지식 (rt_check_helper._baro_view와 같음)"""
    v = dict(t_first=c["t_first"], k_suspect=c["k_suspect"])
    v["k_confirm"] = c["k_confirm"] if c["k_confirm"] is not None and c["k_confirm"] <= T else None
    if c["k_end"] is not None and c["k_end"] <= T:
        v.update(t_end=c["t_end"], k_end=c["k_end"], ride=c["ride"], delta=c["delta"])
    return v


def _abs_cases():
    out = []
    for rec in ABS_RECS:
        for T in (60.0, 120.0):
            out.append(pytest.param(rec, T, id=f"{rec}-{T:g}"))
    return out


@pytest.mark.parametrize("rec,T", _abs_cases())
def test_b_abs_altitude_causal(rec, T, tmp_path):
    """받은 시각 ≤ T인 원본 줄만 남긴 baro_abs.csv로 read_baro_abs_rt를 부르면 그 행들의 (표본 시각 추정, 고도, 받은 시각)이
    전체 파일 결과의 앞부분과 비트까지 같고, online_baro 후보의 w ≤ T 지식도 같다"""
    d = _rec_dir(rec)
    src = d / "baro_abs.csv"
    if not src.exists() or not (d / "meta.json").exists():
        pytest.skip(f"{rec} 절대 고도 데이터 없음")
    t0 = _first_t(d)
    full = rt_pdr.read_baro_abs_rt(src, t0)
    n = _cut_abs(src, tmp_path / "baro_abs.csv", T + t0)
    cut = rt_pdr.read_baro_abs_rt(tmp_path / "baro_abs.csv", t0)
    assert n == len(cut) == int((full[:, 2] <= T).sum()) > 0
    assert len(full) > len(cut)                            # 자른 뒤 표본이 남음(끝이 아님)
    assert np.array_equal(cut, full[:len(cut)])
    cf, _ = rt_pdr.online_baro(full)
    cc, _ = rt_pdr.online_baro(cut)
    fv = [_baro_view(c, T) for c in cf if c["k_suspect"] <= T]
    cv = [_baro_view(c, T) for c in cc if c["k_suspect"] <= T]
    assert fv == cv


def test_b_abs_offset_needs_running_min():
    """음성 대조: 표본 시각 보정을 전체 파일 최솟값(t_recv - t)으로 하면 emart_ex5 60 s 자르기에서 앞 표본 시각이 달라진다
    (위 시험이 미래 표본을 쓰는 방식을 잡을 수 있음)"""
    d = _rec_dir("s2_b_emart_ex5")
    if not (d / "baro_abs.csv").exists():
        pytest.skip("절대 고도 데이터 없음")
    t0 = _first_t(d)
    with open(d / "baro_abs.csv", encoding="utf-8") as f:
        a = np.array([(float(r["t"]), float(r["t_recv"]) - t0) for r in csv.DictReader(f)])
    n = int((a[:, 1] <= 60.0).sum())
    glob_full = a[:n, 0] + np.min(a[:, 1] - a[:, 0])
    glob_cut = a[:n, 0] + np.min(a[:n, 1] - a[:n, 0])
    assert not np.array_equal(glob_full, glob_cut)


# ---------------------------------------------------------------- C. now_errors 인과성·기준
def _rt_rows(rows):
    r = np.zeros(len(rows), dtype=rr.RT_FIELDS)
    for i, x in enumerate(rows):
        for k, v in x.items():
            r[k][i] = v
        r["t"][i] = x["t64"]
        if "ts" not in x:                 # PF 상태 시각: 따로 안 주면 촬영 시각(관측이 제때 온 경우)
            r["ts"][i] = x["t64"]
    return r


def _step(t, L, heading_deg=0.0):
    return dict(t=t, length_m=L, length_raw=L, t_confirm=t, t_known=t, heading_deg=heading_deg)


def test_c_now_errors_ignores_heading_received_after_display():
    """화면 시각 1.5 s, 마지막으로 받은 방향 표본 1.4 s: 1.6 s에 받은 표본(1.6 s)을 더해도 1.5 s 앞당긴 오차가 같다.
    음성 대조: 같은 표본을 1.45 s에 받았다고 하면 달라진다"""
    r = _rt_rows([dict(t64=1.0, w=1.0, ex=0.0, ey=0.0, ec=0.0, href=0.0),
                  dict(t64=1.5, w=2.0, ex=5.0, ey=5.0, ec=0.0, href=0.0)])
    truth = _truth([0.0, 3.0], north=[0.0, 3.0])
    steps = [_step(0.9, 1.4)]
    ht, hh = np.array([0.0, 1.0, 1.4]), np.array([0.0, 0.0, 0.2])
    base = rr.now_errors(r, truth, steps, 0.0, (ht, hh, ht))
    ht2, hh2 = np.r_[ht, 1.6], np.r_[hh, 1.5]
    later = rr.now_errors(r, truth, steps, 0.0, (ht2, hh2, ht2))
    early = rr.now_errors(r, truth, steps, 0.0, (ht2, hh2, np.r_[ht, 1.45]))
    assert np.isfinite(base[1]).all()
    assert base[1][1] > 0                                  # 실제로 앞당김(행 1은 행 0 위치에서 0.5 s)
    assert np.array_equal(base[0], later[0]) and np.array_equal(base[1], later[1])
    assert base[1][1] != early[1][1]


def test_c_now_errors_uses_pf_heading_reference():
    """마지막으로 넣은 걸음 방향 0, 자이로가 출력 프레임 전 0.8 s에 90°(indoornav +, 나침반 서쪽) 돌고 그 뒤 그대로,
    PF 방향 ec = 0(아직 안 돎), href = 0 → 0.5 s 앞당김은 서쪽 1 m = 정답(오차 0).
    href를 출력 시각 자이로 방향(90°)으로 두면(고치기 전 방식) 북쪽으로 가서 √2 m"""
    rows = [dict(t64=1.0, w=1.0, ex=0.0, ey=0.0, ec=0.0, href=0.0),
            dict(t64=1.5, w=2.0, ex=9.0, ey=9.0, ec=0.0, href=0.0)]
    truth = _truth([0.0, 1.0, 1.5, 3.0], east=[0.0, 0.0, -1.0, -1.0])
    steps = [_step(0.2, 2.0), _step(0.4, 2.0)]               # 앞 2 s 보폭 합 4 m → 2 m/s
    ht = np.array([0.0, 0.5, 0.8, 3.0])
    hh = np.array([0.0, 0.0, np.pi / 2, np.pi / 2])
    e_now, e_dr = rr.now_errors(_rt_rows(rows), truth, steps, 0.0, (ht, hh, ht))
    assert e_dr[1] == pytest.approx(0.0, abs=1e-9)
    assert e_now[1] == pytest.approx(1.0)                  # 보여 준 위치(행 0)는 1 m 뒤
    rows[0]["href"] = np.pi / 2
    _, e_old = rr.now_errors(_rt_rows(rows), truth, steps, 0.0, (ht, hh, ht))
    assert e_old[1] == pytest.approx(np.sqrt(2.0), abs=1e-9)


# ---------------------------------------------------------------- D. run_rt 회전 모으기·늦은 관측
class FakeFilter:
    """run_rt가 부르는 순서를 적는 가짜 PF"""

    def __init__(self):
        self.calls = []

    def predict(self, L, dh, moving):
        self.calls.append(("predict", L, dh, moving))

    def rotate(self, d):
        self.calls.append(("rotate", d))

    def update(self, obs, dpsi=None):
        self.calls.append(("update", obs[0]["id"], dpsi))

    def change_floor(self, d):
        self.calls.append(("floor", d))

    def estimate(self):
        return dict(x=0.0, y=0.0, unc=1.0, unc95=2.0, compass=0.0, floor=-1, k=1.0, k_std=0.0, ess=100.0,
                    pdr_suspect=False, m1=(0.0, 0.0), m1_mass=1.0, m2=(0.0, 0.0), m2_mass=0.0)


def _rstep(t, heading_rad, dheading_rad, L=0.7):
    """재생 입력 걸음(t 호스트 = 재생 시각, t0 = 0). heading_deg는 나침반 부호(= −indoornav)"""
    return dict(t=t, length_m=L, dheading_rad=dheading_rad, moving=True, heading_deg=-np.degrees(heading_rad))


@pytest.mark.parametrize("with_avail", [False, True])
def test_d_held_rotation_applied_before_used_obs(with_avail):
    """탑승 의심(hold) 중 기다림 한도로 넣은 90° 걸음은 탑승으로 분류돼 회전만 모이고, 의심이 취소된 뒤 처음 쓰는 관측 앞에
    rotate(모은 회전)가 먼저 들어간다. 관측 회전 보정 = 관측 시각 자이로 방향 − 마지막으로 꺼낸 걸음 방향 = 90° − 90° = 0.
    탑승 중 관측은 안 쓰고, href는 모은 회전을 뺀 PDR 방향(0 → 90°)"""
    steps = [_rstep(1.0, 0.0, 0.0), _rstep(2.0, np.pi / 2, np.pi / 2)]
    cand = dict(t_first=1.5, k_suspect=1.5, k_confirm=None, t_end=8.0, k_end=9.0, ride=False, delta=0)
    bk = rr.OnlineBaro([cand], dict(t=np.array([1.5, 8.0]), t_recv=np.array([1.5, 9.0])), suspect="hold")
    sk = rr.OracleSteps(steps, 0.0)
    obs = [(3.0, [dict(id="o3")]), (10.0, [dict(id="o10")])]
    ht = np.array([0.0, 1.9, 2.1, 20.0])
    hh = np.array([0.0, 0.0, np.pi / 2, np.pi / 2])
    F, log = FakeFilter(), {}
    r = rr.run_rt(F, obs, sk, bk, _truth([0.0, 100.0]), 0.0, [], heading=(ht, hh, ht) if with_avail else (ht, hh),
                  max_wait=5.0, log=log)
    assert [c[0] for c in F.calls] == ["predict", "rotate", "update"]
    assert F.calls[1][1] == pytest.approx(np.pi / 2)
    assert F.calls[2][1] == "o10" and F.calls[2][2] == pytest.approx(0.0, abs=1e-9)
    assert log["held_lost"] == 1 and log["forced"] >= 1
    assert list(r["riding_rt"]) == [True, False] and list(r["n_obs"]) == [0, 1]
    assert np.allclose(r["href"], [0.0, np.pi / 2])
    assert list(r["t64"]) == [3.0, 10.0] and r["w"][0] == pytest.approx(8.0) and r["w"][1] == pytest.approx(10.0)


@pytest.mark.parametrize("with_avail", [False, True])
def test_d_late_obs_after_later_step_gets_negative_correction(with_avail):
    """5 s의 90° 걸음을 넣은 뒤 1 s에 찍은 관측이 11 s(obs_lag 10 s)에 오면 늦은 사건으로 바로 넣고,
    회전 보정 = 방향(1 s) − 마지막으로 꺼낸 걸음 방향(90°) = −90°"""
    steps = [_rstep(0.5, 0.0, 0.0), _rstep(5.0, np.pi / 2, np.pi / 2)]
    ht = np.array([0.0, 4.9, 5.1, 20.0])
    hh = np.array([0.0, 0.0, np.pi / 2, np.pi / 2])
    F, log = FakeFilter(), {}
    r = rr.run_rt(F, [(1.0, [dict(id="o1")])], rr.OracleSteps(steps, 0.0), rr.OracleBaro([], []), _truth([0.0, 100.0]),
                  0.0, [], heading=(ht, hh, ht) if with_avail else (ht, hh), obs_lag=10.0, log=log)
    assert [c[0] for c in F.calls] == ["predict", "predict", "update"]
    assert F.calls[2][2] == pytest.approx(-90.0)
    assert log["late_other"] == 1
    assert len(r) == 1 and r["t64"][0] == 1.0 and r["w"][0] == pytest.approx(11.0)
    assert r["href"][0] == pytest.approx(np.pi / 2)


# ---------------------------------------------------------------- E. 덮어쓰기 막기
@pytest.fixture(scope="module")
def run_rt():
    import run_rt as m
    return m


def test_e_refuse_overwrite_check_ref(run_rt):
    """출력 이름 = --check-ref → 쓰기 전에 멈춤 (실제 파일은 안 읽고 안 씀)"""
    import paths
    a = SimpleNamespace(name="exp_d58_r20", check_ref="exp_d58_r20")
    with pytest.raises(SystemExit):
        run_rt.refuse_overwrite(paths.REC_OUT / a.name, a)


def _npz(d, keys):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "runs.npz", **{k: np.zeros(1) for k in keys})
    return d


def test_e_refuse_overwrite_foreign_runs(run_rt, tmp_path):
    """runs.npz에 실시간 단계 키가 아닌 키(기준 재생 known_tf_기본_0 등)가 있으면 멈춤, 단계_seed 키만 있거나 파일이 없으면 통과"""
    a = SimpleNamespace(name="x", check_ref="exp_d58_r20")
    with pytest.raises(SystemExit):
        run_rt.refuse_overwrite(_npz(tmp_path / "ref", ["known_tf_기본_0"]), a)
    with pytest.raises(SystemExit):
        run_rt.refuse_overwrite(_npz(tmp_path / "mix", ["off_0", "foo_1"]), a)
    run_rt.refuse_overwrite(_npz(tmp_path / "ok", ["off_0", "both_ignore_3"]), a)
    run_rt.refuse_overwrite(tmp_path / "empty", a)
    run_rt.refuse_overwrite(tmp_path / "ok", SimpleNamespace(name="x", check_ref=""))


# ---------------------------------------------------------------- F. 끝 표본에서 처음 3 m
def _baro(z):
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0])
    return np.column_stack([t, np.asarray(z, float), t + 0.3])


def test_f_confirm_at_end_sample():
    """탑승 중 누적 2.99 m, 처음 탑승 아닌 표본(끝)이 3.01 m → ride, 층 +1, 확정 = 끝 수신. OnlineBaro 층 바뀜도 끝 수신에 앎"""
    cands, smp = rt_pdr.online_baro(_baro([0, 0, 0, 0, 0.5, 2.99, 3.01, 3.01]))
    assert list(smp["riding"]) == [False] * 4 + [True, True, False, False]
    assert len(cands) == 1
    c = cands[0]
    assert c["t_first"] == 4.0 and c["t_end"] == 10.0
    assert c["ride"] is True and c["delta"] == 1
    assert c["k_confirm"] == c["k_end"] == pytest.approx(10.3)
    bk = rr.OnlineBaro(cands, smp)
    assert bk.floors == [(10.0, 1, c["k_end"])]
    assert bk._state(c, 10.2) == "suspect" and bk._state(c, 10.3) == "ride"


def test_f_confirm_contrasts():
    """대조: 끝이 2.99 m면 취소(확정 없음·층 그대로), 탑승 중에 3 m를 넘으면 확정이 끝보다 앞"""
    c = rt_pdr.online_baro(_baro([0, 0, 0, 0, 0.5, 2.5, 2.99, 2.99]))[0][0]
    assert c["ride"] is False and c["delta"] == 0 and c["k_confirm"] is None
    c = rt_pdr.online_baro(_baro([0, 0, 0, 0, 0.5, 3.2, 3.3, 3.3]))[0][0]
    assert c["ride"] is True and c["k_confirm"] == pytest.approx(5.3) and c["k_confirm"] < c["k_end"]


# ---------------------------------------------------------------- G. 고친 것이 되돌아가면 실패하는 회귀 시험
def test_g1_now_errors_and_delay_use_t64():
    """프레임 시각 1.00000003(f4로는 1.0)이고 w = t64인 행: now_errors는 그 행을 골라 오차 0,
    rt_metrics 출력 지연은 모든 행에서 정확히 0. 음성 대조: t64 필드를 빼서 f4 t로 계산하면(고치기 전 방식)
    앞 행을 골라 오차 1 m, 지연은 0이 아님"""
    from numpy.lib import recfunctions as rfn
    tk = 1.00000003
    assert float(np.float32(tk)) != tk                     # f4로 못 나타내는 시각
    r = _rt_rows([dict(t64=0.5, w=0.5, ex=0.0, ey=0.0),
                  dict(t64=tk, w=tk, ex=1.0, ey=0.0),
                  dict(t64=5.0, w=5.0, ex=1.0, ey=0.0)])
    assert r["t"][1] < r["w"][1]                           # f4 t는 벽시계보다 앞
    truth = _truth([0.0, 0.5, 0.6, 10.0], east=[0.0, 0.0, 1.0, 1.0])
    e_now, _ = rr.now_errors(r, truth)
    assert np.isfinite(e_now).all()
    assert e_now[1] == 0.0
    m = rr.rt_metrics(r, truth)
    assert m["delay_med"] == 0.0 and m["delay_max"] == 0.0 and m["delay_p90"] == 0.0
    old = rfn.drop_fields(r, ["t64"], usemask=False)       # 고치기 전 방식: f4 t로 비교
    e_old, _ = rr.now_errors(old, truth)
    assert e_old[1] == pytest.approx(1.0)                  # 앞 행(위치 0)을 고름
    assert rr.rt_metrics(old, truth)["delay_max"] > 0


def test_g2_confirm_known_times_full_precision(tmp_path):
    """write_rows → read_rows로 t_confirm·t_known(소수 6자리 넘는 값, nan)이 비트까지 그대로. 음성 대조: 소수 6자리로 쓰면 달라짐"""
    tc, tkn = 123.4567891234, 59861.48554199999
    rows = [dict(t=100.0, length_m=0.7, heading_deg=10.0, dheading_deg=1.0, moving=True, dheading_rad=-0.01,
                 t_confirm=tc, t_known=tkn, idx=5, length_raw=0.7),
            dict(t=101.0, length_m=0.0, heading_deg=10.0, dheading_deg=0.0, moving=False, dheading_rad=0.0,
                 t_confirm=tkn, t_known=float("nan"), idx=105, length_raw=0.6)]
    f = tmp_path / "steps.csv"
    rt_pdr.write_rows(rows, f)
    back = rt_pdr.read_rows(f)
    assert back[0]["t_confirm"] == tc and back[0]["t_known"] == tkn and back[1]["t_confirm"] == tkn
    assert np.isnan(back[1]["t_known"])
    assert float(f"{tc:.6f}") != tc and float(f"{tkn:.6f}") != tkn


class CountFilter(FakeFilter):
    """가짜 PF: x = 지금까지 받은 predict 걸음 길이 합(동쪽), 나침반 방향 90°(동쪽)"""

    def estimate(self):
        e = super().estimate()
        e.update(x=float(sum(c[1] for c in self.calls if c[0] == "predict")), compass=90.0)
        return e


# 격자 봉우리 호스트 시각(소수 6자리로 쓰면 59860.635542)과 재생 시각 원점
G_HOST = 59860.635541999996
T0_REC = 59800.5


def _cut_case(tmp_path):
    """걸음 파일을 거쳐 읽은 걸음 1개(봉우리 = G_HOST 반올림, 판정 = 갱신 1)와 갱신 3개(0: 봉우리 전, 1: 봉우리까지, 2: 0.1 s 뒤)"""
    t1 = G_HOST + 0.85
    row = dict(t=G_HOST, length_m=0.7, heading_deg=0.0, dheading_deg=0.0, moving=True, dheading_rad=0.0,
               t_confirm=t1, t_known=t1, idx=0, length_raw=0.7)
    rt_pdr.write_rows([row], tmp_path / "steps.csv")
    rows = rt_pdr.read_rows(tmp_path / "steps.csv")
    assert rows[0]["t"] == float(f"{G_HOST:.6f}") != G_HOST          # 걸음 파일은 소수 6자리
    tick_t = np.array([30.0 + T0_REC, t1, t1 + 0.1]) - T0_REC
    cut_t = np.array([29.0 + T0_REC, G_HOST, G_HOST + 0.1]) - T0_REC   # 반올림 안 한 격자 값
    return rows, tick_t, cut_t


def test_g3_online_steps_cut_aligned_to_step_precision(tmp_path):
    """판정 끝난 시각(반올림 안 한 격자 값)을 걸음 파일과 같은 소수 6자리로 맞춰서, 갱신 1에서 경계가 그 봉우리 걸음을 이미 통과시킨다.
    음성 대조: 맞추지 않은 원래 값을 경계로 쓰면 걸음 키가 경계보다 큼(한 갱신 기다림)"""
    rows, tick_t, cut_t = _cut_case(tmp_path)
    sk = rr.OnlineSteps(rows, T0_REC, tick_t, cut_t)
    key = (sk.t[0], 1, 0)
    assert key <= sk.bound(tick_t[1])
    assert not key <= (float(cut_t[1]), 9)


def test_g3_run_rt_same_grid_obs_after_step(tmp_path):
    """같은 격자 시각의 관측과 걸음: 갱신 1(벽시계 t1)에서 predict → update 순서로 같이 들어가고 출력 w = t1.
    음성 대조: OnlineSteps.cut_t를 원래 격자 값으로 되돌리면 한 갱신(0.1 s) 늦게 들어감"""
    rows, tick_t, cut_t = _cut_case(tmp_path)

    def go(raw):
        sk = rr.OnlineSteps(rows, T0_REC, tick_t, cut_t)
        if raw:
            sk.cut_t = np.asarray(cut_t, float)
        F = FakeFilter()
        r = rr.run_rt(F, [(sk.t[0], [dict(id="o")])], sk, rr.OracleBaro([], []), _truth([0.0, 1000.0]), T0_REC, [])
        return F, r

    F, r = go(False)
    assert [c[0] for c in F.calls] == ["predict", "update"]
    assert len(r) == 1 and r["w"][0] == tick_t[1]
    F, r = go(True)
    assert [c[0] for c in F.calls] == ["predict", "update"] and r["w"][0] == tick_t[2]


def test_g4_heading_lookup_respects_avail():
    """방향 표본 (시각 0·1·2·3, 값 0·0·1·1 rad, 받은 시각 0·1·2.5·3.5): w = 2.2에 시각 2 방향은 받은 표본(0·1 s)만 보간해 0,
    w = 2.5부터 1. 받은 시각 없는 전체 배열판은 w와 관계없이 1"""
    ht, hh, av = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.5, 3.5])
    f = rr.heading_lookup((ht, hh, av), 0.0)
    assert f(2.0, 2.2) == 0.0 and f(1.5, 2.2) == 0.0
    assert f(2.0, 2.5) == 1.0
    assert f(3.0, 3.0) == 1.0                              # 시각 3 표본은 아직 안 받음 → 마지막 받은 값(2 s)
    g = rr.heading_lookup((ht, hh), 0.0)
    assert g(2.0, 2.2) == 1.0
    assert rr.heading_lookup((ht, hh, None), 0.0)(2.0, 2.2) == 1.0


@pytest.mark.parametrize("lag,want", [(0.2, 0.0), (0.6, 57.29577951308232)])
def test_g4_run_rt_obs_dpsi_from_received_heading(lag, want):
    """2 s 관측을 벽시계 2 + lag에 넣을 때 회전 보정 = 그때까지 받은 방향 표본으로 본 2 s 방향(걸음 없음 → 기준 0).
    시각 2 표본(1 rad)은 2.5 s에 받음 → lag 0.2: 0°, lag 0.6: 57.3°. 받은 시각 없는 전체 배열이면 둘 다 57.3°"""
    ht, hh, av = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.5, 3.5])
    for h, exp in (((ht, hh, av), want), ((ht, hh), 57.29577951308232)):
        F = FakeFilter()
        r = rr.run_rt(F, [(2.0, [dict(id="o")])], rr.OracleSteps([], 0.0), rr.OracleBaro([], []), _truth([0.0, 100.0]), 0.0,
                      [], heading=h, obs_lag=lag)
        assert [c[0] for c in F.calls] == ["update"] and r["w"][0] == pytest.approx(2.0 + lag)
        assert F.calls[0][2] == pytest.approx(exp, abs=1e-9)


def test_g5_late_obs_extrapolate_from_pf_state_time():
    """1 s마다 동쪽 1 m 걸음(1~15 s), 관측 1.1 s·12.1 s가 10 s 늦게 옴: 출력 행 ts = 그때까지 넣은 가장 늦은 사건(걸음 11 s·15 s) > t64.
    화면 12.1 s(마지막 출력 = 11.1 s에 나온 1.1 s 관측 행, 위치 11 m)에서 앞당긴 위치 = ts 11 s부터 1.1 s × 1 m/s = 12.1 m = 정답(1 m/s 동쪽).
    음성 대조: ts 대신 촬영 시각(1.1 s)부터 앞당기면 속도 0.5 m/s × 11 s → 4.4 m 오차"""
    steps = [_rstep(float(t), -np.pi / 2, 0.0, L=1.0) for t in range(1, 16)]
    F = CountFilter()
    obs = [(1.1, [dict(id="o1")]), (12.1, [dict(id="o12")])]
    truth = _truth([0.0, 30.0], east=[0.0, 30.0])
    r = rr.run_rt(F, obs, rr.OracleSteps(steps, 0.0), rr.OracleBaro([], []), truth, 0.0, [], obs_lag=10.0)
    assert list(r["t64"]) == [1.1, 12.1]
    assert r["w"][0] == pytest.approx(11.1) and r["w"][1] == pytest.approx(22.1)
    assert list(r["ex"]) == [11.0, 15.0]
    assert list(r["ts"]) == [11.0, 15.0] and (r["ts"] > r["t64"]).all()
    ht = np.arange(0.0, 30.0, 0.1)
    hd = (ht, np.full(len(ht), -np.pi / 2), ht)
    e_now, e_dr = rr.now_errors(r, truth, steps, 0.0, hd)
    assert np.isnan(e_now[0]) and e_now[1] == pytest.approx(1.1)
    assert e_dr[1] == pytest.approx(0.0, abs=1e-6)
    old = r.copy()
    old["ts"] = old["t64"]
    assert rr.now_errors(old, truth, steps, 0.0, hd)[1][1] == pytest.approx(4.4, abs=1e-6)


def test_e_refuse_overwrite_foreign_summary(run_rt, tmp_path):
    """runs.npz 없이 summary.json만 있어도: summary 키가 실시간 단계가 아니면 멈춤, 단계 키만 있으면 통과"""
    a = SimpleNamespace(name="x", check_ref="exp_d58_r20")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "summary.json").write_text(json.dumps(dict(info={}, summary={"known_tf_기본": {}, "off": {}})), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_rt.refuse_overwrite(bad, a)
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "summary.json").write_text(json.dumps(dict(info={}, summary={"off": {}, "both_ignore": {}})), encoding="utf-8")
    run_rt.refuse_overwrite(ok, a)
    assert not (bad / "runs.npz").exists() and not (ok / "runs.npz").exists()


def _tiny_session(sec=3.0, fs=100.0):
    """시험용 3 s Session(정지 중력 + 2 Hz 세로 흔들림, 회전 없음)"""
    from indoornav.io import Session
    t = np.arange(0.0, sec, 1.0 / fs)
    n = len(t)
    g = np.zeros((n, 3))
    g[:, 2] = -9.81
    acc = np.zeros((n, 3))
    acc[:, 2] = 0.5 * np.sin(2 * np.pi * 2.0 * t)
    return Session(name="tiny", fs=fs, t=t, acc_user=acc, gravity=g, gyro=np.zeros((n, 3)))


@pytest.mark.parametrize("change", [{"step.signal": "norm"}, {"heading.zaru": True}, {"heading.source": "orientation"},
                                    {"step.method": "adaptive"}], ids=lambda d: next(iter(d)))
def test_g7_online_steps_refuses_non_realtime_params(change):
    """앞쪽 값을 쓸 수 있는 설정(pdr_tuned_v4에서 하나만 바꿈)이면 online_steps가 ValueError(실시간 감싸기 확인 문구)"""
    from indoornav.pdr import PDRParams
    p = PDRParams.from_toml(ROOT / "configs/pdr_tuned_v4.toml").with_(**change)
    with pytest.raises(ValueError, match="실시간 감싸기"):
        rt_pdr.online_steps(_tiny_session(), p)


def test_g7_online_steps_accepts_v4():
    """pdr_tuned_v4 그대로는 통과하고 갱신 기록·방향이 나옴"""
    from indoornav.pdr import PDRParams
    p = PDRParams.from_toml(ROOT / "configs/pdr_tuned_v4.toml")
    rt_pdr.check_realtime_params(rt_pdr.delayed_params(p))
    st, tk = rt_pdr.online_steps(_tiny_session(), p)
    assert len(tk["tick_t"]) > 0 and len(tk["heading"]) == 300


# ---------------------------------------------------------------- H. 권장 단계 rec
def test_h_recommended_stage_values(run_rt):
    """RECOMMENDED = "rec", stage("rec") = early 걸음 행·봉우리 뒤 0.5 s·탑승 의심 무시·최대 기다림 5 s(기압은 상대 고도),
    ignore_l05 = online·0.5 s·의심 무시(최대 기다림 기본 5 s)"""
    assert run_rt.RECOMMENDED == "rec" and run_rt.RECOMMENDED in run_rt.STAGES
    sg = run_rt.stage("rec")
    assert {k: v for k, v in sg.items() if k != "note"} == dict(steps="early", look=0.5, baro="ignore", max_wait=5.0,
                                                                  baro_src="rel")
    s2 = run_rt.stage("ignore_l05")
    assert {k: v for k, v in s2.items() if k != "note"} == dict(steps="online", look=0.5, baro="ignore",
                                                                  max_wait=run_rt.MAX_WAIT, baro_src="rel")
    assert run_rt.MAX_WAIT == 5.0


def test_h_recommended_stage_file_names(run_rt, tmp_path):
    """rec 걸음 파일 = steps_early_l050.csv, 갱신 파일 = ticks_l050.npz (prepare의 l{look*100:03d} 이름 규칙),
    기본 봉우리 뒤 0.75 s 단계(both_early)와 겹치지 않음"""
    sg = run_rt.stage("rec")
    assert run_rt.step_file(tmp_path, sg["steps"], sg["look"]) == tmp_path / "steps_early_l050.csv"
    assert f"ticks_l{int(round(sg['look'] * 100)):03d}.npz" == "ticks_l050.npz"
    assert run_rt.step_file(tmp_path, "online", run_rt.stage("ignore_l05")["look"]).name == "steps_online_l050.csv"
    be = run_rt.stage("both_early")
    assert run_rt.step_file(tmp_path, be["steps"], be["look"]).name == "steps_early_l075.csv"


@pytest.mark.parametrize("name, key", [("rec", "steps_early_l050"), ("ignore_l05", "steps_online_l050")])
def test_h_knowledge_reads_rec_files(run_rt, monkeypatch, name, key):
    """run_rt.knowledge가 rec는 steps_early_l050·ticks_l050, ignore_l05는 steps_online_l050·ticks_l050을 읽고,
    기압은 상대 고도 후보로 의심 무시(OnlineBaro suspect="ignore"). 다른 이름 파일만 있으면 KeyError"""
    t0 = 100.0
    rows = [dict(t=101.0, t_confirm=101.5, t_known=101.5, length_m=0.7, heading_deg=0.0, moving=True)]
    other = [dict(t=101.0, t_confirm=101.75, t_known=101.75, length_m=0.7, heading_deg=0.0, moving=True)]
    ticks = (np.array([1.5]), np.array([1.0]))
    smp = dict(t=np.array([0.0]), t_recv=np.array([0.1]))
    R = dict(steps={key: rows, "steps_early_l075": other, "steps_online_l075": other}, ticks={"ticks_l050": ticks},
             baro=([], smp), baro_abs=([], dict(t=np.array([9.0]), t_recv=np.array([9.0]))))
    monkeypatch.setattr(run_rt, "_R", R)
    monkeypatch.setattr(run_rt.run_i1, "_G", dict(t0=t0))
    sk, bk = run_rt.knowledge(run_rt.stage(name))
    assert isinstance(sk, rr.OnlineSteps) and sk.steps is rows
    assert np.array_equal(sk.tick_t, ticks[0])
    assert isinstance(bk, rr.OnlineBaro) and bk.suspect == "ignore" and bk.recv is smp["t_recv"]
    monkeypatch.setattr(run_rt, "_R", dict(R, steps={k: v for k, v in R["steps"].items() if k != key}))
    with pytest.raises(KeyError):
        run_rt.knowledge(run_rt.stage(name))
