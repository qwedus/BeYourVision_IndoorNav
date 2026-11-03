"""보폭 계수 온라인 보정 시뮬레이션.

파티클 필터가 아직 없으므로, '서버가 랜드마크로 위치를 확정해 준다'는 상황을 흉내 낸다.

가정 (정답이 걸은 총거리뿐이라 필요한 단순화):
  - 한 세션 안에서는 첫 걸음~마지막 걸음 사이를 일정한 속도로 걸었다고 보고, 시각 t 까지의 실제 거리를
    총거리 × (경과 시간 비율)로 만든다.
  - fix_spacing_m 마다 위치 확정(fix)이 들어오고, 위치 오차는 표준편차 pos_noise_m 의 잡음으로 넣는다.
  - 서버가 주는 거리는 두 점 사이 직선거리다. 꺾인 구간은 보정기가 방향 변화로 걸러낸다고 보고,
    시뮬레이션에서는 구간 경로 거리를 그대로 준다.

평가 지표:
  - 구간 거리 오차: 위치 확정 사이 구간마다 |PDR 거리 − 실제 거리| / 실제 거리.
    파티클 필터는 확정 사이를 PDR 로 버티므로 이 값이 실제로 중요하다.

사용 예:
    uv run python scripts/sim_online_calib.py
    uv run python scripts/sim_online_calib.py --config configs/pdr_tuned_v3.toml --spacing 10 --noise 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from indoornav.eval.runner import load_all  # noqa: E402
from indoornav.pdr import PDRParams, run_pdr  # noqa: E402
from indoornav.pdr.online_calib import CalibParams, StrideCalibrator  # noqa: E402


def simulate(steps: pd.DataFrame, dist_gt: float, calib: StrideCalibrator | None, k_fixed: float,
             spacing: float, noise: float, rng: np.random.Generator, t_offset: float = 0.0) -> dict:
    """한 세션을 걸음 순서대로 재생. calib 가 None 이면 K 고정.
    t_offset: 여러 세션을 이어 걸을 때 시각이 겹치지 않도록 더하는 값 [s]."""
    t = steps["t"].to_numpy() + t_offset
    f = steps["feature"].to_numpy()
    h = steps["heading"].to_numpy()
    t0, t1 = t[0], t[-1]
    true_at = lambda tt: dist_gt * (tt - t0) / (t1 - t0) if t1 > t0 else 0.0  # noqa: E731

    seg_err, k_trace = [], []
    last_fix_t, last_fix_true = t0, 0.0
    last_fix_meas = 0.0 + rng.normal(0, noise)
    seg_pdr, total_pdr = 0.0, 0.0
    for ti, fi, hi in zip(t, f, h):
        L = calib.add_step(ti, fi, hi) if calib else k_fixed * fi
        seg_pdr += L
        total_pdr += L
        k_trace.append(calib.k if calib else k_fixed)
        d_true = true_at(ti)
        if d_true - last_fix_true >= spacing:
            seg_true = d_true - last_fix_true
            seg_err.append(abs(seg_pdr - seg_true) / seg_true)
            meas = d_true + rng.normal(0, noise)
            if calib:
                calib.correct(last_fix_t, ti, meas - last_fix_meas, uncertainty_m=noise * 1.5)
            last_fix_t, last_fix_true, last_fix_meas, seg_pdr = ti, d_true, meas, 0.0
    return dict(seg_err=seg_err, total_pdr=total_pdr, k_end=k_trace[-1] if k_trace else np.nan,
                t_end=float(t[-1]) if len(t) else t_offset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs/pdr_tuned_v3.toml")
    ap.add_argument("--data", default=ROOT / "data")
    ap.add_argument("--sessions", default=ROOT / "docs/sessions.csv")
    ap.add_argument("--routes", default=ROOT / "docs/routes.csv")
    ap.add_argument("--spacing", type=float, default=10.0, help="위치 확정 간격 [m]")
    ap.add_argument("--noise", type=float, default=0.3, help="위치 확정 오차 표준편차 [m]")
    ap.add_argument("--repeats", type=int, default=20, help="잡음 무작위 반복 횟수")
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    params = PDRParams.from_toml(a.config)
    k_true = params.length.k
    pairs, _ = load_all(a.data, a.sessions, a.routes, None, params.fs)
    # 제자리 움직임(보폭 0) 걸음은 빼고 실제 이동 걸음만 재생한다. 앞뒤 정지 구간의 발 움직임이
    # '일정 속도' 가정의 시작·끝 시각을 틀어지게 하지 않도록.
    def moving_steps(st):
        return st[st["moving"]].reset_index(drop=True) if "moving" in st else st
    runs = [(gt.session_id[:8] + "_" + gt.session_id[9:], moving_steps(run_pdr(s, params).steps), gt.distance_m)
            for s, gt in pairs if gt.distance_m]
    cp = CalibParams()

    # ---------- 1) 세션마다 새로 시작 (K = 튜닝값에서 출발)
    rows = []
    for name, st, d in runs:
        e_fix, e_cal, k_end = [], [], []
        for r in range(a.repeats):
            rng = np.random.default_rng(r)
            e_fix += simulate(st, d, None, k_true, a.spacing, a.noise, rng)["seg_err"]
            rng = np.random.default_rng(r)
            out = simulate(st, d, StrideCalibrator(k_true, cp), k_true, a.spacing, a.noise, rng)
            e_cal += out["seg_err"][1:]           # 첫 구간은 보정 전이라 제외
            k_end.append(out["k_end"])
        rows.append({"세션": name, "구간 수": len(e_fix) // a.repeats,
                     "K 고정 구간오차 %": 100 * np.mean(e_fix) if e_fix else np.nan,
                     "온라인 보정 구간오차 %": 100 * np.mean(e_cal) if e_cal else np.nan,
                     "끝 K": np.mean(k_end)})
    df1 = pd.DataFrame(rows)
    print(f"[1] 세션마다 K={k_true:.3f} 에서 시작, 위치 확정 {a.spacing:g} m 마다, 위치 오차 σ={a.noise:g} m")
    print(df1.round(3).to_string(index=False))
    print(f"   평균: K 고정 {df1['K 고정 구간오차 %'].mean():.1f}%  →  온라인 보정 {df1['온라인 보정 구간오차 %'].mean():.1f}%")

    # ---------- 2) 다른 사람 흉내: 틀린 K 로 시작해서 세션을 이어서 걸을 때 K 가 어디로 가나
    order = [n for n in ["S1_normal_01", "S2_normal_01", "T2_normal_01", "L1_normal_01",
                         "S1_fast_01", "S1_fast_02"] if any(r[0] == n for r in runs)]
    print("\n[2] 틀린 K 로 시작해 보통·빠른 걸음 세션을 이어서 걸었을 때 K 변화 (정답 근처:", round(k_true, 3), ")")
    for k0 in (0.25, 0.55):
        cal = StrideCalibrator(k0, cp)
        rng = np.random.default_rng(0)
        trace = [f"시작 {k0:.2f}"]
        t_off = 0.0
        for name in order:
            st, d = next((r[1], r[2]) for r in runs if r[0] == name)
            t_off = simulate(st, d, cal, k0, a.spacing, a.noise, rng, t_off)["t_end"] + 60.0
            trace.append(f"{name} 후 {cal.k:.3f}")
        applied = sum(r.applied for r in cal.log)
        reasons = pd.Series([r.reason for r in cal.log]).value_counts().to_dict()
        print(f"   K0={k0}: " + " → ".join(trace))
        print(f"           위치 확정 {len(cal.log)}회 중 반영 {applied}회, 사유별: {reasons}")


if __name__ == "__main__":
    main()
