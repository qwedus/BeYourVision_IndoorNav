"""PF 평가: 저장된 PF 설정으로 여러 시나리오를 돌려 표와 그림을 남긴다.

시나리오
  map_only      : 지도만 (관측 없음)                          ← 튜닝 기준
  half_obs      : 체크포인트 절반을 '랜드마크 관측'으로, 나머지로 채점
  half_obs_bad  : half_obs + 관측 위치에 1 m 치우침 (비전 오차 흉내)
  unknown_head  : 지도만 + 출발 방향 모름 (U(0, 360°))
  n_sweep       : 지도만, 파티클 수 250/500/1000/2000

사용 예:
    uv run python scripts/run_pf.py --config configs/pf_home_v3.toml --plots
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from indoornav.pf.evaluate import load_pdr_runs, run_scenario, score, with_  # noqa: E402
from indoornav.pf.map import WallMap  # noqa: E402
from indoornav.pf.plots import plot_replay, set_korean_font  # noqa: E402
from indoornav.pf.replay import replay  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from tune_pf import params_from_toml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs/pf_home_v3.toml")
    ap.add_argument("--pdr-config", default=ROOT / "configs/pdr_tuned_v4.toml")
    ap.add_argument("--map", default=ROOT / "maps/home_loop/map.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=ROOT / "reports")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    wm = WallMap.load(a.map)
    runs = load_pdr_runs(ROOT, a.pdr_config)
    P = params_from_toml(a.config)
    seeds = tuple(range(a.seeds))
    out = Path(a.out) / f"pf_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    t = time.time()
    df_map, res_map = run_scenario(runs, wm, P, "map_only", seeds, keep_results=True)
    summary["map_only"] = score(df_map)
    df_half = run_scenario(runs, wm, P, "half_obs", seeds)
    summary["half_obs"] = score(df_half)
    # 관측 치우침: replay 의 obs_bias 사용
    rows = []
    for r in runs:
        from indoornav.pf.evaluate import obs_split
        from indoornav.pf.replay import route_checkpoints
        names = [c.name for c in route_checkpoints(r["exp"], r["pdr"], r["t"], wm)]
        for si in (0, 1):
            for sd in seeds:
                res = replay(r["sid"], r["exp"], r["pdr"], r["t"], wm, P, obs_names=obs_split(names, si),
                             seed=sd, obs_bias=1.0 / 2**0.5)
                rows += res.rows
    summary["half_obs_bad(1m)"] = score(pd.DataFrame(rows))
    summary["unknown_head"] = score(run_scenario(runs, wm, with_(P, init_h_unknown=True), "map_only", seeds))
    for n in (250, 500, 1000, 2000):
        summary[f"map_only_N{n}"] = score(run_scenario(runs, wm, with_(P, n=n), "map_only", seeds))
    print(f"재생 {time.time() - t:.0f}s")

    s = pd.DataFrame(summary).T[["n", "pf_median", "pf_p90", "pf_max", "pdr_median", "pdr_p90", "pdr_max",
                                 "lost_rate", "coverage95"]]
    print("\n[시나리오 요약] 오차 m (채점 체크포인트 = 관측에 안 쓴 것)")
    print(s.round(2).to_string())
    per = (df_map[~df_map.observed].groupby(["session_id", "checkpoint"])
           [["err_pdr", "err_pf", "unc95", "k_final"]].mean().round(2))
    print("\n[세션·체크포인트별, 지도만, 시드 평균]")
    print(per.to_string())
    s.to_csv(out / "summary.csv", encoding="utf-8-sig")
    per.to_csv(out / "per_checkpoint.csv", encoding="utf-8-sig")
    df_map.to_csv(out / "map_only_rows.csv", index=False, encoding="utf-8-sig")

    if a.plots:
        set_korean_font()
        n = len(res_map)
        cols = 5
        fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 4 * ((n + cols - 1) // cols)))
        for ax, res in zip(axs.flat, res_map):
            plot_replay(ax, wm, res)
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        axs.flat[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"PF 지도만 (관측 없음) · {Path(a.config).name} · 회색 PDR 단독 / 빨강 PF", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "tracks_map_only.png", dpi=90)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
