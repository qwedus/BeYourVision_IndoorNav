"""PF 파라미터 튜닝 (집 복도 home_loop, 지도만 사용·관측 없음 단계).

방법: 기록 세션 재생 → 경로 중간 체크포인트(모서리·끝점) 오차로 채점 → 격자 탐색.
세션 하나씩 빼고 고른 값으로 빼둔 세션을 채점(교차검증)해 '처음 보는 데이터' 성능을 본다.
목적함수 = 체크포인트 오차 p90 + 10 × 위치 잃음 비율(오차 > 3 m).

사용 예:
    uv run python scripts/tune_pf.py --save configs/pf_home_v1.toml --note "1차"
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from indoornav.pf.evaluate import load_pdr_runs, run_scenario, score, with_  # noqa: E402
from indoornav.pf.filter import PFParams  # noqa: E402
from indoornav.pf.map import WallMap  # noqa: E402

GRID = {
    "sigma_len": [0.05, 0.10, 0.15],
    "sigma_h0": [np.radians(d) for d in (0.2, 0.5, 1.0, 2.0)],
    "sigma_h1": [0.005, 0.01, 0.03],
    "sigma_k": [0.002, 0.005, 0.01, 0.02, 0.04],
}

_G = {}


def _init(runs, wm, base, seeds):
    _G.update(runs=runs, wm=wm, base=base, seeds=seeds)


def _eval_combo(combo):
    p = with_(_G["base"], **dict(zip(GRID, combo)))
    df = run_scenario(_G["runs"], _G["wm"], p, "map_only", _G["seeds"])
    df["combo"] = str(combo)
    return df


def params_to_toml(p: PFParams) -> str:
    lines = []
    for k, v in p.to_dict().items():
        if isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif k.startswith("sigma_h0") or k in ("init_h_sigma", "rough_h"):
            lines.append(f"{k} = {float(v):.6f}   # {np.degrees(v):.2f} deg")
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def params_from_toml(path) -> PFParams:
    import tomllib
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return PFParams(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-config", default=ROOT / "configs/pdr_tuned_v4.toml")
    ap.add_argument("--map", default=ROOT / "maps/home_loop/map.json")
    ap.add_argument("--n", type=int, default=500, help="튜닝 중 파티클 수 (속도용)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--save", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--log", default=ROOT / "docs/pf_tuning_log.md")
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    wm = WallMap.load(a.map)
    runs = load_pdr_runs(ROOT, a.pdr_config)
    base = PFParams(n=a.n)
    seeds = tuple(range(a.seeds))
    combos = list(itertools.product(*GRID.values()))
    print(f"세션 {len(runs)}개, 조합 {len(combos)}개, 시드 {len(seeds)}개, N={a.n}")
    t = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(runs, wm, base, seeds)) as pool:
        dfs = pool.map(_eval_combo, combos, chunksize=4)
    df = pd.concat(dfs, ignore_index=True)
    print(f"재생 {time.time() - t:.0f}s")

    sids = sorted(df.session_id.unique())
    # 교차검증: 세션 하나 빼고 목적함수 최소 조합 선택 → 빼둔 세션 채점
    cv_rows, picks = [], []
    for sid in sids:
        tr = df[df.session_id != sid]
        obj = tr.groupby("combo").apply(lambda g: score(g)["objective"], include_groups=False)
        best = obj.idxmin()
        picks.append(best)
        cv_rows.append(df[(df.session_id == sid) & (df.combo == best)])
    cv = pd.concat(cv_rows)
    cvs = score(cv)
    obj_all = df.groupby("combo").apply(lambda g: pd.Series(score(g)), include_groups=False)
    best = obj_all.objective.idxmin()
    best_p = with_(base, **dict(zip(GRID, eval(best))))
    print("\n[교차검증] 빼둔 세션 체크포인트 기준")
    print(pd.Series(cvs).round(3).to_string())
    print(f"\n교차검증 반복마다 고른 조합 종류: {len(set(picks))}개 / {len(picks)}회")
    print("\n[전체 데이터 상위 10 조합]")
    top = obj_all.sort_values("objective").head(10).copy()
    print(top[["pf_median", "pf_p90", "pf_max", "lost_rate", "coverage95", "objective"]].round(3).to_string())
    print(f"\n[최종] {best}")

    if a.save:
        Path(a.save).write_text(f"# PF 튜닝 결과 ({datetime.now():%Y-%m-%d}) {a.note}\n"
                                f"# 지도: {Path(a.map).name}, PDR: {Path(a.pdr_config).name}\n"
                                + params_to_toml(best_p), encoding="utf-8")
        log = Path(a.log)
        head = "" if log.exists() else "# PF 튜닝 기록\n\n자동 기록 (scripts/tune_pf.py). 분석·메모는 직접 채운다.\n"
        body = [f"\n## {datetime.now():%Y-%m-%d %H:%M} `{Path(a.save).name}` {a.note}",
                f"- 세션 {len(sids)}개, 조합 {len(combos)}개, 시드 {len(seeds)}, N={a.n}, 관측 없음(지도만)",
                "- 격자: " + "; ".join(f"`{k}` {np.round(np.degrees(v), 2).tolist() if k == 'sigma_h0' else v}"
                                     for k, v in GRID.items()),
                "", "| 지표 (교차검증) | 값 |", "|---|---|",
                *[f"| {k} | {v:.3f} |" for k, v in cvs.items()],
                "", f"최종 조합 (sigma_len, sigma_h0[rad], sigma_h1, sigma_k): `{best}`", ""]
        log.write_text((log.read_text(encoding="utf-8") if log.exists() else head) + "\n".join(body),
                       encoding="utf-8")
        print(f"저장: {a.save}, 기록: {log}")


if __name__ == "__main__":
    main()
