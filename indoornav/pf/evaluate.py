"""PF 평가: 여러 세션 × 시드 재생, 체크포인트 오차 집계, 관측 시나리오."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.runner import load_all
from ..pdr import PDRParams, run_pdr
from .filter import PFParams
from .map import WallMap
from .replay import replay, route_checkpoints

PF_EXPERIMENTS = ("S1", "S2", "L1", "L2", "L4", "T2")
LOST_M = 3.0   # 이 이상 틀리면 '위치 잃음' (복도 폭 1.4~1.8 m → 다른 변·모서리로 착각한 수준)


def load_pdr_runs(root: Path, pdr_config: str | Path, include_ref: bool = False) -> list[dict]:
    """PF 재생에 필요한 PDR 결과를 세션별로 한 번만 계산."""
    p = PDRParams.from_toml(pdr_config)
    pairs, _ = load_all(root / "data", root / "docs/sessions.csv", root / "docs/routes.csv", None, p.fs,
                        include_ref)
    runs = []
    for s, gt in pairs:
        if gt.experiment_id in PF_EXPERIMENTS:
            runs.append({"sid": gt.session_id, "exp": gt.experiment_id, "pdr": run_pdr(s, p), "t": s.t,
                         "ref": gt.flag == "ref"})
    return runs


def obs_split(names: list[str], which: int) -> set[str]:
    """START 다음 체크포인트부터 번갈아 관측(which=0: 홀수 번째, 1: 짝수 번째). 나머지를 채점."""
    rest = [n for n in names if n != "START"]
    return {n for i, n in enumerate(rest) if i % 2 == which}


def run_scenario(runs, wm: WallMap, params: PFParams, scenario: str = "map_only",
                 seeds=(0, 1, 2), keep_results: bool = False):
    """scenario: map_only | half_obs (두 가지 번갈아 관측 분할 모두 실행)"""
    rows, results = [], []
    for r in runs:
        names = [c.name for c in route_checkpoints(r["exp"], r["pdr"], r["t"], wm)]
        splits = [set()] if scenario == "map_only" else [obs_split(names, 0), obs_split(names, 1)]
        for si, obs in enumerate(splits):
            for sd in seeds:
                res = replay(r["sid"], r["exp"], r["pdr"], r["t"], wm, params, obs_names=obs, seed=sd)
                for x in res.rows:
                    x.update(seed=sd, split=si, exp=r["exp"], k_final=res.k_final,
                             all_dead=res.stats.get("all_dead", 0))
                rows += res.rows
                if keep_results and sd == seeds[0] and si == 0:
                    results.append(res)
    df = pd.DataFrame(rows)
    return (df, results) if keep_results else df


def score(df: pd.DataFrame) -> dict:
    """채점 체크포인트(관측에 안 쓴 것)만으로 지표."""
    e = df[~df.observed]
    if e.empty:
        return {}
    inside = (e.err_pf <= e.unc95).mean()
    return {
        "n": len(e),
        "pf_median": e.err_pf.median(), "pf_p90": e.err_pf.quantile(0.9), "pf_max": e.err_pf.max(),
        "pdr_median": e.err_pdr.median(), "pdr_p90": e.err_pdr.quantile(0.9), "pdr_max": e.err_pdr.max(),
        "lost_rate": (e.err_pf > LOST_M).mean(),
        "coverage95": inside,
        "objective": e.err_pf.quantile(0.9) + 10 * (e.err_pf > LOST_M).mean(),
    }


def with_(p: PFParams, **kw) -> PFParams:
    return replace(p, **kw)
