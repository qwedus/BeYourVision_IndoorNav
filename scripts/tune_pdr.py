"""PDR 파라미터 튜닝.

두 가지 방식이 있다.

--method cv (기본, 교차검증)
    세션 하나를 빼고 나머지로 값을 고른 뒤, 빼둔 세션으로 오차를 잰다. 모든 세션을 한 번씩 빼며 반복.
    → '처음 보는 데이터에서의 오차'를 데이터를 버리지 않고 추정할 수 있다.
    → 반복마다 골라진 값이 얼마나 흔들리는지도 보여준다 (데이터가 충분한지 판단).
    마지막에 전체 세션으로 값을 한 번 더 골라 저장한다.

--method split (tune/eval 고정 분할)
    sessions.csv 의 split=tune 으로 고르고 split=eval 로 확인한다.

각 반복 안에서 값을 고르는 순서:
  1) 걸음 검출: GRID 의 모든 조합 중 걸음 수 평균 절대 오차(%)가 가장 작은 조합
  2) 보폭: 모델별로 지수 후보(EXPONENTS)마다 계수를 최소제곱으로 구하고, 거리 오차가 가장 작은 지수
       - weinberg      : L = K·r^e
       - weinberg_freq : L = K·r^e + k_freq·f + offset   (f = 걸음 빈도)
  3) 두 보폭 모델 중 교차검증 거리 오차가 작은 쪽을 최종 선택 (--model 로 고정 가능)

사용 예:
    uv run python scripts/tune_pdr.py --save configs/pdr_tuned_v2.toml --note "2차 (반복 측정 추가)"
    uv run python scripts/tune_pdr.py --method split

--save 를 주면 docs/tuning_log.md 에 기록이 자동으로 추가된다. 기록 끝의 '분석·메모'는 직접 채운다.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from indoornav.eval.runner import evaluate_all, load_all  # noqa: E402
from indoornav.pdr import PDRParams, run_pdr  # noqa: E402
from indoornav.pdr.step_length import calibrate_freq, calibrate_k  # noqa: E402

GRIDS = {
    # step.method = "peak": 수직 가속도 봉우리
    "peak": {
        "step.min_height": [0.15, 0.2, 0.3, 0.4, 0.5],
        "step.min_prominence": [0.2, 0.3, 0.5],
        "preprocess.lpf_cutoff_hz": [2.0, 2.5, 3.0],
        "step.min_interval_s": [0.30, 0.40, 0.50],
    },
    # step.method = "adaptive": 속도 적응형 (느린 걸음은 가속도 크기 신호 사용)
    "adaptive": {
        "step.slow_energy": [0.2, 0.25, 0.3, 0.35],
        "step.adaptive_prom": [0.5, 0.8, 1.0, 1.2],
        "step.min_quality": [0.1, 0.2, 0.3],
    },
}
GRID = GRIDS["peak"]
EXPONENTS = [round(float(x), 2) for x in np.arange(0.15, 1.01, 0.05)]
MODELS = ["weinberg", "weinberg_freq"]


# ------------------------------------------------------------------ 캐시
class Cache:
    """같은 (설정, 세션) 조합을 여러 번 돌리지 않도록 결과를 저장."""

    def __init__(self, pairs, base: PDRParams, grid: dict):
        self.pairs = {gt.session_id: (s, gt) for s, gt in pairs}
        self.base = base
        self.combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        self._steps: dict[tuple[int, str], int] = {}
        self._feat: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def params(self, ci: int) -> PDRParams:
        return self.base.with_(**self.combos[ci])

    def steps(self, ci: int, sid: str) -> int:
        if (ci, sid) not in self._steps:
            self._steps[(ci, sid)] = run_pdr(self.pairs[sid][0], self.params(ci)).n_steps
        return self._steps[(ci, sid)]

    def raw_features(self, ci: int, sid: str) -> tuple[np.ndarray, np.ndarray]:
        """(걸음별 a_max − a_min, 걸음별 빈도). 지수=1 로 돌리면 feature 가 원값이다."""
        if (ci, sid) not in self._feat:
            r = run_pdr(self.pairs[sid][0], self.params(ci).with_(**{"length.exponent": 1.0}))
            mv = r.steps["moving"].to_numpy() if "moving" in r.steps else np.ones(len(r.steps), bool)
            # 제자리 움직임(보폭 0 처리된 걸음)은 거리 계산에서 뺀다
            self._feat[(ci, sid)] = (r.steps["feature"].to_numpy()[mv], r.steps["freq"].to_numpy()[mv])
        return self._feat[(ci, sid)]


# ------------------------------------------------------------------ 값 고르기
def choose_step(cache: Cache, sids: list[str]) -> tuple[int, float]:
    best, best_err = 0, float("inf")
    for ci in range(len(cache.combos)):            # 격자 순서대로 → 동점이면 앞쪽 조합
        errs = [abs(cache.steps(ci, s) - cache.pairs[s][1].steps) / cache.pairs[s][1].steps
                for s in sids if cache.pairs[s][1].steps]
        e = float(np.mean(errs)) if errs else float("inf")
        if e < best_err - 1e-12:
            best, best_err = ci, e
    return best, 100 * best_err


def fit_length(cache: Cache, ci: int, sids: list[str], model: str) -> dict:
    """주어진 걸음 검출 설정에서 보폭 모델의 지수와 계수를 고른다."""
    use = [s for s in sids if cache.pairs[s][1].distance_m]
    raw = [cache.raw_features(ci, s) for s in use]
    d = [cache.pairs[s][1].distance_m for s in use]
    best = None
    for e in EXPONENTS:
        f = [r ** e for r, _ in raw]
        q = [fr for _, fr in raw]
        if model == "weinberg":
            k, kf, c = calibrate_k(f, d), 0.0, 0.0
        else:
            k, kf, c = calibrate_freq(f, q, d)
        pred = [k * fi.sum() + kf * qi.sum() + c * len(fi) for fi, qi in zip(f, q)]
        err = float(np.mean([abs(p - di) / di for p, di in zip(pred, d)]))
        if best is None or err < best["err"]:
            best = dict(err=err, exponent=e, k=k, k_freq=kf, offset_m=c)
    return best


def predict_distance(cache: Cache, ci: int, sid: str, L: dict) -> float:
    r, q = cache.raw_features(ci, sid)
    return float((L["k"] * r ** L["exponent"] + L["k_freq"] * q + L["offset_m"]).sum())


def apply(base: PDRParams, cache: Cache, ci: int, model: str, L: dict) -> PDRParams:
    return base.with_(**cache.combos[ci], **{
        "length.model": model, "length.exponent": L["exponent"], "length.k": round(L["k"], 4),
        "length.k_freq": round(L["k_freq"], 4), "length.offset_m": round(L["offset_m"], 4)})


# ------------------------------------------------------------------ 교차검증
def cross_validate(cache: Cache, sids: list[str]) -> pd.DataFrame:
    rows = []
    for hold in sids:
        train = [s for s in sids if s != hold]
        ci, _ = choose_step(cache, train)
        gt = cache.pairs[hold][1]
        n_est = cache.steps(ci, hold)
        for model in MODELS:
            L = fit_length(cache, ci, train, model)
            dist = predict_distance(cache, ci, hold, L)
            rows.append({
                "빼둔 세션": hold, "model": model,
                **{k.split(".")[1]: v for k, v in cache.combos[ci].items()},
                "exponent": L["exponent"], "k": L["k"], "k_freq": L["k_freq"], "offset_m": L["offset_m"],
                "steps_gt": gt.steps, "steps_est": n_est,
                "step_err_pct": 100 * (n_est - gt.steps) / gt.steps if gt.steps else np.nan,
                "dist_gt": gt.distance_m, "dist_est": dist,
                "dist_err_pct": 100 * (dist - gt.distance_m) / gt.distance_m if gt.distance_m else np.nan,
            })
    return pd.DataFrame(rows)


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    g = cv.groupby("model")
    return pd.DataFrame({
        "걸음 수 오차 %": g["step_err_pct"].agg(lambda x: x.abs().mean()),
        "거리 오차 %": g["dist_err_pct"].agg(lambda x: x.abs().mean()),
        "거리 오차 최대 %": g["dist_err_pct"].agg(lambda x: x.abs().max()),
        "지수 범위": g["exponent"].agg(lambda x: f"{x.min():.2f}~{x.max():.2f}"),
        "K 범위": g["k"].agg(lambda x: f"{x.min():.3f}~{x.max():.3f}"),
    }).round(1)


# ------------------------------------------------------------------ 출력·기록
def to_toml(p: PDRParams) -> str:
    d = p.to_dict()
    lines = [f"fs = {d.pop('fs')}", ""]
    for sec, vals in d.items():
        lines.append(f"[{sec}]")
        for k, v in vals.items():
            if isinstance(v, str):
                s = f'"{v}"'
            elif isinstance(v, (bool, np.bool_)):
                s = str(bool(v)).lower()
            elif isinstance(v, (int, np.integer)):
                s = str(int(v))
            else:
                s = repr(round(float(v), 6))
            lines.append(f"{k} = {s}")
        lines.append("")
    return "\n".join(lines)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v))
                                     else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def changed(base: PDRParams, best: PDRParams) -> list[tuple[str, object, object]]:
    b, t = base.to_dict(), best.to_dict()
    return [(f"{sec}.{k}", b[sec][k], v) for sec, vals in t.items() if isinstance(vals, dict)
            for k, v in vals.items() if b[sec][k] != v]


def append_log(log: Path, header: str, body: str) -> None:
    if not log.exists():
        log.write_text("# PDR 튜닝 기록\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n## {datetime.now():%Y-%m-%d %H:%M} — {header}\n\n{body}\n")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cv", "split"], default="cv")
    ap.add_argument("--model", choices=MODELS, default=None, help="보폭 모델 고정 (기본: 교차검증으로 선택)")
    ap.add_argument("--step-method", choices=list(GRIDS), default="peak", help="걸음 검출 방식")
    ap.add_argument("--config", default=ROOT / "configs/pdr_default.toml")
    ap.add_argument("--data", default=ROOT / "data")
    ap.add_argument("--sessions", default=ROOT / "docs/sessions.csv")
    ap.add_argument("--routes", default=ROOT / "docs/routes.csv")
    ap.add_argument("--save", default=None, help="튜닝 결과 TOML 저장 경로 (주면 튜닝 기록도 남김)")
    ap.add_argument("--note", default="")
    ap.add_argument("--log", default=ROOT / "docs/tuning_log.md")
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    base = PDRParams.from_toml(a.config).with_(**{"step.method": a.step_method})
    grid = GRIDS[a.step_method]
    pairs, missing = load_all(a.data, a.sessions, a.routes, None, base.fs)
    pairs = [(s, gt) for s, gt in pairs if gt.steps or gt.distance_m]
    if missing:
        print(f"[경고] data/ 에 없는 세션: {', '.join(missing)}")
    cache = Cache(pairs, base, grid)
    all_ids = [gt.session_id for _, gt in pairs]
    print(f"세션 {len(all_ids)}개, 걸음 검출 조합 {len(cache.combos)}개, 보폭 지수 후보 {len(EXPONENTS)}개")

    log_parts = [f"- 방식: `{a.method}`  걸음 검출: `{a.step_method}`  기준 설정: `{Path(a.config).name}`",
                 f"- 세션 ({len(all_ids)}): {', '.join(all_ids)}",
                 "- 걸음 검출 탐색 격자: " + "; ".join(f"`{k}` {v}" for k, v in grid.items()),
                 f"- 보폭 지수 후보: {EXPONENTS[0]}~{EXPONENTS[-1]} (0.05 간격), 모델 후보: {', '.join(MODELS)}"]

    if a.method == "cv":
        cv = cross_validate(cache, all_ids)
        summ = cv_summary(cv)
        print("\n[교차검증] 빼둔 세션 기준 오차 (처음 보는 데이터에서의 예상 오차)")
        print(summ.to_string())
        model = a.model or summ["거리 오차 %"].idxmin()
        print(f"\n선택한 보폭 모델: {model}")
        grid_cols = [k.split(".")[1] for k in grid]
        show = cv[cv.model == model][["빼둔 세션", *grid_cols, "exponent", "k", "k_freq", "offset_m",
                                      "steps_gt", "steps_est", "step_err_pct", "dist_err_pct"]]
        print(show.round(3).to_string(index=False))
        train_ids = all_ids
        log_parts += ["", "### 교차검증 결과 (빼둔 세션 기준)", md_table(summ, index=True),
                      "", f"선택한 보폭 모델: **{model}**", "",
                      "### 반복별로 골라진 값 (값이 많이 흔들리면 데이터 부족)", md_table(show.round(3))]
    else:
        train_ids = [gt.session_id for _, gt in pairs if gt.split == "tune"]
        model = a.model
        if model is None:   # tune 세션 안에서 거리 오차가 작은 모델
            ci, _ = choose_step(cache, train_ids)
            model = min(MODELS, key=lambda m: fit_length(cache, ci, train_ids, m)["err"])
        print(f"\ntune {len(train_ids)}개로 선택, 보폭 모델: {model}")

    # 최종 값: train_ids 전체로 한 번 더 고른다
    ci, step_err = choose_step(cache, train_ids)
    L = fit_length(cache, ci, train_ids, model)
    best = apply(base, cache, ci, model, L)
    print(f"\n[최종] 걸음 검출 {cache.combos[ci]}  (학습 걸음 오차 {step_err:.1f}%)")
    print(f"[최종] 보폭 {model}: 지수 {L['exponent']}, K {L['k']:.4f}, k_freq {L['k_freq']:.4f}, "
          f"offset {L['offset_m']:.4f}  (학습 거리 오차 {100 * L['err']:.1f}%)")

    df_base, df_best = evaluate_all(pairs, base), evaluate_all(pairs, best)
    per = df_base[["session_id", "split", "steps_gt", "steps_est", "dist_gt", "dist_est"]].merge(
        df_best[["session_id", "steps_est", "dist_est", "end_err_m", "turn_est"]],
        on="session_id", suffixes=("_기준", "_튜닝"))
    print("\n[최종 설정으로 전체 세션]" + (" (학습에 쓴 데이터라 실제보다 좋게 나옴)" if a.method == "cv" else ""))
    print(per.round(2).to_string(index=False))
    if a.method == "split":
        e = df_best[df_best.split == "eval"]
        print(f"\neval 평균 절대 오차: 걸음 {e.step_err_pct.abs().mean():.1f}%, 거리 {e.dist_err_pct.abs().mean():.1f}%")

    if a.save:
        Path(a.save).write_text(to_toml(best), encoding="utf-8")
        print(f"\n저장: {a.save}")
        ch = changed(base, best)
        log_parts += ["", "### 최종 선택 (전체 학습 데이터로 다시 고른 값)",
                      "| 파라미터 | 기준 | 튜닝 |", "|---|---|---|",
                      *[f"| `{k}` | {v0} | {v1} |" for k, v0, v1 in ch],
                      "", "### 세션별 (최종 설정)", md_table(per.round(2)),
                      "", "### 분석·메모 (직접 작성)", "- "]
        append_log(Path(a.log), f"`{Path(a.save).name}` {a.note}", "\n".join(log_parts))
        print(f"기록 추가: {a.log}")


if __name__ == "__main__":
    main()
