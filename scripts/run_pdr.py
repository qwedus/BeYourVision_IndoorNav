"""모든 세션에 PDR을 돌리고 정답과 비교한 표를 출력·저장한다.

사용 예 (레포 최상위 폴더에서):
    uv run python scripts/run_pdr.py
    uv run python scripts/run_pdr.py --config configs/pdr_default.toml --split eval
    uv run python scripts/run_pdr.py --plots          # 세션별 그래프 PNG 저장
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from indoornav.eval.runner import evaluate_all, load_all, summarize  # noqa: E402
from indoornav.pdr import PDRParams, run_pdr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs/pdr_default.toml")
    ap.add_argument("--data", default=ROOT / "data")
    ap.add_argument("--sessions", default=ROOT / "docs/sessions.csv")
    ap.add_argument("--routes", default=ROOT / "docs/routes.csv")
    ap.add_argument("--split", choices=["tune", "eval"], default=None)
    ap.add_argument("--out", default=ROOT / "reports")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--include-ref", action="store_true", help="flag=ref(극단 조건 참고용) 세션도 포함")
    a = ap.parse_args()

    params = PDRParams.from_toml(a.config)
    pairs, missing = load_all(a.data, a.sessions, a.routes, a.split, params.fs, a.include_ref)
    if missing:
        print(f"[경고] data/ 에 없는 세션 {len(missing)}개: {', '.join(missing)}")
    if not pairs:
        print("돌릴 세션이 없습니다. data/<session_id>/ 폴더와 docs/sessions.csv 를 확인하세요.")
        return

    df = evaluate_all(pairs, params)
    pd.set_option("display.width", 200)
    show = ["session_id", "split", "steps_gt", "steps_est", "steps_iphone", "step_err_pct",
            "dist_gt", "dist_est", "dist_err_pct", "end_err_m", "turn_est", "turn_err_deg"]
    print(df[show].round(2).to_string(index=False))
    print("\n보행 구간 (정답 거리·걸음 수 ÷ 걷기 시간)")
    print(df[["session_id", "rec_s", "walk_s", "still_s", "speed_gt", "cadence_gt"]]
          .round(2).to_string(index=False))
    print("\n평균 절대 오차 (split별)")
    print(summarize(df).to_string())

    out = Path(a.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    (out / "params.json").write_text(json.dumps(params.to_dict(), indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    if a.plots:
        from indoornav.eval.plots import plot_heading, plot_steps, plot_trajectory
        for s, _ in pairs:
            r = run_pdr(s, params)
            fig = plt.figure(figsize=(14, 8))
            plot_steps(s, r, fig.add_subplot(3, 2, (1, 2)))
            plot_heading(s, r, fig.add_subplot(3, 2, (3, 4)))
            plot_trajectory(r, fig.add_subplot(3, 2, 5))
            fig.tight_layout()
            fig.savefig(out / f"{s.name}.png", dpi=110)
            plt.close(fig)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
