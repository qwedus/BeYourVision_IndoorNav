"""비전 추가 기능 비교표: 앞 단계까지 채택한 상태 vs 기능 하나 더 켬 (작업기록 3차 4~7번).

    .venv/Scripts/python integration/compare_features.py <표 제목> <결과 파일> <이름>[=<표시>] ...
    → integration/out/i1/<결과 파일>
이름은 integration/out/i1/<이름>/summary.json. 첫 이름이 비교 기준. 모두 indoornav PDR + PF, seed 10, N 3000, 같은 입력·초기 표본.
칸: 위치 오차 중앙 / 90% / 최대 · 끝 (m), 오차÷불확실도, 오차 ≤ 2×불확실도, unc95 포함률, 자신 있게 틀림
"""
import json
import sys

import paths
from compare import ROWS, SC, OB, key, f

NOTE = ("녹화 1회 재생(1차 답사 보행, **손에 든 가로 촬영**)이라 낙관적일 수 있고 조끼 성능이 아님. "
        "indoornav PDR + PF, seed 10개 중앙, N 3000, 같은 입력·초기 표본. 관측 설정 기본 = 5 Hz·σ 5°·바닥값 0.1, 중간 = 2 Hz·σ 10°·바닥값 0.1. "
        "칸 = 오차 중앙 / 90% / 최대 · 끝 (m), 오차÷불확실도, 오차 ≤ 2×불확실도, unc95 포함률, 자신 있게 틀림")
WANT = [r for r in ROWS if r[1] in ("raw", "tf")]


def cell(m):
    if m is None:
        return "—"
    return (f"{f(m['med_all'], 2)} / {f(m['p90_all'])} / {f(m['max_all'])} · {f(m['end'])}, {f(m['ratio'])}, "
            f"{f(100 * m['cover2'], 0)}%, {f(100 * m['in95'], 0)}%, {m['wrong']}/{m['n']}")


def table(title, runs):
    res = []
    for r in runs:
        name, _, label = r.partition("=")
        res.append((label or name, json.loads((paths.OUT / "i1" / name / "summary.json").read_text(encoding="utf-8"))["summary"]))
    L = [f"# {title}", "", NOTE, "", "| 조건 | 설정 | " + " | ".join(c for c, _ in res) + " |", "|---|---|" + "---|" * len(res)]
    for sc, ob, st in WANT:
        L.append(f"| {SC[sc]} · {OB[ob]} | {st} | " + " | ".join(cell(s.get(key(sc, ob, st))) for _, s in res) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    title, out, *runs = sys.argv[1:]
    t = table(title, runs)
    (paths.OUT / "i1" / out).write_text(t, encoding="utf-8")
    print(t)
