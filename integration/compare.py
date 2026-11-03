"""I1 결과 비교표: 시뮬 PF 병합 기준(b852501)과 팀장 PF 재생 결과를 같은 조건 칸에 나란히 (연결안 7절 4·5단계).

    .venv/Scripts/python integration/compare.py [--sim-teampdr <localization_design cv_test/out/i1/check/sim_teampdr.json>]
→ integration/out/i1/compare.md
열: 기준 = 시뮬 PF + 임시 PDR / ① 시뮬 PF + 팀장 PDR (localization_design에서 돌린 값) / ② 팀장 PF + 임시 PDR /
    ③ 팀장 PDR + 팀장 PF / 5단계 ②·③에서 팀장 PF 걸음당 방향 잡음 sigma_h0 0.5° → 1.5°
"""
import argparse
import json
import sys

import paths

RUNS = [("simple_pdr_team_pf", "② 팀장 PF + 임시 PDR"), ("team_pdr_pf", "③ 팀장 PDR + 팀장 PF"),
        ("simple_pdr_team_pf_h15", "② 방향 잡음 1.5°"), ("team_pdr_pf_h15", "③ 방향 잡음 1.5°")]
ROWS = [("known", "pdr_only", "—"), ("known", "none", "—"), ("known", "raw", "기본"), ("known", "raw", "중간"),
        ("known", "tf", "기본"), ("known", "tf", "중간"),
        ("drift", "pdr_only", "—"), ("drift", "none", "—"), ("drift", "raw", "기본"), ("drift", "raw", "중간"),
        ("drift", "tf", "기본"), ("drift", "tf", "중간")]
SC = {"known": "출발점 앎", "drift": "출발점 앎 · PDR 틀어짐"}
OB = {"pdr_only": "PDR만 (지도 없음)", "none": "PF 비전 끔", "raw": "PF + 비전 필터 없음", "tf": "PF + 비전 시간 필터"}


def key(sc, ob, st):
    return f"{sc}/{ob}" if ob in ("pdr_only", "none") else f"{sc}/{ob}/{st}"


def f(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def cell(s):
    if s is None:
        return "—"
    return f"{f(s['med_all'])} / {f(s['p90_all'])} / {f(s['max_all'])} · 끝 {f(s['end'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-teampdr")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    base = json.loads((paths.I1 / "baseline_summary.json").read_text(encoding="utf-8"))["summary"]
    cols = [("기준: 시뮬 PF + 임시 PDR", base)]
    if a.sim_teampdr:
        cols.append(("① 시뮬 PF + 팀장 PDR", json.loads(open(a.sim_teampdr, encoding="utf-8").read())["summary"]))
    res = {}
    for name, label in RUNS:
        p = paths.OUT / "i1" / name / "summary.json"
        if p.exists():
            res[name] = json.loads(p.read_text(encoding="utf-8"))
            cols.append((label, res[name]["summary"]))
    L = ["# I1 비교표 — 팀장 PF vs 시뮬 PF 기준", "",
         "칸: 걷는 구간 위치 오차 **중앙 / 90% / 최대 (m) · 끝(3-1 문 앞) 오차 (m)**, seed 10개 중앙. "
         "에스컬레이터 구간(78.3~103 s) 제외, 지표 정의는 localization_design run_i1.metrics와 같음. "
         "N 3000, 관측 설정 기본 = 5 Hz·σ 5°·바닥값 0.1, 중간 = 2 Hz·σ 10°·바닥값 0.1.", "",
         "| 조건 | 설정 | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for sc, ob, st in ROWS:
        k = key(sc, ob, st)
        L.append(f"| {SC[sc]} · {OB[ob]} | {st} | " + " | ".join(cell(s.get(k)) for _, s in cols) + " |")
    for name, label in RUNS:
        if name not in res:
            continue
        s = res[name]["summary"]
        L += ["", f"## {label} — 불확실도·모드·PF 진단 ({name})", "",
              "| 조건 | 설정 | 오차÷불확실도 | 오차≤2×불확실도 | unc95 포함률 | 자신 있게 틀림 | 출발 4~16 s 오차 | 방향 오차 (°) | "
              "두 모드(질량>0.1) 비율 / 분리 (m) | 끝 k | 리샘플 / 벽 제거 / 전멸 복구 | 가시성 누락 (평균/최대) | 관측·예측 (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sc, ob, st in ROWS:
            m = s.get(key(sc, ob, st))
            if m is None or ob == "pdr_only":
                continue
            L.append(f"| {SC[sc]} · {OB[ob]} | {st} | {f(m['ratio'])} | {f(100 * m['cover2'], 0)}% | {f(100 * m['in95'], 0)}% | "
                     f"{m['wrong']}/{m['n']} | {f(m['start_med'])} | {f(m['herr_med'])} | {f(100 * m['two_mode'], 0)}% / "
                     f"{f(m['sep_med'])} | {f(m['k_end'], 2)} | {f(m['diag_resamples'], 0)} / {f(m['diag_wall_kills'], 0)} / "
                     f"{f(m['diag_all_dead'], 0)} | {f(100 * m['diag_miss_mean'], 1)}% / {f(100 * m['diag_miss_max'], 0)}% | "
                     f"{f(m['diag_ms_update'], 1)} · {f(m['diag_ms_predict'], 1)} |")
    p = paths.OUT / "i1/team_pdr_pf_n1000/summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))["summary"]
        m = s["known/tf/중간"]
        L += ["", f"처리 시간(N 1000, ③ 출발점 앎·시간 필터·중간): 관측 한 프레임 {f(m['diag_ms_update'], 1)} ms, "
              f"걸음 하나 {f(m['diag_ms_predict'], 1)} ms. 오차 {cell(m)}"]
    out = paths.OUT / "i1/compare.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
