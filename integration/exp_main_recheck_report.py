"""exp_main_recheck 결과 9개 기록 → integration/out/main_recheck.md·main_recheck.json (표만 만듦, 재생 안 함).

같은 seed 짝 차이 = 조건별로 seed마다 (바꾼 조건 − 기본) 값을 구해 그 중앙. 잡음 기준 = 기본 조건 seed 0~9 중앙과 10~19 중앙의 차이.
"""
import json
from pathlib import Path

import numpy as np

from provenance import RECORDINGS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
Q = ("med_all", "p90_all", "max_all", "cover2", "end")


def rec_dir(rec):
    rc = json.loads((HERE / "recordings" / f"{rec}.json").read_text(encoding="utf-8"))
    return HERE.parent / rc["out"] / "main_recheck"


def load():
    D = {}
    for rec in RECORDINGS:
        f = rec_dir(rec) / "result.json"
        if f.exists():
            D[rec] = json.loads(f.read_text(encoding="utf-8"))
    return D


def med(run, key, src="m"):
    v = [s[src][key] for s in run["seeds"] if s[src].get(key) is not None]
    return float(np.median(v)) if v else None


def wrong(run):
    return sum(bool(s["m"]["wrong"]) for s in run["seeds"])


def paired(a, b, key, src="m"):
    """seed 짝 (b − a) 중앙"""
    A = {s["seed"]: s[src].get(key) for s in a["seeds"]}
    d = [s[src][key] - A[s["seed"]] for s in b["seeds"] if s[src].get(key) is not None and A.get(s["seed"]) is not None]
    return float(np.median(d)) if d else None


f1 = lambda v: "—" if v is None else f"{v:.1f}"
f2 = lambda v: "—" if v is None else f"{v:.2f}"
s2 = lambda v: "—" if v is None else f"{v:+.2f}"
pc = lambda v: "—" if v is None else f"{100 * v:.0f}%"
SHORT = lambda r: r.replace("s2_", "").replace("route2_1030", "1차")


def report():
    D = load()
    J, L = {}, ["# main 수정 건의 재확인 (exp_main_recheck)", "", f"기록 {len(D)}개: " + ", ".join(SHORT(r) for r in D) + ". seed 10개 중앙, N 3000, 출발점 앎, 채택 비전 옵션.", ""]

    # 1. 현상
    L += ["## 1. 벽 제거와 k (비전 끔 = PF만)", "",
          "| 기록 | PDR÷정답 | 1/비 (k 범위 0.7~1.4) | 끝 k 벽 있음 / 없음 | 이상적 걸음 끝 k 벽 있음 / 없음 | Δk 막힘·리샘플·랜덤워크 | 벽 제거 | 리샘플 걸음 비율 | 리샘플 직전 ESS 중앙(÷N) | 막힌 비율 중앙 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for rec, d in D.items():
        R, st = d["runs"], d["info"]["stride"]
        a, o, ia, io = R["pdr/kill/none"], R["pdr/off/none"], R["ideal/kill/none"], R["ideal/off/none"]
        ratio = st["all"]["ratio"]
        J.setdefault(rec, {})["phen_none"] = dict(ratio=ratio, k_end=med(a, "k_end"), k_end_off=med(o, "k_end"), k_end_ideal=med(ia, "k_end"),
                                                 k_end_ideal_off=med(io, "k_end"), dk_wall=med(a, "dk_wall", "log"), dk_res=med(a, "dk_res", "log"),
                                                 dk_rw=med(a, "dk_rw", "log"), kills=med(a, "wall_kills", "log"), res_frac=med(a, "res_frac", "log"),
                                                 ess=med(a, "ess_pre_med", "log"), killfrac=med(a, "kill_frac_med", "log"),
                                                 ideal_dk_wall=med(ia, "dk_wall", "log"), ideal_kills=med(ia, "wall_kills", "log"))
        L.append(f"| {SHORT(rec)} | {ratio:.2f} | {1 / ratio:.2f}{'(범위 밖)' if not 0.7 <= 1 / ratio <= 1.4 else ''} | {f2(med(a, 'k_end'))} / {f2(med(o, 'k_end'))} | {f2(med(ia, 'k_end'))} / {f2(med(io, 'k_end'))} | "
                 f"{s2(med(a, 'dk_wall', 'log'))} · {s2(med(a, 'dk_res', 'log'))} · {s2(med(a, 'dk_rw', 'log'))} | {med(a, 'wall_kills', 'log'):.0f} | {pc(med(a, 'res_frac', 'log'))} | "
                 f"{f2(med(a, 'ess_pre_med', 'log'))} | {100 * med(a, 'kill_frac_med', 'log'):.2f}% |")
    L += ["", "## 1-2. 벽 제거와 k (PF + 비전, 시간 필터·기본)", "",
          "| 기록 | 끝 k 벽 있음 / 없음 | 이상적 걸음 끝 k 벽 있음 / 없음 | Δk 막힘·리샘플·관측 | 벽 제거 | 리샘플 걸음 비율 | 리샘플 직전 ESS 중앙 | 오차 중앙/90% 벽 있음 | 벽 없음 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for rec, d in D.items():
        R = d["runs"]
        a, o, ia, io = R["pdr/kill/tf"], R["pdr/off/tf"], R["ideal/kill/tf"], R["ideal/off/tf"]
        J[rec]["phen_tf"] = dict(k_end=med(a, "k_end"), k_end_off=med(o, "k_end"), k_end_ideal=med(ia, "k_end"), k_end_ideal_off=med(io, "k_end"),
                                 dk_wall=med(a, "dk_wall", "log"), dk_upd=med(a, "dk_upd", "log"), kills=med(a, "wall_kills", "log"),
                                 res_frac=med(a, "res_frac", "log"), ess=med(a, "ess_pre_med", "log"))
        L.append(f"| {SHORT(rec)} | {f2(med(a, 'k_end'))} / {f2(med(o, 'k_end'))} | {f2(med(ia, 'k_end'))} / {f2(med(io, 'k_end'))} | "
                 f"{s2(med(a, 'dk_wall', 'log'))} · {s2(med(a, 'dk_res', 'log'))} · {s2(med(a, 'dk_upd', 'log'))} | {med(a, 'wall_kills', 'log'):.0f} | {pc(med(a, 'res_frac', 'log'))} | "
                 f"{f2(med(a, 'ess_pre_med', 'log'))} | {f1(med(a, 'med_all'))} / {f1(med(a, 'p90_all'))} | {f1(med(o, 'med_all'))} / {f1(med(o, 'p90_all'))} |")

    # 2. 에스컬레이터
    L += ["", "## 2. 에스컬레이터 탄 기록", "",
          "| 기록 | 비전 | PDR÷정답 탑승 전 / 뒤 | 탑승 직전 k 벽 있음 / 없음 | 하차 뒤 오차 중앙 그대로 / k=1 | 끝 오차 그대로 / k=1 | 오차 90% 그대로 / k=1 |", "|---|---|---|---|---|---|---|"]
    for rec, d in D.items():
        R, st = d["runs"], d["info"]["stride"]
        if "reset/tf" not in R:
            continue
        for v, nm in (("none", "끔"), ("tf", "시간·기본"), ("tf중간", "시간·중간")):
            a = R[f"pdr/kill/{v}"]
            o = R.get(f"pdr/off/{v}")
            rs = R[f"reset/{v}"]
            J[rec][f"esc_{v}"] = dict(k_board=med(a, "k_board"), k_board_off=med(o, "k_board") if o else None, after=med(a, "err_after"),
                                      after_reset=med(rs, "err_after"), end=med(a, "end"), end_reset=med(rs, "end"),
                                      p90=med(a, "p90_all"), p90_reset=med(rs, "p90_all"))
            L.append(f"| {SHORT(rec)} | {nm} | {st['before_ride']['ratio']:.2f} / {st['after_ride']['ratio']:.2f} | {f2(med(a, 'k_board'))} / {f2(med(o, 'k_board')) if o else '—'} | "
                     f"{f1(med(a, 'err_after'))} / {f1(med(rs, 'err_after'))} | {f1(med(a, 'end'))} / {f1(med(rs, 'end'))} | {f1(med(a, 'p90_all'))} / {f1(med(rs, 'p90_all'))} |")
        for v, nm in (("none", "끔"), ("tf", "시간·기본")):
            a, b = R[f"drift/free/{v}"], R[f"drift/reset/{v}"]
            J[rec][f"esc_drift_{v}"] = dict(after=med(a, "err_after"), after_reset=med(b, "err_after"), end=med(a, "end"), end_reset=med(b, "end"))
            L.append(f"| {SHORT(rec)} PDR 틀어짐 | {nm} | — | {f2(med(a, 'k_board'))} / — | {f1(med(a, 'err_after'))} / {f1(med(b, 'err_after'))} | {f1(med(a, 'end'))} / {f1(med(b, 'end'))} | "
                     f"{f1(med(a, 'p90_all'))} / {f1(med(b, 'p90_all'))} |")

    # 잡음 기준
    L += ["", "## 잡음 기준 (기본 조건, seed 0~9 중앙 − seed 10~19 중앙, 절댓값)", "", "| 기록 | 비전 | 오차 중앙 | 90% | 최대 | cover2 | 끝 |", "|---|---|---|---|---|---|---|"]
    NZ = {}
    for rec, d in D.items():
        R = d["runs"]
        for v in ("none", "tf"):
            a, b = R[f"pdr/kill/{v}"], R[f"noise/{v}"]
            z = {q: abs(med(a, q) - med(b, q)) if med(a, q) is not None and med(b, q) is not None else None for q in Q}
            NZ.setdefault(v, []).append(z)
            L.append(f"| {SHORT(rec)} | {v} | {f2(z['med_all'])} | {f2(z['p90_all'])} | {f2(z['max_all'])} | {f2(z['cover2'])} | {f2(z['end'])} |")
    J["noise"] = NZ

    # 3. 고칠 후보
    VAR = [("ess", "ESS 기준 리샘플"), ("pen01", "점수 ×0.1"), ("pen05", "점수 ×0.5"), ("slide", "벽 따라 미끄러짐"), ("slide_ess", "미끄러짐 + ESS")]
    L += ["", "## 3. 고칠 후보 (기본과 같은 seed 짝 차이 중앙, 기록마다)", "",
          "칸 = 오차 중앙 / 90% / 최대 차이 (m), cover2 차이(%p), 자신 있게 틀림 seed 수 기본→후보, 끝 k 기본→후보, 리샘플 걸음 비율 기본→후보", ""]
    for v, vn in (("none", "비전 끔"), ("tf", "시간 필터·기본"), ("tf중간", "시간 필터·중간")):
        L += [f"### {vn}", "", "| 기록 | " + " | ".join(n for _, n in VAR) + " |", "|---" * (len(VAR) + 1) + "|"]
        for rec, d in D.items():
            R = d["runs"]
            a = R[f"pdr/kill/{v}"]
            cells = []
            for k, _ in VAR:
                b = R[f"pdr/{k}/{v}"]
                dd = {q: paired(a, b, q) for q in Q}
                J[rec].setdefault("cand", {})[f"{k}/{v}"] = dict(diff=dd, wrong=(wrong(a), wrong(b)), k_end=(med(a, "k_end"), med(b, "k_end")),
                                                                 res_frac=(med(a, "res_frac", "log"), med(b, "res_frac", "log")),
                                                                 base={q: med(a, q) for q in Q}, cand={q: med(b, q) for q in Q},
                                                                 kills=(med(a, "wall_kills", "log"), med(b, "wall_kills", "log")))
                cells.append(f"{s2(dd['med_all'])} / {s2(dd['p90_all'])} / {s2(dd['max_all'])}, {100 * dd['cover2']:+.0f}%p, {wrong(a)}→{wrong(b)}, "
                             f"k {f2(med(a, 'k_end'))}→{f2(med(b, 'k_end'))}, 리샘플 {pc(med(a, 'res_frac', 'log'))}→{pc(med(b, 'res_frac', 'log'))}")
            L.append(f"| {SHORT(rec)} | " + " | ".join(cells) + " |")
        L += [""]
        # 기록 합산: 차이 평균, 잡음(이 비전 조건의 기록별 잡음 최대) 넘게 나빠진 기록 수
        nzv = NZ["none" if v == "none" else "tf"]
        L += ["| 후보 | 오차 중앙 차이 평균 | 90% 차이 평균 | 최대 차이 평균 | cover2 차이 평균 | 90%가 0.2 m 넘게 나빠진 기록 | 0.2 m 넘게 좋아진 기록 |", "|---|---|---|---|---|---|---|"]
        for k, n in VAR:
            ds = [J[r]["cand"][f"{k}/{v}"]["diff"] for r in D]
            mean = lambda q: float(np.mean([x[q] for x in ds if x[q] is not None]))
            worse = sum(x["p90_all"] > 0.2 for x in ds)
            better = sum(x["p90_all"] < -0.2 for x in ds)
            L.append(f"| {n} | {mean('med_all'):+.2f} | {mean('p90_all'):+.2f} | {mean('max_all'):+.2f} | {100 * mean('cover2'):+.1f}%p | {worse} | {better} |")
        L += [""]
    L += ["### 이상적 걸음(실제 k = 1)에서 후보의 끝 k (비전 끔)", "", "| 기록 | 기본 | 벽 없음 | " + " | ".join(n for _, n in VAR) + " |", "|---" * (len(VAR) + 3) + "|"]
    for rec, d in D.items():
        R = d["runs"]
        L.append(f"| {SHORT(rec)} | {f2(med(R['ideal/kill/none'], 'k_end'))} | {f2(med(R['ideal/off/none'], 'k_end'))} | " + " | ".join(f2(med(R[f'ideal/{k}/none'], 'k_end')) for k, _ in VAR) + " |")
        J[rec]["ideal_k_end"] = {k: med(R[f"ideal/{k}/none"], "k_end") for k, _ in VAR}

    # 4. init
    L += ["", "## 4. init() 그대로 vs 걸을 수 있는 칸에서 다시 뽑기", "",
          "| 기록 | 한 점에 몰린 비율 중앙 (최소~최대) | 비전 끔: 오차 중앙 / 90% / 시작 4~16 s 중앙 차이 | 시간 필터·기본: 차이 | 자신 있게 틀림 기본→그대로 |", "|---|---|---|---|---|"]
    for rec, d in D.items():
        R, ic = d["runs"], d["info"]["init_collapse"]
        c0 = [paired(R["pdr/kill/none"], R["init_main/none"], q) for q in ("med_all", "p90_all", "start_med")]
        c1 = [paired(R["pdr/kill/tf"], R["init_main/tf"], q) for q in ("med_all", "p90_all", "start_med")]
        J[rec]["init"] = dict(collapse=ic, none=c0, tf=c1, wrong=(wrong(R["pdr/kill/tf"]), wrong(R["init_main/tf"])))
        L.append(f"| {SHORT(rec)} | {pc(ic['med'])} ({pc(ic['min'])}~{pc(ic['max'])}) | {' / '.join(s2(x) for x in c0)} | {' / '.join(s2(x) for x in c1)} | "
                 f"{wrong(R['pdr/kill/tf'])}→{wrong(R['init_main/tf'])} |")

    # 5. 회전
    L += ["", "## 5. 걸음 사이 회전 |Δ| (도)", "", "| 기록 | 관측 프레임 수 | 중앙 | 90% | 최대 | 10° 넘음 | 보정 끔 − 켬: 시간·기본 중앙/90%/cover2 | 시간·중간 |", "|---|---|---|---|---|---|---|---|"]
    allrot = []
    for rec, d in D.items():
        R, ro = d["runs"], d["info"]["rotation"]["tf_obs"]
        allrot.append(ro)
        x = [paired(R["pdr/kill/tf"], R["bfix_off/tf"], q) for q in ("med_all", "p90_all", "cover2")]
        y = [paired(R["pdr/kill/tf중간"], R["bfix_off/tf중간"], q) for q in ("med_all", "p90_all", "cover2")]
        J[rec]["rot"] = dict(stats=d["info"]["rotation"], tf=x, tf_mid=y)
        L.append(f"| {SHORT(rec)} | {ro['n']} | {ro['med']:.1f} | {ro['p90']:.1f} | {ro['max']:.1f} | {pc(ro['over10'])} | {s2(x[0])} / {s2(x[1])} / {100 * x[2]:+.0f}%p | "
                 f"{s2(y[0])} / {s2(y[1])} / {100 * y[2]:+.0f}%p |")
    (OUT / "main_recheck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "main_recheck.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
