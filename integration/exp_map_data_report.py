"""exp_map_data 결과 9개 기록 → integration/out/map_data.md·map_data.json (표만, 재생 안 함).

같은 seed 짝 차이 = seed마다 (바꾼 조건 − 지금) 중앙. 잡음 기준은 exp_main_recheck 결과(기본 조건 seed 0~9 중앙 − 10~19 중앙)를 그대로 읽음.
'뺀 칸' 질량: 프레임마다 seed 10개 중앙을 낸 곡선의 최대·그 시각, 곡선이 0.1 넘는 구간.
"""
import json
from pathlib import Path

import numpy as np

from provenance import RECORDINGS
from exp_main_recheck_report import med, paired, wrong, SHORT, rec_dir as rc_main

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
Q = ("med_all", "p90_all", "max_all", "cover2", "unc_med")


def rec_dir(rec):
    return rc_main(rec).parent / "map_data"


def spans(t, ok):
    """ok가 이어지는 프레임 구간 [(시작, 끝)] (중간에 ok 아닌 프레임이 하나라도 있으면 끊음)"""
    out, prev = [], False
    for x, o in zip(t, ok):
        if o and prev:
            out[-1][1] = float(x)
        elif o:
            out.append([float(x), float(x)])
        prev = bool(o)
    return out


def report():
    D = {r: json.loads((rec_dir(r) / "result.json").read_text(encoding="utf-8")) for r in RECORDINGS if (rec_dir(r) / "result.json").exists()}
    N = {r: json.loads((rc_main(r) / "result.json").read_text(encoding="utf-8")) for r in D}
    J = {}
    L = ["# main 건의 4번 지도 자료 재확인 (exp_map_data)", "", f"기록 {len(D)}개, seed 10개 중앙, N 3000, 나머지 exp_d58_r20 그대로.", ""]
    for v, vn in (("none", "비전 끔"), ("tf", "시간 필터·기본")):
        L += [f"## 보행 마스크: main 원래 − 방 뺀 사본 ({vn})", "",
              "| 기록 | 오차 중앙 / 90% / 최대 지금 | 차이 중앙 / 90% / 최대 | cover2 차이 | 불확실도 중앙 지금 → 원래 | 자신 있게 틀림 지금 → 원래 | 벽 막힘 지금 → 원래 | 뺀 칸 질량 최대(시각) | 0.1 넘는 구간 (s) | 잡음 90% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for rec, d in D.items():
            R = d["runs"]
            a, b = R[f"noroom/{v}"], R[f"orig/{v}"]
            dd = {q: paired(a, b, q) for q in Q}
            room = np.median([s["log"]["room"] for s in b["seeds"]], axis=0)
            t = np.array(next(s["log"]["room_t"] for s in b["seeds"] if s["log"].get("room_t")))   # 프레임 시각(같은 비전 조건이면 seed마다 같음)
            tmax = float(t[int(room.argmax())])
            over = spans(t, room > 0.1)
            nz = N[rec]["runs"][f"noise/{v}"]
            base_n = N[rec]["runs"][f"pdr/kill/{v}"]
            noise = abs(med(base_n, "p90_all") - med(nz, "p90_all"))
            J.setdefault(rec, {})[f"mask_{v}"] = dict(base={q: med(a, q) for q in Q}, diff=dd, wrong=(wrong(a), wrong(b)),
                                                     kills=(med(a, "wall_kills", "log"), med(b, "wall_kills", "log")),
                                                     room_max=float(room.max()), room_med=float(np.median(room)), room_t_max_med=tmax,
                                                     room_over10=over, noise_p90=noise)
            L.append(f"| {SHORT(rec)} | {med(a, 'med_all'):.2f} / {med(a, 'p90_all'):.2f} / {med(a, 'max_all'):.2f} | {dd['med_all']:+.2f} / {dd['p90_all']:+.2f} / {dd['max_all']:+.2f} | "
                     f"{100 * dd['cover2']:+.0f}%p | {med(a, 'unc_med'):.2f} → {med(b, 'unc_med'):.2f} | {wrong(a)} → {wrong(b)} | "
                     f"{med(a, 'wall_kills', 'log'):.0f} → {med(b, 'wall_kills', 'log'):.0f} | {100 * room.max():.0f}% ({tmax:.0f} s) | {', '.join(f'{x:.1f}~{y:.1f}' for x, y in over if y - x >= 1.0)[:60] or '1 s 넘게 이어진 구간 없음'} | {noise:.2f} |")
        L.append("")
    L += ["## 정답 궤적과 마스크", "", "| 기록 | 정답 점 수 | 뺀 칸 위 | 방 뺀 사본 밖 | 원래 마스크 밖 |", "|---|---|---|---|---|"]
    for rec, d in D.items():
        g = d["info"]["geometry"]
        J[rec]["geom"] = {k: g[k] for k in ("removed", "out_noroom", "out_orig")}
        sp = lambda k: f"{100 * g[k]['frac']:.1f}%" + (" " + ", ".join(f"{a:.1f}~{b:.1f}" for a, b in g[k]["spans"][:6]) if g[k]["spans"] else "")
        L.append(f"| {SHORT(rec)} | {g['n']} | {sp('removed')} | {sp('out_noroom')} | {sp('out_orig')} |")
    g = next(iter(D.values()))["info"]["geometry"]["escalators"]
    L += ["", "## 에스컬레이터 아래 끝 (aligned x, 축 y 위에서 0.1 m 간격으로 처음 걸을 수 있는 x)", "",
          "| 에스컬레이터 | 그래프 아래 끝 | main 아래 끝 | 위 끝 그래프 / main | 원래 마스크 | 방 뺀 사본 | 역사 패키지 보행칸(메시) |", "|---|---|---|---|---|---|---|"]
    for e in g:
        L.append(f"| {e['main_name']} | {e['graph_bottom'][0]:.2f}, {e['graph_bottom'][1]:.2f} | {e['main_bottom'][0]:.2f}, {e['main_bottom'][1]:.2f} | "
                 f"{e['graph_top'][0]:.2f} / {e['main_top'][0]:.2f} | {e['first_free_orig']:.1f} | {e['first_free_noroom']:.1f} | {e['first_free_station']:.1f} |")
    J["escalators"] = g
    L += ["", "## 하차 좌표: main − 그래프 (에스컬레이터 탄 기록)", "",
          "| 기록 | 층 이동 | 비전 | 하차 뒤 오차 중앙 지금 → main | 하차 뒤 90% 지금 → main | 끝 오차 지금 → main | 전체 90% 차이 | 자신 있게 틀림 지금 → main |", "|---|---|---|---|---|---|---|---|"]
    for rec, d in D.items():
        R = d["runs"]
        if "escmain/tf" not in R:
            continue
        mv = "B1→B2" if d["info"]["floors"][0][1] < 0 else "B2→B1"
        for v, vn, bk in (("none", "끔", "noroom/none"), ("tf", "시간·기본", "noroom/tf"), ("tf중간", "시간·중간", "noroom/tf중간")):
            a, b = R[bk], R[f"escmain/{v}"]
            J[rec][f"esc_{v}"] = dict(after=(med(a, "err_after"), med(b, "err_after")), p90_after=(med(a, "p90_after"), med(b, "p90_after")),
                                      end=(med(a, "end"), med(b, "end")), d_after=paired(a, b, "err_after"), d_p90=paired(a, b, "p90_all"),
                                      wrong=(wrong(a), wrong(b)))
            L.append(f"| {SHORT(rec)} | {mv} | {vn} | {med(a, 'err_after'):.2f} → {med(b, 'err_after'):.2f} | {med(a, 'p90_after'):.2f} → {med(b, 'p90_after'):.2f} | "
                     f"{med(a, 'end'):.2f} → {med(b, 'end'):.2f} | {paired(a, b, 'p90_all'):+.2f} | {wrong(a)} → {wrong(b)} |")
    (OUT / "map_data.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "map_data.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
