"""보행 마스크 비교: 지금(방 뺀 사본, 평면도 그림에서 만든 것) vs 정식 도면에서 만든 마스크(plan_mask.py). 기본 설정은 안 바꿈.

    .venv/Scripts/python integration/exp_plan_mask.py --make [--gates <gates.json>]     비교용 마스크를 out/plan_mask/<이름>/에 만듦
    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_plan_mask.py [--seeds 10] [--procs 11]
    .venv/Scripts/python integration/exp_plan_mask.py --report                          9개 기록 → integration/out/plan_mask.md·.json

조건(나머지는 exp_main_recheck 기본 = exp_d58_r20 재생과 같음): mask × 비전(끔 / 시간 필터·기본)
    noroom    지금 쓰는 마스크 (integration/maps)
    plan_g20  보행칸 0.5 m 넓힘 + 벽 띠 0.2 m + 개찰구 줄 엶
    plan_g30  같고 벽 띠 0.3 m
마스크는 초기 다시 뽑기·벽 판정·하차 다시 뽑기에 모두 쓰인다.
기하 확인(재생 없음): 정답 궤적(탑승 중 뺌)이 각 마스크 밖을 지나는 비율·시각.

판단 기준 (재생 전에 정함, 11-13). 주 지표는 실제 설정인 '시간 필터·기본', 같은 seed 짝 차이(새 − 지금)의 seed 10개 중앙
    1. 9개 기록 평균: 오차 중앙 +0.05 m 이하, 오차 90% +0.15 m 이하로만 나빠짐
    2. 기록 하나도 오차 90%가 max(0.5 m, 그 기록 잡음 × 3) 넘게 나빠지지 않음 (잡음 = exp_main_recheck의 seed 0~9 ↔ 10~19 차이)
    3. 자신 있게 틀린 seed 수 합이 늘지 않음 (+2까지는 같은 것으로 봄)
    4. 정답 궤적이 마스크 밖을 지나는 비율이 지금 마스크보다 크지 않음 (기록마다)
    네 가지를 모두 지키면 채택. '비전 끔'은 참고(벽만의 효과).
출력 <녹화 out>/plan_mask/result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import paths

NAME = "plan_mask"
VARIANTS = {"plan_g20": dict(thick=0.2, grow=0.5), "plan_g30": dict(thick=0.3, grow=0.5)}
MASKS = ("noroom", *VARIANTS)
VIS = (("none", "비전 끔"), ("tf", "시간 필터·기본"))
LIMIT = dict(mean_med=0.05, mean_p90=0.15, rec_p90=0.5, rec_noise_x=3.0, wrong=2)
_M = {}


def variant_meta(name):
    return paths.OUT / NAME / name / "map_meta_plan.json"


def make(gates):
    import plan_mask
    for name, v in VARIANTS.items():
        rep = plan_mask.make(paths.OUT / NAME / name, v["thick"], v["grow"], gates or plan_mask.GATES)
        print(name, json.dumps(rep, ensure_ascii=False), flush=True)


def _init(n):
    import exp_main_recheck as E
    from vision_filter import load_maps
    E._init(n)
    missing = [str(variant_meta(v)) for v in VARIANTS if not variant_meta(v).is_file()]
    if missing:
        raise SystemExit(f"비교용 마스크 없음 {missing} — 먼저 --make")
    _M.update(E=E, maps=dict(noroom=E._G["maps"], **{v: load_maps(variant_meta(v)) for v in VARIANTS}))


def _job(c):
    E = _M["E"]
    c0, m, lg, r = E._job(dict(c, ret_r=True), None, _M["maps"][c["mask"]])
    walk = ~r["esc"] & ~r["riding"] & ~paths.excluded(r["t"])
    m["unc_med"] = float(np.median(r["unc"][walk]))
    c0.pop("ret_r", None)
    return c0, m, lg


def conds(seeds):
    base = dict(steps="pdr", wall="kill", res="force", init="redraw", k="free", bfix="on", set="기본")
    return [dict(base, mask=mk, vis=v, name=f"{mk}/{v}", seed=s) for mk in MASKS for v, _ in VIS for s in range(seeds)]


def geometry():
    """정답 궤적(탑승·채점 제외 구간 뺌)이 각 마스크 밖을 지나는 비율과 구간"""
    from station_pkg import enu_to_aligned
    E = _M["E"]
    tr = E._G["truth"]
    rot = E._G["smeta"]["aligned_from_enu_rot_deg"]
    keep = (tr["on_esc"] == 0) & ~paths.excluded(tr["t"])
    xy = enu_to_aligned(np.c_[tr["east"], tr["north"]], rot)
    out = dict(n=int(keep.sum()))
    for mk in MASKS:
        bad = np.zeros(len(tr), bool)
        for fl in np.unique(tr["floor"]):
            s = keep & (tr["floor"] == fl)
            bad[s] = ~_M["maps"][mk][int(fl)].free(xy[s])
        spans = []
        for x in tr["t"][bad]:
            if spans and x - spans[-1][1] <= 0.25:
                spans[-1][1] = float(x)
            else:
                spans.append([float(x), float(x)])
        out[mk] = dict(frac=float(bad[keep].mean()), spans=spans[:20])
    return out


def main_one(a):
    from pool_guard import run_pool
    res = run_pool(a.procs, _init, (a.n,), _job, conds(a.seeds))
    _init(a.n)
    R = {}
    for c, m, lg in res:
        R.setdefault(c["name"], dict(cond={k: v for k, v in c.items() if k not in ("seed", "name")}, seeds=[]))["seeds"].append(dict(seed=c["seed"], m=m, log=lg))
    out = paths.REC_OUT / NAME
    out.mkdir(parents=True, exist_ok=True)
    masks = {v: json.loads(variant_meta(v).read_text(encoding="utf-8"))["plan_mask"] for v in VARIANTS}
    info = dict(rec=paths.RECORDING, n=a.n, seeds=a.seeds, exclude=paths.RC["exclude"], floors=_M["E"]._G["floors"], geometry=geometry(), masks=masks)
    (out / "result.json").write_text(json.dumps(dict(info=info, runs=R), ensure_ascii=False), encoding="utf-8")
    print(paths.RECORDING, "→", out / "result.json", len(res), "runs", flush=True)


# ---------------------------------------------------------------- 보고
def report():
    from provenance import RECORDINGS
    from exp_main_recheck_report import med, paired, wrong, SHORT, rec_dir as rc_main
    Q = ("med_all", "p90_all", "max_all", "cover2", "unc_med")
    rec_dir = lambda rec: rc_main(rec).parent / NAME
    D = {r: json.loads((rec_dir(r) / "result.json").read_text(encoding="utf-8")) for r in RECORDINGS if (rec_dir(r) / "result.json").exists()}
    N = {r: json.loads((rc_main(r) / "result.json").read_text(encoding="utf-8")) for r in D if (rc_main(r) / "result.json").exists()}
    J = {}
    one = next(iter(D.values()))["info"]
    L = ["# 보행 마스크 비교: 지금(방 뺀 사본) ↔ 정식 도면에서 만든 마스크 (exp_plan_mask)", "",
         f"기록 {len(D)}개, seed {one['seeds']}개 중앙, N {one['n']}, 나머지 exp_d58_r20 그대로. 차이 = 같은 seed 짝 (새 − 지금).", "",
         "## 마스크", "", "| 이름 | 벽 띠 | 보행칸 넓힘 | B01 넓이 | B02 넓이 | 버린 조각 B01 / B02 | 개찰구로 연 넓이 B01 |", "|---|---|---|---|---|---|---|"]
    for v, mk in one["masks"].items():
        r = mk["report"]
        L.append(f"| {v} | {mk['thick_m']} m | {mk['grow_m']} m | {r['B01']['mask_m2']:.0f} m² | {r['B02']['mask_m2']:.0f} m² | "
                 f"{r['B01']['dropped_m2']:.0f} / {r['B02']['dropped_m2']:.0f} m² | {r['B01']['opened_m2']:.1f} m² |")
    L.append("")
    for vis, vn in VIS:
        for v in VARIANTS:
            L += [f"## {v} − 지금 ({vn})", "",
                  "| 기록 | 오차 중앙 / 90% / 최대 지금 | 차이 중앙 / 90% / 최대 | cover2 차이 | 불확실도 중앙 지금 → 새 | 자신 있게 틀림 지금 → 새 | 벽 막힘 지금 → 새 | 모두 막힘 지금 → 새 | 평균 k 지금 → 새 | 잡음 90% |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for rec, d in D.items():
                a, b = d["runs"][f"noroom/{vis}"], d["runs"][f"{v}/{vis}"]
                dd = {q: paired(a, b, q) for q in Q}
                noise = None
                if rec in N and f"noise/{vis}" in N[rec]["runs"]:
                    noise = abs(med(N[rec]["runs"][f"pdr/kill/{vis}"], "p90_all") - med(N[rec]["runs"][f"noise/{vis}"], "p90_all"))
                J.setdefault(rec, {})[f"{v}/{vis}"] = dict(
                    base={q: med(a, q) for q in Q}, diff=dd, wrong=(wrong(a), wrong(b)),
                    kills=(med(a, "wall_kills", "log"), med(b, "wall_kills", "log")),
                    all_dead=(med(a, "all_dead", "log"), med(b, "all_dead", "log")),
                    k_mean=(med(a, "k_mean_walk", "log"), med(b, "k_mean_walk", "log")), noise_p90=noise)
                L.append(f"| {SHORT(rec)} | {med(a, 'med_all'):.2f} / {med(a, 'p90_all'):.2f} / {med(a, 'max_all'):.2f} | "
                         f"{dd['med_all']:+.2f} / {dd['p90_all']:+.2f} / {dd['max_all']:+.2f} | {100 * dd['cover2']:+.0f}%p | "
                         f"{med(a, 'unc_med'):.2f} → {med(b, 'unc_med'):.2f} | {wrong(a)} → {wrong(b)} | "
                         f"{med(a, 'wall_kills', 'log'):.0f} → {med(b, 'wall_kills', 'log'):.0f} | {med(a, 'all_dead', 'log'):.0f} → {med(b, 'all_dead', 'log'):.0f} | "
                         f"{med(a, 'k_mean_walk', 'log'):.3f} → {med(b, 'k_mean_walk', 'log'):.3f} | {'—' if noise is None else f'{noise:.2f}'} |")
            L.append("")
    L += ["## 정답 궤적이 마스크 밖을 지나는 비율", "", "| 기록 | 정답 점 수 | " + " | ".join(MASKS) + " |", "|---|---|" + "---|" * len(MASKS)]
    for rec, d in D.items():
        g = d["info"]["geometry"]
        J[rec]["geom"] = {k: g[k] for k in MASKS}
        sp = lambda k: f"{100 * g[k]['frac']:.1f}%" + (" (" + ", ".join(f"{a:.1f}~{b:.1f} s" for a, b in g[k]["spans"][:4]) + ")" if g[k]["spans"] else "")
        L.append(f"| {SHORT(rec)} | {g['n']} | " + " | ".join(sp(k) for k in MASKS) + " |")
    # 판단
    L += ["", "## 판단 (기준은 재생 전에 정함, 시간 필터·기본)", "",
          "| 마스크 | 1. 평균 오차 중앙 차이 | 1. 평균 90% 차이 | 2. 90%가 크게 나빠진 기록 | 3. 자신 있게 틀림 합 지금 → 새 | 4. 정답이 밖을 더 지나는 기록 | 결과 |",
          "|---|---|---|---|---|---|---|"]
    verdict = {}
    for v in VARIANTS:
        rows = {rec: J[rec][f"{v}/tf"] for rec in D}
        d_med = float(np.mean([r["diff"]["med_all"] for r in rows.values()]))
        d_p90 = float(np.mean([r["diff"]["p90_all"] for r in rows.values()]))
        worse = [SHORT(rec) for rec, r in rows.items()
                 if r["diff"]["p90_all"] > max(LIMIT["rec_p90"], LIMIT["rec_noise_x"] * (r["noise_p90"] or 0.0))]
        w0, w1 = sum(r["wrong"][0] for r in rows.values()), sum(r["wrong"][1] for r in rows.values())
        out_more = [SHORT(rec) for rec in D if J[rec]["geom"][v]["frac"] > J[rec]["geom"]["noroom"]["frac"] + 1e-9]
        ok = [d_med <= LIMIT["mean_med"] and d_p90 <= LIMIT["mean_p90"], not worse, w1 <= w0 + LIMIT["wrong"], not out_more]
        verdict[v] = dict(mean_med=d_med, mean_p90=d_p90, worse=worse, wrong=(w0, w1), out_more=out_more, ok=ok, passed=all(ok))
        L.append(f"| {v} | {d_med:+.3f} m | {d_p90:+.3f} m | {', '.join(worse) or '없음'} | {w0} → {w1} | {', '.join(out_more) or '없음'} | "
                 f"{'기준 지킴' if all(ok) else '기준 못 지킴 (' + ', '.join(str(i + 1) for i, o in enumerate(ok) if not o) + '번)'} |")
    J["verdict"] = dict(limit=LIMIT, masks=verdict)
    out = paths.OUT
    (out / f"{NAME}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"{NAME}.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--make", action="store_true", help="비교용 마스크를 만듦")
    ap.add_argument("--gates", help="--make: 개찰구 줄 파일 (기본 vision/inputs/graph/gates.json)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.make:
        return make(a.gates)
    if a.report:
        return report()
    main_one(a)


if __name__ == "__main__":
    main()
