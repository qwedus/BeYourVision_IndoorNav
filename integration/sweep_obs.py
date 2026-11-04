"""관측 설정 다시 맞추기 (작업기록 3차 7번, 계획 '종류별 문턱값 순서' ③): 초당 횟수·방위 σ·바닥값을 앞 단계까지 채택한 상태에서 훑는다.

    .venv/Scripts/python integration/sweep_obs.py --name <이름> [--cls-th ...]
앞 단계: 걸음 사이 회전 보정(방향으로) + track별 반복 제한 1 s. indoornav PDR + PF, 출발점 앎·PDR 틀어짐, seed 10, N 3000.
출력: integration/out/i1/<이름>.md·.json. 같은 녹화로 고르고 평가 → 낙관적일 수 있음, 손에 든 촬영이라 조끼 성능 아님.
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool

import numpy as np

import paths
import replay as rp
import run_i1 as R

HZ = (None, 3.0, 2.0, 1.0)
SIGMA = (5.0, 7.0, 10.0)
EPS = (0.1, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cls-th")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    th = {k: float(v) for k, v in (x.split("=") for x in a.cls_th.split(","))} if a.cls_th else None
    opt = dict(bearing_fix="theta", track_limit=1.0, cls_th=th)
    conds = []
    for hz, sg, ep in itertools.product(HZ, SIGMA, EPS):
        st = dict(obs_params={"sigma_bearing": sg, "eps": ep}, obs_hz=hz)
        conds += [dict(sc=sc, obs=o, set=f"{hz or 5:g}|{sg:g}|{ep:g}", st=st, seed=s)
                  for sc in R.SCENARIOS for o in ("raw", "tf") for s in range(a.seeds)]
    with Pool(10, R._init, ("indoornav", 3000, None, opt)) as pool:
        res = pool.map(R._job, conds, chunksize=4)
    ms = {}
    for c, r, _ in res:
        ms.setdefault((c["set"], c["obs"], c["sc"]), []).append(rp.metrics(r))
    rows = []
    for (st, o, sc), m in ms.items():
        s = rp.summarize(m)
        rows.append(dict(set=st, obs=o, sc=sc, med=s["med_all"], p90=s["p90_all"], max=s["max_all"], end=s["end"], ratio=s["ratio"],
                         cover2=s["cover2"], in95=s["in95"], start=s["start_med"], wrong=s["wrong"]))
    (paths.OUT / "i1" / f"{a.name}.json").write_text(json.dumps(dict(opt=opt, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# 관측 설정 훑기 ({a.name})", "", "녹화 1회 재생(손에 든 촬영)이라 낙관적일 수 있고 조끼 성능이 아님. 같은 녹화로 고르고 평가함. "
         f"앞 단계: 회전 보정 + track 제한 1 s, 문턱 {th or '모두 0.25'}. seed {a.seeds}개 중앙, N 3000.", "",
         "칸 = 오차 중앙 / 90% / 최대 · 끝 (m), 오차÷불확실도, 오차 ≤ 2×불확실도, unc95 포함률, 자신 있게 틀림", ""]
    for o in ("tf", "raw"):
        L += [f"## {'시간 필터' if o == 'tf' else '필터 없음'}", "", "| 초당 | σ (°) | 바닥값 | 출발점 앎 | PDR 틀어짐 |", "|---|---|---|---|---|"]
        by = {(r["set"], r["sc"]): r for r in rows if r["obs"] == o}
        for hz, sg, ep in itertools.product(HZ, SIGMA, EPS):
            st = f"{hz or 5:g}|{sg:g}|{ep:g}"
            cell = lambda r: (f"{r['med']:.2f} / {r['p90']:.1f} / {r['max']:.1f} · {r['end']:.1f}, {r['ratio']:.1f}, "
                              f"{100 * r['cover2']:.0f}%, {100 * r['in95']:.0f}%, {r['wrong']}/{a.seeds}")
            L.append(f"| {hz or 5:g} | {sg:g} | {ep:g} | {cell(by[(st, 'known')])} | {cell(by[(st, 'drift')])} |")
        L.append("")
    (paths.OUT / "i1" / f"{a.name}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
