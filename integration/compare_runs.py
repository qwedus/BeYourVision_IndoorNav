"""조건 비교: 두 재생 결과(runs.npz)를 같은 seed끼리 짝지어 차이를 낸다. 위치 결과는 안 바꾸고 저장된 배열만 읽음.

    .venv/Scripts/python integration/compare_runs.py --rec s2_a_ex3_ex6 --a exp_base --b exp_d58_r20
    ... --rec all --a exp_base --b exp_d58_r20                기록 9개, 끝에 기록별 요약
    ... --range 38-53 75.6-111.7 --seg 10 --out 비교.md       구간을 직접 지정, 칸 길이 10 s
--a(기준)·--b(비교): 기록 out 아래 폴더 이름, 폴더 경로, npz 경로. --cond: runs.npz 조건 이름(기본 known_tf_기본).
차이 = B - A. seed마다 낸 차이의 중앙·범위·방향 일치 수를 표로 낸다(seed가 같으면 초기 파티클이 같아 짝이 뜻을 가짐).
    지표: 오차 중앙·90%, cover2(오차 ≤ 2×불확실도), in95(오차 ≤ unc95), 자신 있게 틀린 시간(불확실도 < 2 m인데 오차 > 3 m). 정의는 replay.metrics와 같음
    구간: 칸마다 걷는 프레임의 오차 중앙을 seed별로 비교. 차이 중앙이 --bad(기본 0.5 m) 이상이고 seed 80% 이상 같은 방향이면 '나빠짐'·'좋아짐',
          크기는 넘는데 seed가 갈리면 'seed 갈림'. 전체 p90이 좋아져도 어느 구간이 나빠졌는지 보려고 둠
주의: seed 범위는 같은 입력에서의 PF 표본 잡음이다. 새 보행·입력 변화의 불확실성이 아니다. 이 지표는 모두 정답에 의존한다.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import paths
import replay as rp
from provenance import RECORDINGS

# (이름, 제목, 나은 방향, 소수 자리)
METRICS = (("med_all", "오차 중앙 (m)", -1, 2), ("p90_all", "오차 90% (m)", -1, 2), ("cover2", "cover2", 1, 3), ("in95", "in95", 1, 3),
           ("wrong_s", "자신 있게 틀린 시간 (s)", -1, 1))
AGREE = 0.8      # 구간 표시에 필요한 seed 방향 일치 비율
ZERO = 1e-6      # 이보다 작은 차이는 변화 없음


def use_recording(name):
    """paths의 녹화 값을 바꿈 (채점 제외 구간·결과 폴더). 한 프로세스에서 여러 기록을 볼 때"""
    paths.RECORDING = name
    paths.RC = json.loads((paths.RECORDINGS / f"{name}.json").read_text(encoding="utf-8"))
    paths.REC_OUT = paths.ROOT / paths.RC["out"]


def npz_path(name):
    p = Path(name)
    if p.suffix == ".npz":
        return p
    return (p if p.is_dir() else paths.REC_OUT / name) / "runs.npz"


def load_runs(path, cond):
    """→ {seed: 기록 배열}"""
    if not path.exists():
        raise SystemExit(f"결과 파일 없음: {paths.RECORDING} {path.parent.name}/{path.name}")
    z = np.load(path)
    runs = {int(k.rsplit("_", 1)[1]): z[k] for k in z.files if k.rsplit("_", 1)[0] == cond}
    if not runs:
        names = sorted({k.rsplit("_", 1)[0] for k in z.files if not k.startswith("x_")})
        raise SystemExit(f"{path.parent.name}에 조건 {cond} 없음. 있는 조건: {names}")
    return runs


def scored_dt(t):
    """프레임마다 다음 프레임까지 시간에서 채점 제외 구간과 겹치는 만큼 뺀 값 (프레임 사이가 제외 구간을 가로지를 때 제외 시간을 안 세게)"""
    t = np.asarray(t, float)
    t1 = np.r_[t[1:], t[-1]] if len(t) else t
    dt = t1 - t
    spans = []
    for a, b in sorted(paths.RC["exclude"]):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    for a, b in spans:
        dt = dt - np.clip(np.minimum(t1, b) - np.maximum(t, a), 0.0, None)
    return dt


def seed_metrics(r):
    """seed 하나의 지표. wrong_s: 불확실도 < 2 m인데 오차 > 3 m였던 시간 합(cover2와 같은 프레임 범위)"""
    m = rp.metrics(r)
    t = r["t"]
    cw = ~r["esc"] & ~r["riding"] & ~paths.excluded(t) & (t > 2)
    return dict(med_all=m["med_all"], p90_all=m["p90_all"], cover2=m["cover2"], in95=m["in95"],
                wrong_s=float(scored_dt(t)[cw & (r["unc"] < 2) & (r["err"] > 3)].sum()))


def paired(da, db, better=-1):
    """같은 seed 짝 차이 d = B - A 요약. better: 나은 방향 부호. → 중앙·범위·방향"""
    d = np.asarray(db, float) - np.asarray(da, float)
    med = float(np.median(d))
    sign = 0 if abs(med) < ZERO else int(np.sign(med))
    agree = int(np.sum(np.abs(d) < ZERO)) if sign == 0 else int(np.sum(sign * d >= ZERO))
    return dict(a=float(np.median(da)), b=float(np.median(db)), med=med, lo=float(d.min()), hi=float(d.max()), n=len(d), agree=agree,
                dir="변화 없음" if sign == 0 else "좋아짐" if sign == better else "나빠짐")


def segment_err(r, t0, t1):
    """구간 [t0, t1) 안 걷는 프레임의 오차 중앙. 프레임이 3개 안 되면 None"""
    t = r["t"]
    sel = ~r["esc"] & ~r["riding"] & ~paths.excluded(t) & (t >= t0) & (t < t1)
    return float(np.median(r["err"][sel])) if sel.sum() >= 3 else None


def segment_row(A, B, seeds, t0, t1, bad):
    pairs = [(segment_err(A[s], t0, t1), segment_err(B[s], t0, t1)) for s in seeds]
    pairs = [p for p in pairs if None not in p]
    if not pairs:
        return None
    r = paired([p[0] for p in pairs], [p[1] for p in pairs])
    flag = "비슷"
    if abs(r["med"]) >= bad:
        flag = r["dir"] if r["agree"] >= math.ceil(AGREE * r["n"]) else "seed 갈림"
    return dict(r, t0=float(t0), t1=float(t1), flag=flag)


def input_diff(pa, pb):
    """두 결과 옆 summary.json의 설정·입력 기록 차이. 옛 결과엔 입력 기록이 없을 수 있음.
    config: 양쪽에 있는데 값이 다른 칸(실제 PF 값 pf_params 포함), only_a·only_b: 한쪽에만 있는 칸(실행기가 달라도 생김, 실험 설정일 수 있어 숨기지 않음)"""
    def load(p):
        s = p.parent / "summary.json"
        return json.loads(s.read_text(encoding="utf-8")).get("info", {}) if s.exists() else {}
    ia, ib = load(pa), load(pb)
    skip = {"name", "provenance", "steps", "rides", "floors"}
    cfg = {k: (ia[k], ib[k]) for k in sorted(set(ia) & set(ib)) if k not in skip and ia[k] != ib[k]}
    only_a = {k: ia[k] for k in sorted(set(ia) - set(ib)) if k not in skip}
    only_b = {k: ib[k] for k in sorted(set(ib) - set(ia)) if k not in skip}
    pra, prb = ia.get("provenance"), ib.get("provenance")
    out = dict(config=cfg, only_a=only_a, only_b=only_b, a_recorded=bool(pra), b_recorded=bool(prb), files=[], sources=None)
    if pra and prb:
        for sect in ("pf_params",):
            sa, sb = pra.get(sect) or {}, prb.get(sect) or {}
            cfg.update({f"{sect}.{k}": (sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)})
        cfg.update({f"환경.{k}": (pra[k], prb[k]) for k in ("recording", "rec_t0", "python", "numpy", "scipy") if k in pra and k in prb and pra[k] != prb[k]})
        fa, fb = pra.get("files", {}), prb.get("files", {})
        out["files"] = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        if pra.get("source_digest") and prb.get("source_digest"):
            out["sources"] = pra["source_digest"] == prb["source_digest"]
    return out


def compare(rec, a, b, cond="known_tf_기본", seg=10.0, ranges=(), bad=0.5):
    """기록 하나 비교 → 결과 dict"""
    use_recording(rec)
    pa, pb = npz_path(a), npz_path(b)
    A, B = load_runs(pa, cond), load_runs(pb, cond)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit(f"{rec}: 짝지을 seed 없음 (A {sorted(A)}, B {sorted(B)})")
    notes = []
    if set(A) != set(B):
        notes.append(f"seed가 달라 겹치는 {len(seeds)}개만 비교함 (A {len(A)}개, B {len(B)}개)")
    if any(len(A[s]) != len(B[s]) or not np.array_equal(A[s]["t"], B[s]["t"]) for s in seeds):
        notes.append("프레임 시각이 달라 입력 프레임이 다른 결과임 (구간 비교는 시각 기준)")
    ma, mb = {s: seed_metrics(A[s]) for s in seeds}, {s: seed_metrics(B[s]) for s in seeds}
    table = []
    for key, title, better, digits in METRICS:
        table.append(dict(paired([ma[s][key] for s in seeds], [mb[s][key] for s in seeds], better), key=key, title=title, digits=digits))
    tmax = float(max(max(r["t"][-1] for r in A.values()), max(r["t"][-1] for r in B.values())))
    edges = np.arange(0.0, tmax + seg, seg)
    bins = [segment_row(A, B, seeds, x, y, bad) for x, y in zip(edges[:-1], edges[1:])]
    picked = [segment_row(A, B, seeds, x, y, bad) for x, y in ranges]
    return dict(rec=rec, a=pa.parent.name, b=pb.parent.name, cond=cond, seeds=seeds, notes=notes, table=table, seg=seg, bad=bad,
                bins=[r for r in bins if r], picked=[r for r in picked if r], inputs=input_diff(pa, pb))


def fmt_row(r, digits=2):
    x = f".{digits}f"
    return f"{r['a']:{x}} | {r['b']:{x}} | {r['med']:+{x}} | {r['lo']:+{x}} ~ {r['hi']:+{x}} | {r['dir']} {r['agree']}/{r['n']} |"


def seg_lines(rows):
    L = ["| 구간 (s) | A 오차 중앙 | B 오차 중앙 | 차이 중앙 | seed 범위 | 방향 일치 | 표시 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['t0']:g}~{r['t1']:g} | {fmt_row(r)} {r['flag']} |")
    return L


def render(res):
    n = len(res["seeds"])
    L = [f"## {res['rec']}: A {res['a']} → B {res['b']}", "",
         f"조건 {res['cond']}, 같은 seed {n}개 짝. 차이 = B − A. seed 범위는 같은 입력의 PF 표본 잡음이며 입력 변화·새 보행의 불확실성이 아님.", ""]
    for x in res["notes"]:
        L += [f"**주의**: {x}", ""]
    inp = res["inputs"]
    if inp["config"]:
        L.append("설정 차이: " + ", ".join(f"{k} {v[0]} → {v[1]}" for k, v in inp["config"].items()))
    for side, key in (("A", "only_a"), ("B", "only_b")):
        if inp[key]:
            L.append(f"{side}에만 있는 설정: " + ", ".join(f"{k} = {v}" for k, v in inp[key].items()))
    if inp["a_recorded"] and inp["b_recorded"]:
        L.append("입력 파일 차이: " + (", ".join(inp["files"]) if inp["files"] else "없음"))
        if inp["sources"] is not None:
            L.append("코드: " + ("같음" if inp["sources"] else "다름"))
    else:
        L.append(f"입력 기록: A {'있음' if inp['a_recorded'] else '없음'}, B {'있음' if inp['b_recorded'] else '없음'} (없는 쪽은 입력 파일을 대조할 수 없음)")
    L += ["", "| 지표 | A 중앙 | B 중앙 | 차이 중앙 | seed 범위 | 방향 일치 |", "|---|---|---|---|---|---|"]
    for r in res["table"]:
        L.append(f"| {r['title']} | {fmt_row(r, r['digits'])}")
    cnt = {k: sum(r["flag"] == k for r in res["bins"]) for k in ("나빠짐", "좋아짐", "seed 갈림", "비슷")}
    L += ["", f"구간 {res['seg']:g} s 칸 {len(res['bins'])}개: 나빠짐 {cnt['나빠짐']}, 좋아짐 {cnt['좋아짐']}, seed 갈림 {cnt['seed 갈림']}, 비슷 {cnt['비슷']} (표시 기준 {res['bad']:g} m)", ""]
    order = {"나빠짐": 0, "seed 갈림": 1, "좋아짐": 2}
    flagged = sorted((r for r in res["bins"] if r["flag"] in order), key=lambda r: (order[r["flag"]], r["t0"]))
    if flagged:
        L += seg_lines(flagged) + [""]
    if res["picked"]:
        L += ["지정 구간", ""] + seg_lines(res["picked"]) + [""]
    return L


def summary_lines(results):
    L = ["## 기록별 요약", "", "| 기록 | 오차 90% 차이 중앙 | 방향 | cover2 차이 | 틀린 시간 차이 (s) | 나빠진 칸 | 좋아진 칸 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        t = {x["key"]: x for x in r["table"]}
        L.append(f"| {r['rec']} | {t['p90_all']['med']:+.2f} | {t['p90_all']['dir']} {t['p90_all']['agree']}/{t['p90_all']['n']} | {t['cover2']['med']:+.3f} | "
                 f"{t['wrong_s']['med']:+.1f} | {sum(x['flag'] == '나빠짐' for x in r['bins'])} | {sum(x['flag'] == '좋아짐' for x in r['bins'])} |")
    return L + [""]


def parse_range(s):
    a, b = s.split("-")
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rec", nargs="+", required=True, help="기록 이름, all이면 9개")
    ap.add_argument("--a", required=True, help="기준 결과")
    ap.add_argument("--b", required=True, help="비교 결과")
    ap.add_argument("--cond", default="known_tf_기본")
    ap.add_argument("--seg", type=float, default=10.0, help="구간 칸 길이 (s)")
    ap.add_argument("--range", nargs="*", default=[], type=parse_range, help="직접 지정한 구간 38-53 ...")
    ap.add_argument("--bad", type=float, default=0.5, help="구간 표시 기준 (m)")
    ap.add_argument("--out", help="표를 저장할 md 파일")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    recs = list(RECORDINGS) if a.rec == ["all"] else a.rec
    results = [compare(r, a.a, a.b, a.cond, a.seg, a.range, a.bad) for r in recs]
    L = []
    for r in results:
        L += render(r)
    if len(results) > 1:
        L += summary_lines(results)
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
