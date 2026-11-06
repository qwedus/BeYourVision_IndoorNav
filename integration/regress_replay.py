"""회귀 확인: 기본 재생(출발점 앎·시간 필터·기본, seed 10개, 채택 비전 옵션)을 다시 돌려 저장된 runs.npz와 err·unc 배열이 같은지 본다.

    .venv/Scripts/python integration/regress_replay.py --ref exp_d58_r20                   # 9개 기록 전부
    .venv/Scripts/python integration/regress_replay.py --ref exp_d58_r20 --rec s2_a_ex6_ex3 --save <폴더>

--ref: 각 기록 out 아래 비교할 결과 폴더 이름. --ref-dir <폴더>: <폴더>/<기록>.npz와 비교(--save로 저장한 것).
--save: 이번 재생 배열을 <폴더>/<기록>.npz로 저장(같은 키 이름 known_tf_기본_<seed>).
PF·관측·설정은 아무것도 안 바꾼다. 같은 입력·같은 seed면 기록의 모든 필드(좌표·층·시각 포함)가 비트까지 같아야 하며, 다르면 코드 1을 돌려준다.
저장 경로가 기준 파일과 같으면 기준을 덮어쓰게 되므로 재생 전에 멈춘다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from provenance import RECORDINGS

HERE = Path(__file__).resolve().parent
KEY = "known_tf_기본"
SHOWN = ("err", "unc", "ex", "ey")      # 차이 크기를 보여 줄 필드


def options():
    return json.loads((HERE / "vision_options.json").read_text(encoding="utf-8"))


def replay(n, seeds, procs):
    """→ {키: 기록 배열}. run_i1과 같은 _init·_job으로 돌린다."""
    import run_i1
    conds = [dict(sc="known", obs="tf", set="기본", seed=s) for s in range(seeds)]
    from pool_guard import run_pool
    res = run_pool(procs, run_i1._init, ("indoornav", n, None, options()), run_i1._job, conds)
    return {f"{KEY}_{c['seed']}": r for c, r, _ in res}


def diff(new, old_path):
    """→ {seed 키: dict(같음 여부, 다른 필드 목록, 주요 필드 최대 차이)}. 모든 필드를 바이트로 비교(NaN도 같은 값이면 같음)"""
    import numpy as np
    z = np.load(old_path)
    out = {}
    for k, r in new.items():
        if k not in z.files:
            out[k] = dict(same=False, why="기준에 키 없음")
            continue
        o = z[k]
        if len(o) != len(r):
            out[k] = dict(same=False, why=f"길이 다름 {len(r)} vs {len(o)}")
            continue
        if r.dtype != o.dtype:
            out[k] = dict(same=False, why=f"필드 구성 다름 {r.dtype.names} vs {o.dtype.names}")
            continue
        fields = [f for f in r.dtype.names if r[f].tobytes() != o[f].tobytes()]
        out[k] = dict(same=not fields, fields=fields, **{f"max_{f}": float(np.nanmax(np.abs(r[f] - o[f]))) if len(r) else 0.0 for f in SHOWN})
    return out


def same_file(a, b):
    a, b = Path(a), Path(b)
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    return os.path.normcase(a.resolve()) == os.path.normcase(b.resolve())


def one(a):
    """I1_RECORDING 하나를 이 프로세스에서 처리 (paths가 불러올 때 녹화를 정하므로 기록마다 새 프로세스)"""
    import numpy as np
    import paths
    refs = {}
    if a.ref:
        refs[a.ref] = paths.REC_OUT / a.ref / "runs.npz"
    if a.ref_dir:
        refs["ref_dir"] = Path(a.ref_dir) / f"{paths.RECORDING}.npz"
    target = Path(a.save) / f"{paths.RECORDING}.npz" if a.save else None
    for name, p in refs.items():
        if target is not None and same_file(target, p):
            raise SystemExit(f"{paths.RECORDING}: 저장 경로가 기준({name}) 파일과 같음 → 기준을 덮어쓰고 자기와 비교하게 되어 멈춤. --save를 다른 폴더로")
    new = replay(a.n, a.seeds, a.procs)
    res = dict(rec=paths.RECORDING, seeds=len(new), checks={name: diff(new, p) for name, p in refs.items()})
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **new)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", help="각 기록 out 아래 결과 폴더 이름 (예: exp_d58_r20)")
    ap.add_argument("--ref-dir", help="--save로 저장해 둔 폴더")
    ap.add_argument("--save", help="이번 재생 배열을 저장할 폴더")
    ap.add_argument("--rec", nargs="+", default=list(RECORDINGS))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--_one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a._one:
        return one(a)
    bad = 0
    for rec in a.rec:
        cmd = [sys.executable, __file__, "--_one", "--n", str(a.n), "--seeds", str(a.seeds), "--procs", str(a.procs)]
        for flag, v in (("--ref", a.ref), ("--ref-dir", a.ref_dir), ("--save", a.save)):
            if v:
                cmd += [flag, str(v)]
        p = subprocess.run(cmd, env={**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8"}, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        if p.returncode or line is None:
            print(f"{rec}: 실패 (코드 {p.returncode})\n{p.stdout[-800:]}\n{p.stderr[-800:]}", flush=True)
            bad += 1
            continue
        res = json.loads(line[7:])
        if not res["checks"]:
            print(f"{rec}: 저장만 함 (seed {res['seeds']}개)", flush=True)
        for name, d in res["checks"].items():
            same = sum(v["same"] for v in d.values())
            worst = {f: max(v.get(f"max_{f}", float("nan")) for v in d.values()) for f in SHOWN}
            fields = sorted({f for v in d.values() for f in v.get("fields", [])})
            why = sorted({v["why"] for v in d.values() if "why" in v})
            print(f"{rec} vs {name}: 전체 필드 같음 {same}/{len(d)} seed, 최대 차이 err {worst['err']:.6g}, unc {worst['unc']:.6g}, 좌표 {worst['ex']:.6g}·{worst['ey']:.6g} m"
                  + (f", 다른 필드 {fields}" if fields else "") + (f", {why}" if why else ""), flush=True)
            bad += same != len(d)
    print("회귀 확인: " + ("모두 같음" if not bad else f"다른 기록 {bad}개"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
