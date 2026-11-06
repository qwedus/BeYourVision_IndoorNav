"""실험 입력 기록: 재생·실험 결과(summary.json)에 남길 입력 해시·소스 해시, 옛 파일 재사용 검사(run_recording).

    run_record(...)      → summary.json info.provenance. 어떤 입력·코드로 나온 결과인지 나중에 대조할 수 있게 한다
    cache_warning(...)   기록된 입력과 지금 입력이 다르면 경고 문구, 같으면 None
    write_record(...)    파일을 새로 만든 입력을 <입력 폴더>/input_record.json에 남김
결과 파일이 저장소에 들어갈 수 있어 저장소 밖 경로는 폴더 이름까지만 적는다(PC 경로를 남기지 않음).
위치 결과에 영향 없음: 결과가 다 나온 뒤에 파일을 읽기만 한다.
"""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("integration/*.py", "vision/*.py", "indoornav/**/*.py")
RECORD_NAME = "input_record.json"
# 평가에 쓰는 기록: 1차 손 녹화 1개 + 2차 조끼 보행 8개
RECORDINGS = ("route2_1030", "s2_a_71_ex2", "s2_a_ex3_ex6", "s2_a_ex6_ex3", "s2_a_platform", "s2_b_elev_pillar",
              "s2_b_emart_ex5", "s2_b_ex2_31", "s2_b_pillar_map")
INIT = "known: N(정답 t=0, 2² I) 보행 마스크 거절 샘플링, 방향 N(정답, 20°), k ~ U(k0_lo, k0_hi)"


def sha(p):
    """파일 내용 해시. 줄 끝을 LF로 맞춰 잰다(autocrlf로 줄 끝만 바뀌어도 같은 값)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def rel(p):
    """저장소 기준 경로. 저장소 밖이면 폴더 이름까지만."""
    p = Path(p)
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else f"(저장소 밖)/{p.parent.name}/{p.name}"


def file_sig(p):
    """없는 파일은 None (기록을 남기다가 멈추지 않음)"""
    return sha(p) if Path(p).is_file() else None


def file_hashes(files):
    return {rel(f): file_sig(f) for f in files}


def digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]


def tree_sig(folder, pattern="**/*.py"):
    """폴더 안 파일들을 합친 해시. 폴더가 없으면 None"""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    return digest({str(f.relative_to(folder)).replace("\\", "/"): sha(f) for f in folder.glob(pattern) if f.is_file()})


def source_hashes():
    """재생에 쓰는 코드: 연결 코드·비전 코어·indoornav 패키지"""
    return {rel(f): sha(f) for g in SOURCES for f in sorted(ROOT.glob(g))}


def run_record(steps_file, pf_params, seeds, opt=None, pdr_config=None, extra=()):
    """→ provenance dict. steps_file: 이 실행이 읽은 걸음 파일, pf_params: 실제로 쓴 PF 값(dict), opt: 비전 옵션,
    pdr_config: 걸음을 그 자리에서 다시 만들었으면 그 설정(없으면 기본), extra: 그 밖에 읽은 파일."""
    import numpy as np
    import scipy
    import paths
    git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout.strip()
    man = json.loads((paths.VISION / "MANIFEST.json").read_text(encoding="utf-8"))
    files = [steps_file, paths.REC_OUT / "steps_indoornav.csv", paths.REC_OUT / "heading_indoornav.npz",
             paths.I1 / "obs_nofilter.jsonl", paths.I1 / "obs_timefilter.jsonl", paths.I1 / "truth.csv",
             paths.I1 / RECORD_NAME, paths.REC / "baro.csv", paths.INPUTS / "dongchun_station.npz",
             paths.INPUTS / "dongchun_station.json", paths.HERE / "vision_options.json",
             paths.RECORDINGS / f"{paths.RECORDING}.json", paths.MASK_META, paths.PF_CONFIG, pdr_config or paths.PDR_CONFIG,
             *paths.MASK_DIR.glob("*.png"), *extra]
    if (opt or {}).get("text_check"):
        files.append(paths.I1 / opt["text_check"])
    src = source_hashes()
    return dict(branch_commit=git("rev-parse", "HEAD"), branch_dirty=bool(git("status", "--porcelain")),
                main_commit="9040d70", vision_from=man.get("localization_design_head"), recording=paths.RECORDING,
                files=file_hashes(files), source_digest=digest(src), sources=src,
                pf_params=pf_params, python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, seeds=list(range(seeds)), init=INIT)


def read_record(folder):
    p = Path(folder) / RECORD_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_record(folder, what, now):
    Path(folder).mkdir(parents=True, exist_ok=True)
    rec = read_record(folder)
    rec[what] = now
    (Path(folder) / RECORD_NAME).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cache_warning(folder, what, now, label):
    """있는 파일을 그대로 쓸 때 부름. → 경고 문구, 문제없으면 None"""
    old = read_record(folder).get(what)
    if old is None:
        return f"{label}: 입력 기록 없음 → 만들 때의 모델·설정·입력을 확인할 수 없음 (--force로 다시 만들면 기록이 생김)"
    keys = sorted(set(old) | set(now))
    changed = [k for k in keys if old.get(k) is not None and now.get(k) is not None and old[k] != now[k]]
    unknown = [k for k in keys if (old.get(k) is None) != (now.get(k) is None)]
    if not changed and not unknown:
        return None
    msg = [f"{label}: 옛 파일을 그대로 씀"]
    if changed:
        msg.append(f"기록과 달라진 입력 {', '.join(changed)}")
    if unknown:
        msg.append(f"비교할 수 없는 입력 {', '.join(unknown)}")
    return " / ".join(msg) + " (--force로 다시 만들 것)"
