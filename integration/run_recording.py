"""앱 녹화 한 건을 한 번에: zip·폴더 → 파일 검사 → 검출 관측 → PDR → 정답 → 기본 재생 → 실시간 재생 → 경로 그림.

    .venv/Scripts/python integration/run_recording.py <zip 또는 폴더> [--name 이름]
    .venv/Scripts/python integration/run_recording.py <이미 설정이 있는 이름>          (다시 돌리기, 기준점 채운 뒤)
    .venv/Scripts/python integration/run_recording.py s2_b_ex2_31 --data-root <녹화 폴더들의 상위 폴더>
    .venv/Scripts/python integration/run_recording.py all --data-root <상위 폴더>        (2차 보행 8개를 차례로)

--data-root(또는 환경 변수 I1_DATA_ROOT): 설정 integration/recordings/<이름>.json의 session(앱 세션 폴더 이름)을 이 폴더 아래에서 찾음.
없으면 설정 rec(저장소 기준 경로, 이 PC는 integration/out/rec/<이름>/rec 링크)를 씀.

단계 (끝난 단계는 파일이 있으면 건너뜀, --force면 다시. 건너뛸 때 <입력 폴더>/input_record.json의 기록과 지금 입력을 비교해 다르면 경고, 보고서에도 남김)
    1 풀기   zip → integration/out/rec/<이름>/rec/, 폴더는 그 자리를 그대로 씀. 설정 integration/recordings/<이름>.json을 만듦
    2 검사   motion·baro·meta·frames 있는지, 길이·빠진 값 (없으면 이유를 적고 멈춤)
    3 관측   detections.jsonl 있으면 앱 검출 변환(사본 vision/tools/session_to_replay.py → vision/cv_module cvmod replay, 이 가상환경만),
             없고 영상이 있으면 PC 검출(cvmod yolo26n, C2 추가학습 후 모델, 5 fps — torch·cv2 있는 파이썬 I1_VISION_PY와 모델 I1_MODEL 필요),
             둘 다 없으면 멈춤. 그다음 시간 필터
    4 PDR    임시 PDR(정답 보간용 걸음 시각, 사본 vision/pdr_simple.py) + indoornav PDR
    5 정답   설정의 기준점 → truth.csv. 기준점이 없으면 후보·그림을 만들고 무엇을 채울지 알려 준 뒤 멈춤
    6 기본 재생   exp_cause.py --name exp_d58_r20 (출발점 앎, 비전 끔·시간 필터 기본·중간, seed 10개, 채택 비전 옵션 = regress_replay 기준)
    7 실시간 재생 run_rt.py --name rt_compare --stages off,rec --check-ref exp_d58_r20 (off가 6과 비트까지 같은지 확인)
    8 그림   fig_plan.py(exp_d58_r20/fig_plan.png), fig_plan_rt.py(rt_fig/fig_plan_rec.png·fig_plan_compare.png)
    (--old-replay: 6단계를 예전 run_i1 --adopted 재생·fig_i1·report.md로)
    6~8도 결과 파일이 있으면 건너뜀(--force-replay면 다시)

--same-walk <설정 이름>: 같은 걸음을 다시 재생한 기록(앱 재생 모드 등, 시각이 같음)이면 그 설정의 기준점·채점 구간을 가져옴.
PF 동작·관측 모델·비전 설정은 여기서 바꾸지 않는다(채택 설정 integration/vision_options.json 그대로).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

from provenance import binary_sig, cache_warning, file_sig, tree_sig, write_record

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "integration"
RECORDINGS = HERE / "recordings"
LOCAL = ROOT.parent / "localization_design"
CV_MODULE = ROOT / "vision/cv_module"                       # cvmod 사본 (원본 localization_design cv_module, vision/MANIFEST.json)
S2R = ROOT / "vision/tools/session_to_replay.py"            # 사본 (원본 ocr_test_app tools/)
PDR_SIMPLE = ROOT / "vision/pdr_simple.py"                  # 사본 (원본 localization_design cv_test/i1/)
MODEL = Path(os.environ.get("I1_MODEL", LOCAL / "cv_test/out/c2/runs/dc26n_dense_s0/weights/last.pt")).resolve()   # 영상 PC 검출에만
VISION_PY = os.environ.get("I1_VISION_PY", str(Path.home() / "AppData/Local/Programs/Python/Python312/python.exe"))   # 영상 PC 검출에만
REF = "exp_d58_r20"            # 기본 재생 결과 이름 (regress_replay 기준)
RT = "rt_compare"              # 실시간 재생 결과 이름
RT_STAGES = "off,rec"
S2 = ("s2_a_71_ex2", "s2_a_ex3_ex6", "s2_a_ex6_ex3", "s2_a_platform", "s2_b_elev_pillar", "s2_b_emart_ex5", "s2_b_ex2_31",
      "s2_b_pillar_map")      # 'all' = 2차 보행 8개
NEED = ("motion.csv", "baro.csv", "meta.json", "frames.jsonl")
RUN_NAME = "replay"
CAMERA_H, VIDEO_RATE = "1.4", "5"
PDR_CONFIG = ROOT / "configs/pdr_tuned_v4.toml"
WARNINGS = []


class Stop(Exception):
    """이유를 알려 주고 멈춤 (오류가 아니라 다음에 사람이 할 일)"""


def nfc(s):
    return unicodedata.normalize("NFC", s)


def run(cmd, env=None, cwd=None):
    print("  $", " ".join(str(c) for c in cmd)[:220], flush=True)
    r = subprocess.run([str(c) for c in cmd], env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}, cwd=cwd,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit(f"실패 (코드 {r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout


def unpack(src, name, force):
    """zip이면 풀어서, 폴더면 그대로. → 녹화 폴더 (motion.csv가 있는 곳), 설정에 적을 경로"""
    if src.is_dir():
        return src
    dest = HERE / "out/rec" / name / "rec"
    if dest.exists() and any(dest.iterdir()) and not force:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        names = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename]
        for i in names:      # 맨 위 폴더 하나는 벗김
            base = Path(i.filename).name
            with z.open(i) as f, open(dest / nfc(base), "wb") as g:
                shutil.copyfileobj(f, g)
    return dest


def check(rec):
    """→ 검사 결과 dict. 필수 파일이 없으면 Stop"""
    miss = [n for n in NEED if not (rec / n).exists()]
    if miss:
        raise Stop(f"필수 파일 없음: {', '.join(miss)} (폴더 {rec}). 이 기록으로는 PDR·재생을 돌릴 수 없음")
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    motion = (rec / "motion.csv").read_text(encoding="utf-8").splitlines()
    head = motion[0].split(",")
    lack = [c for c in ("t", "gx", "gy", "gz", "ax", "ay", "az", "wx", "wy", "wz") if c not in head]
    if lack:
        raise Stop(f"motion.csv에 열이 없음: {lack}")
    t_m = [float(x.split(",")[0]) for x in (motion[1], motion[-1])]
    baro = (rec / "baro.csv").read_text(encoding="utf-8").splitlines()
    frames = (rec / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    t_f = [json.loads(frames[0])["t"], json.loads(frames[-1])["t"]]
    info = dict(app=meta.get("app"), app_version=meta.get("app_version"), input=(meta.get("input") or {}).get("kind"),
                motion_s=round(t_m[1] - t_m[0], 1), motion_rows=len(motion) - 1, frames=len(frames),
                frames_s=round(t_f[1] - t_f[0], 1), baro_rows=len(baro) - 1,
                video=next((n for n in ("video.mov", "video.mp4") if (rec / n).exists()), None),
                detections=(rec / "detections.jsonl").exists(), marks=(rec / "marks.jsonl").exists())
    if info["motion_rows"] < 200:
        raise Stop(f"motion.csv가 너무 짧음({info['motion_rows']}줄)")
    if info["baro_rows"] < 3:
        print("  주의: 기압 기록이 거의 없음 → 층 이동을 못 찾음")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def warn_cache(inputs, what, now, label):
    """있는 파일을 그대로 쓸 때: 기록된 입력과 다르면 경고"""
    w = cache_warning(inputs, what, now, label)
    if w:
        print("  경고:", w)
        WARNINGS.append(w)


def obs_inputs(rec, info):
    """cv_raw.json을 만드는 입력. 다른 저장소 파일은 없으면 None"""
    common = dict(cv_config=file_sig(CV_MODULE / "config/default.json"), cvmod=tree_sig(CV_MODULE / "cvmod"), camera_height=CAMERA_H)
    if info["detections"]:
        return dict(common, how="앱 검출", detections=file_sig(rec / "detections.jsonl"), frames=file_sig(rec / "frames.jsonl"),
                    meta=file_sig(rec / "meta.json"), session_to_replay=file_sig(S2R))
    if info["video"]:
        return dict(common, how="PC 검출", video=binary_sig(rec / info["video"]), frames=file_sig(rec / "frames.jsonl"), meta=file_sig(rec / "meta.json"),
                    model=file_sig(MODEL), rate=VIDEO_RATE)
    return dict(common, how="없음")


def steps_simple_inputs(rec):
    return dict(motion=file_sig(rec / "motion.csv"), pdr_simple=file_sig(PDR_SIMPLE))


def steps_indoornav_inputs(rec):
    return dict(motion=file_sig(rec / "motion.csv"), baro=file_sig(rec / "baro.csv"), meta=file_sig(rec / "meta.json"), config=file_sig(PDR_CONFIG),
                pdr_steps=file_sig(HERE / "pdr_steps.py"), app_session=file_sig(HERE / "app_session.py"), indoornav_pdr=tree_sig(ROOT / "indoornav/pdr"))


def observations(rec, inputs, info, env, force):
    """→ 관측 출처 설명. inputs/obs_nofilter.jsonl·obs_timefilter.jsonl"""
    raw = inputs / "cv_raw.json"
    inputs.mkdir(parents=True, exist_ok=True)
    src = None
    now = obs_inputs(rec, info)
    if raw.exists() and not force:
        src = "있는 파일"
        warn_cache(inputs, "cv_raw", now, "관측 cv_raw.json")
    elif info["detections"]:
        replay = inputs / "replay_input.json"
        run([sys.executable, S2R, rec, "--out", replay, "--which", "all", "--cv-module", CV_MODULE, "--camera-height", CAMERA_H])
        run([sys.executable, "-m", "cvmod.run", "--detector", "replay", "--replay", replay, "--out", raw], cwd=CV_MODULE)
        src = "앱 검출(detections.jsonl) → PC 문턱값"
    elif info["video"]:
        if not MODEL.exists():
            raise Stop(f"검출 기록(detections.jsonl)이 없어 영상으로 PC 검출을 해야 하는데 모델이 없음: {MODEL.name} (I1_MODEL로 지정)")
        if not (Path(VISION_PY).exists() or shutil.which(VISION_PY)):
            raise Stop("검출 기록(detections.jsonl)이 없어 영상으로 PC 검출을 해야 하는데 torch·ultralytics·cv2 있는 파이썬이 없음 (I1_VISION_PY로 지정)")
        run([VISION_PY, "-m", "cvmod.run", "--detector", "yolo26n", "--model", MODEL, "--recording", rec, "--rate", VIDEO_RATE,
             "--pick", "gyro", "--frames-out", inputs / "frames_5fps_gyro", "--camera-height", CAMERA_H, "--out", raw],
            cwd=CV_MODULE)
        src = "PC 검출(영상, C2 추가학습 후, 5 fps)"
    else:
        raise Stop("검출 기록(detections.jsonl)도 영상도 없어 관측을 만들 수 없음. 영상까지 내보낸 기록이 필요함")
    if src != "있는 파일":
        write_record(inputs, "cv_raw", now)
    print(run([sys.executable, HERE / "_obs_files.py", raw, inputs], env=env).strip())
    return src


def write_report(name, src, info, out):
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]
    SC = {"known": "출발점 앎", "drift": "PDR 틀어짐(보폭 ×1.15, 0.4°/걸음)"}
    rows = [("pdr_only", "PDR만"), ("none", "PF 비전 끔"), ("tf/기본", "PF + 비전 시간 필터·기본"), ("tf/중간", "PF + 비전 시간 필터·중간"),
            ("raw/기본", "PF + 비전 필터 없음·기본"), ("raw/중간", "PF + 비전 필터 없음·중간")]
    f = lambda v: "—" if v is None else f"{v:.1f}"
    L = [f"# {name} 재생 결과", "",
         f"앱 {info.get('app')} {info.get('app_version')}, 프레임 {info.get('frames')}, 관측 출처: {src}. seed 10개 중앙, 비전 설정 = 채택 상태.",
         "자신 있게 틀림: 걷는 동안 불확실도 < 2 m인데 오차 > 3 m인 프레임이 있었던 시행 수. 녹화 하나 결과라 낙관적일 수 있음.", ""]
    for sc, title in SC.items():
        L += [f"## {title}", "", "| 조건 | 오차 중앙 / 90% / 최대 (m) | B1 / B2 중앙 (m) | 끝 오차 (m) | 층 맞음 | 자신 있게 틀림 |", "|---|---|---|---|---|---|"]
        for k, lab in rows:
            m = s.get(f"{sc}/{k}")
            if not m:
                continue
            fl = "—" if m.get("floor_ok") is None else f"{m['floor_ok']:.2f}"
            wrong = f"{m['wrong']}/{m['n']}" if "wrong" in m else "—"
            L.append(f"| {lab} | {f(m['med_all'])} / {f(m['p90_all'])} / {f(m['max_all'])} | {f(m['b1_med'])} / {f(m['b2_med'])} | "
                     f"{f(m['end'])} | {fl} | {wrong} |")
        L.append("")
    if WARNINGS:
        L += ["**입력 확인 경고**: 옛 파일을 다시 썼고 입력을 확인하지 못했거나 달라짐", "", *[f"- {w}" for w in WARNINGS], ""]
    L += ["그림: `fig_paths.png`(정답·PDR만·PF·PF + 비전 경로), `fig_error.png`(시간별 오차), `fig_map.png`. 정답 확인: 위 폴더의 `truth_check.md`·`truth_check.png`."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def config_for(name, rec, same_walk):
    p = RECORDINGS / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    base = HERE / "out/rec" / name
    rel = lambda q: str(q.relative_to(ROOT)).replace("\\", "/") if q.is_relative_to(ROOT) else str(q)
    cfg = dict(name=name, note="", rec=rel(rec), inputs=rel(base / "inputs"), out=rel(base / "out"), exclude=[], end_start=1e9,
               waypoints=[])
    if same_walk:
        o = json.loads((RECORDINGS / f"{same_walk}.json").read_text(encoding="utf-8"))
        cfg.update(waypoints=o["waypoints"], exclude=o["exclude"], end_start=o["end_start"], note=f"{same_walk}와 같은 걸음의 재생 기록")
    RECORDINGS.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cfg


def pipeline(a):
    src = Path(a.src)
    if src.exists():
        name = a.name or nfc(src.stem)
    elif (RECORDINGS / f"{a.src}.json").exists():
        name = a.src
    else:
        raise Stop(f"없는 파일·폴더, 설정도 없음: {a.src}")
    print(f"[1] 녹화 {name}")
    if src.exists():
        rec = unpack(src, name, a.force)
    else:
        import paths
        rec = paths.rec_dir(json.loads((RECORDINGS / f"{name}.json").read_text(encoding="utf-8")), a.data_root)
    cfg = config_for(name, rec, a.same_walk)
    env = {"I1_RECORDING": name}
    inputs, out = ROOT / cfg["inputs"], ROOT / cfg["out"]
    py = sys.executable
    print("[2] 파일 검사")
    info = check(rec)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[3] 관측")
    obs_src = observations(rec, inputs, info, env, a.force)
    print("  출처:", obs_src)
    print("[4] PDR")
    now = steps_simple_inputs(rec)
    if a.force or not (inputs / "steps_simple.csv").exists():
        print(" ", run([sys.executable, PDR_SIMPLE, "--recording", rec, "--out", inputs / "steps_simple.csv"]).strip())
        write_record(inputs, "steps_simple", now)
    else:
        warn_cache(inputs, "steps_simple", now, "임시 PDR steps_simple.csv")
    now = steps_indoornav_inputs(rec)
    if a.force or not (out / "steps_indoornav.csv").exists():
        print(" ", run([py, HERE / "pdr_steps.py"], env=env).strip())
        write_record(inputs, "steps_indoornav", now)
    else:
        warn_cache(inputs, "steps_indoornav", now, "indoornav PDR steps_indoornav.csv")
    print("[5] 정답")
    if not cfg["waypoints"]:
        run_out = run([py, HERE / "truth_helper.py", "candidates"], env=env)
        print(run_out)
        raise Stop(f"기준점이 비어 있어 여기서 멈춤. 후보·그림: {out}\n"
                   f"할 일: waypoints_draft.json에서 쓸 후보를 골라 st(역 좌표, plan_B01·B02.png에서 읽음) 또는 node, floor, ref를 채우고\n"
                   f"       {RECORDINGS / (name + '.json')}의 waypoints에 넣은 뒤 같은 명령을 다시 실행.\n"
                   f"       기둥 지나는 순간은 truth_helper.py sheet --from A --to B (I1_RECORDING={name}). 판은 기준으로 쓰지 않음.")
    if not cfg["exclude"] and cfg["end_start"] > 1e8:
        cfg = fill_scoring(name, cfg)
    print(run([py, HERE / "truth_build.py"], env=env))
    if a.old_replay:
        print("[6] 재생 (예전 방식)")
        print(run([py, HERE / "run_i1.py", "--adopted", "--name", RUN_NAME, "--seeds", str(a.seeds)], env=env)[-1500:])
        run([py, HERE / "fig_i1.py", "--name", RUN_NAME, "--set", "기본", "--obs", "tf"], env=env)
        write_report(name, obs_src, info, out / RUN_NAME)
        print(f"끝: {out / RUN_NAME / 'report.md'}")
        return
    replays(out, env, a)


def fresh(f, deps, keys=()):
    """결과 f를 그대로 써도 되나: 있고, 입력(deps)보다 나중에 만들어졌고(파일 시각이 더 새 입력은 같은 폴더 summary.json의
    입력 해시와 내용이 같으면 통과), npz면 필요한 키(keys)가 다 있음"""
    from provenance import rel, sha
    if not f.exists():
        return False
    t = f.stat().st_mtime
    newer = [Path(d) for d in deps if Path(d).exists() and Path(d).stat().st_mtime > t]
    if newer:
        s = f.parent / "summary.json"
        rec = json.loads(s.read_text(encoding="utf-8")).get("info", {}).get("provenance", {}).get("files", {}) if s.exists() else {}
        if not all(rec.get(rel(d)) == sha(d) for d in newer):
            return False
    if keys:
        import numpy as np
        with np.load(f) as z:
            return set(keys) <= set(z.files)
    return True


def replays(out, env, a):
    """6 기본 재생 → 7 실시간 재생 → 8 그림. 결과가 있고 입력보다 새것이며 필요한 seed·단계가 다 있으면 건너뜀(--force-replay면 다시)"""
    py = sys.executable
    seeds, procs = ["--seeds", str(a.seeds)], ["--procs", str(a.procs)]
    ins = json.loads((RECORDINGS / f"{env['I1_RECORDING']}.json").read_text(encoding="utf-8"))
    inputs = ROOT / ins["inputs"]
    deps = [inputs / "obs_timefilter.jsonl", inputs / "obs_nofilter.jsonl", inputs / "truth.csv", out / "steps_indoornav.csv",
            out / "heading_indoornav.npz"]          # 기준점은 truth.csv에 반영됨(설정 json 자체는 안 봄)
    use = lambda f, d, k=(): not a.force_replay and fresh(f, d, k)
    ref_npz, rt_npz = out / REF / "runs.npz", out / RT / "runs.npz"

    def need_run(f, d, k, what):
        """없거나 --force-replay면 돌림, 있고 맞으면 건너뜀. 있는데 입력보다 오래됐거나 seed·단계가 모자라면 덮어쓰지 않고 멈춤
        (실시간 재생 결과에는 이 흐름보다 많은 단계가 들어 있을 수 있어서)"""
        if a.force_replay or not f.exists():
            return True
        if fresh(f, d, k):
            print("  있음, 건너뜀")
            return False
        raise Stop(f"있는 {what} 결과({f.parent.name})가 입력보다 오래됐거나 seed·단계가 모자람. "
                   f"다시 만들려면 --force-replay (그 폴더를 덮어씀), 남기려면 폴더를 옮긴 뒤 다시 실행")

    print(f"[6] 기본 재생 {REF}")
    if need_run(ref_npz, deps, [f"known_tf_기본_{s}" for s in range(a.seeds)] + ["known_none_0", "known_pdr_only_0"], "기본 재생"):
        print(run([py, HERE / "exp_cause.py", "--name", REF, *seeds, *procs], env=env)[-800:])
    print(f"[7] 실시간 재생 {RT} ({RT_STAGES})")
    if need_run(rt_npz, deps + [ref_npz], [f"{st}_{s}" for st in RT_STAGES.split(",") for s in range(a.seeds)], "실시간 재생"):
        o = run([py, HERE / "run_rt.py", "--name", RT, "--stages", RT_STAGES, "--check-ref", REF, *seeds, *procs], env=env)
        res = [json.loads(line[7:]) for line in o.splitlines() if line.startswith("RESULT ")]
        chk = res[-1]["check"].get("check_ref") if res else None
        print("  off vs", REF, chk)
        if not chk or chk["same"] != chk["n"]:
            raise Stop(f"실시간 재생 off 단계가 기본 재생 {REF}와 비트까지 같지 않음(또는 확인 못 함): {chk}. 입력·코드를 확인할 것")
    print("[8] 그림")
    if not use(out / REF / "fig_plan.png", [ref_npz]):
        run([py, HERE / "fig_plan.py", "--name", REF], env=env)
    data = out / "rt_fig" / "rec_paths.npz"
    if not use(data, [rt_npz, ref_npz]):
        run([py, HERE / "fig_plan_rt.py", "data"], env=env)
    if not all(use(out / "rt_fig" / f, [data, ref_npz]) for f in ("fig_plan_rec.png", "fig_plan_compare.png")):
        run([py, HERE / "fig_plan_rt.py", "draw"], env=env)
    print(f"끝: {out / REF / 'fig_plan.png'}, {out / 'rt_fig'}")


def fill_scoring(name, cfg):
    """기준점에서 층 이동 채점 제외·끝 오차 시작을 채워 설정에 저장 (truth_build.derived와 같은 규칙, 1차 녹화 78.3~103·114.5가 이 규칙)"""
    wps = cfg["waypoints"]
    cfg["exclude"] = [[wps[i]["t"], wps[i + 1]["t"] + 2.0] for i, w in enumerate(wps[:-1]) if w.get("esc")]
    cfg["end_start"] = wps[-2]["t"]
    (RECORDINGS / f"{name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cfg


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="zip, 폴더, 이미 있는 설정 이름, 또는 all(2차 보행 8개)")
    ap.add_argument("--name", help="녹화 이름 (기본: 파일 이름)")
    ap.add_argument("--same-walk", help="같은 걸음의 설정 이름에서 기준점을 가져옴")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="관측·PDR을 다시 만듦")
    ap.add_argument("--force-replay", action="store_true", help="기본·실시간 재생과 그림을 다시 만듦")
    ap.add_argument("--old-replay", action="store_true", help="6단계를 예전 run_i1 --adopted 재생으로")
    ap.add_argument("--data-root", help="앱 녹화 폴더들의 상위 폴더 (환경 변수 I1_DATA_ROOT와 같음)")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    if a.data_root:
        a.data_root = str(Path(a.data_root).resolve())
        os.environ["I1_DATA_ROOT"] = a.data_root          # 하위 스크립트(paths.REC)도 같은 폴더에서 찾게
    a.data_root = a.data_root or os.environ.get("I1_DATA_ROOT")
    names = list(S2) if a.src == "all" else [a.src]
    failed = []
    for n in names:
        if len(names) > 1:
            print(f"\n===== {n} =====", flush=True)
            a.src, a.name = n, None
            WARNINGS.clear()
        try:
            pipeline(a)
        except (Stop, SystemExit) as e:          # 녹화 폴더 못 찾음·하위 명령 실패도 그 기록만 멈추고 다음 기록으로
            if isinstance(e, SystemExit) and len(names) == 1:
                raise
            print(f"\n멈춤: {e}")
            failed.append(n)
    if failed:
        print(f"\n멈춘 기록: {failed}")
        sys.exit(2)


if __name__ == "__main__":
    main()
