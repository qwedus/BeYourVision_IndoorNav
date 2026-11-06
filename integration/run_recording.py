"""앱 녹화 한 건을 한 번에: zip·폴더 → 파일 검사 → 검출 관측 → PDR → 정답 → PDR만·PF·PF + 비전 → 경로 그림·오차 표.

    .venv/Scripts/python integration/run_recording.py <zip 또는 폴더> [--name 이름]
    .venv/Scripts/python integration/run_recording.py <이미 설정이 있는 이름>          (다시 돌리기, 기준점 채운 뒤)

단계 (끝난 단계는 파일이 있으면 건너뜀, --force면 다시. 건너뛸 때 <입력 폴더>/input_record.json의 기록과 지금 입력을 비교해 다르면 경고, 보고서에도 남김)
    1 풀기   zip → integration/out/rec/<이름>/rec/, 폴더는 그 자리를 그대로 씀. 설정 integration/recordings/<이름>.json을 만듦
    2 검사   motion·baro·meta·frames 있는지, 길이·빠진 값 (없으면 이유를 적고 멈춤)
    3 관측   detections.jsonl 있으면 앱 검출 변환(ocr_test_app session_to_replay → cvmod replay),
             없고 영상이 있으면 PC 검출(cvmod yolo26n, C2 추가학습 후 모델, 5 fps), 둘 다 없으면 멈춤. 그다음 시간 필터
    4 PDR    임시 PDR(정답 보간용 걸음 시각) + indoornav PDR
    5 정답   설정의 기준점 → truth.csv. 기준점이 없으면 후보·그림을 만들고 무엇을 채울지 알려 준 뒤 멈춤
    6 재생   run_i1 --adopted (출발점 앎·PDR 틀어짐 × 비전 끔·필터 없음·시간 필터 × 기본·중간, seed 10개), 그림, 표

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
OCR_APP = ROOT.parent / "ocr_test_app"
MODEL = LOCAL / "cv_test/out/c2/runs/dc26n_dense_s0/weights/last.pt"
VISION_PY = os.environ.get("I1_VISION_PY", str(Path.home() / "AppData/Local/Programs/Python/Python312/python.exe"))
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
    common = dict(cv_config=file_sig(LOCAL / "cv_module/config/default.json"), cvmod=tree_sig(LOCAL / "cv_module/cvmod"), camera_height=CAMERA_H)
    if info["detections"]:
        return dict(common, how="앱 검출", detections=file_sig(rec / "detections.jsonl"), frames=file_sig(rec / "frames.jsonl"),
                    meta=file_sig(rec / "meta.json"), session_to_replay=file_sig(OCR_APP / "tools/session_to_replay.py"))
    if info["video"]:
        return dict(common, how="PC 검출", video=binary_sig(rec / info["video"]), frames=file_sig(rec / "frames.jsonl"), meta=file_sig(rec / "meta.json"),
                    model=file_sig(MODEL), rate=VIDEO_RATE)
    return dict(common, how="없음")


def steps_simple_inputs(rec):
    return dict(motion=file_sig(rec / "motion.csv"), pdr_simple=file_sig(LOCAL / "cv_test/i1/pdr_simple.py"))


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
        run([VISION_PY, OCR_APP / "tools/session_to_replay.py", rec, "--out", replay, "--which", "all",
             "--cv-module", LOCAL / "cv_module", "--camera-height", CAMERA_H])
        run([VISION_PY, "-m", "cvmod.run", "--detector", "replay", "--replay", replay, "--out", raw], cwd=LOCAL / "cv_module")
        src = "앱 검출(detections.jsonl) → PC 문턱값"
    elif info["video"]:
        if not MODEL.exists():
            raise Stop(f"PC 검출에 쓸 모델이 없음: {MODEL}")
        run([VISION_PY, "-m", "cvmod.run", "--detector", "yolo26n", "--model", MODEL, "--recording", rec, "--rate", VIDEO_RATE,
             "--pick", "gyro", "--frames-out", inputs / "frames_5fps_gyro", "--camera-height", CAMERA_H, "--out", raw],
            cwd=LOCAL / "cv_module")
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
        c = json.loads((RECORDINGS / f"{name}.json").read_text(encoding="utf-8"))
        rec = ROOT / c["rec"]
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
        print(" ", run([VISION_PY, LOCAL / "cv_test/i1/pdr_simple.py", "--recording", rec, "--out", inputs / "steps_simple.csv"]).strip())
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
    print("[6] 재생")
    print(run([py, HERE / "run_i1.py", "--adopted", "--name", RUN_NAME, "--seeds", str(a.seeds)], env=env)[-1500:])
    run([py, HERE / "fig_i1.py", "--name", RUN_NAME, "--set", "기본", "--obs", "tf"], env=env)
    write_report(name, obs_src, info, out / RUN_NAME)
    print(f"끝: {out / RUN_NAME / 'report.md'}")


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
    ap.add_argument("src", help="zip, 폴더, 또는 이미 있는 설정 이름")
    ap.add_argument("--name", help="녹화 이름 (기본: 파일 이름)")
    ap.add_argument("--same-walk", help="같은 걸음의 설정 이름에서 기준점을 가져옴")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="관측·PDR을 다시 만듦")
    a = ap.parse_args()
    try:
        pipeline(a)
    except Stop as e:
        print(f"\n멈춤: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
