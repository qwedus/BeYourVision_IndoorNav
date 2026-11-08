"""OCR 시험 앱 세션 폴더 → cv_module 재생 입력(replay JSON). PC 재생기·위치추정 재생 시험(cv_test/i1)에 그대로 넣는 용.

    python tools/session_to_replay.py <세션 폴더> --out <replay.json> [--which all|kept] [--camera-height 1.4] [--cv-module <경로>]

  --which all  : 앱이 저장한 검출 전부(점수 store_conf_floor 이상). PC config 문턱값으로 다시 거른다 → YOLO 쪽 값은 PC에서 바꿔 볼 수 있음
  --which kept : 앱이 최종으로 남긴 검출만(시간 필터·OCR·종류 바로잡기 반영, 종류 = final_label). 앱 판정 그대로 위치추정에 넣을 때
그다음 (localization_design/cv_module에서):
    python -m cvmod.run --detector replay --replay <replay.json> --out <obs.json>
  → obs.json의 frames가 cv_test/i1 run_i1.py의 cv_raw_5fps.json과 같은 형식 (I1은 REC를 이 세션 폴더로 바꾸고 --skip-cv 대신 이 파일을 씀)

frame_info는 cv_module cvmod.sources.recording(frames.jsonl → FrameInfo)으로 만든다 → 녹화 모드와 같은 K·자세 처리.
frame_id = <meta.json session_name>/f<i:05d> (폴더 이름을 나중에 바꿔도 검출 id와 맞게 처음 이름을 씀). 이름·id는 NFC로 맞춤.
표준 라이브러리 + cv_module만 (영상은 안 풂).
"""
import argparse
import datetime
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_CV = Path(r"cv_module")


def nfc(s):
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # 앱이 저장 중 끊겼으면 마지막 줄이 잘릴 수 있음
                print(f"경고: {Path(path).name} {n}번째 줄을 건너뜀 (잘림?)", file=sys.stderr)
    return rows


def build(folder, which="all", cv_module=DEFAULT_CV, camera_height=None):
    sys.path.insert(0, str(Path(cv_module)))
    from cvmod.schema import SCHEMA_VERSION
    from cvmod.sources import recording

    folder = Path(folder)
    meta = recording.read_meta(folder)
    name = nfc(meta.get("session_name") or folder.name)   # 앱 0.2.1까지는 NFD로 남은 기록이 있음
    tags = {t["i"]: t for t in recording.read_tags(folder)}
    dets = read_jsonl(folder / "detections.jsonl")
    decs = {}
    for d in read_jsonl(folder / "decisions.jsonl"):
        decs[nfc(d["detection_id"])] = d

    frames, k_did, n = [], Counter(), Counter()
    for row in dets:
        tag = tags.get(row["i"])
        if tag is None:
            n["no_tag"] += 1
            continue
        fi, did = recording.frame_info_for_tag(tag, None, name, camera_height=camera_height)
        k_did[did] += 1
        out = []
        for d in row["detections"]:
            d = dict(d)
            d["detection_id"] = nfc(d["detection_id"])
            if which == "kept":
                dc = decs.get(d["detection_id"])
                if not dc or dc.get("final") != "kept":
                    n["not_kept"] += 1
                    continue
                d["label"] = dc.get("final_label") or d["label"]
            out.append({k: d.get(k) for k in ("detection_id", "source", "label", "score", "box_xyxy",
                                               "parent_detection_id", "text")})
        n["detections"] += len(out)
        frames.append({"schema_version": SCHEMA_VERSION, "frame_info": fi.to_dict() if hasattr(fi, "to_dict") else _asdict(fi),
                       "status": "ok" if out else "empty", "detections": out, "observations": [], "dropped": []})
    run = {"run_id": f"ocrapp-{which}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}", "detector": "ocr_test_app",
           "session": name, "folder": str(folder), "which": which,
           "model": meta.get("model"), "ocr": {k: (meta.get("ocr") or {}).get(k) for k in ("engine", "mode", "choice")},
           "fps": meta.get("fps"), "config": meta.get("config"), "k_handling": dict(k_did), "counts": dict(n),
           "input": {"kind": "recording", "name": folder.name, "rate": None, "pick": "phone_detected",
                     "frames_in_recording": len(tags), "frames_selected": len(frames)}}
    return {"schema_version": SCHEMA_VERSION, "run": run, "frames": frames}


def _asdict(x):
    from dataclasses import asdict
    return asdict(x)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("folder")
    p.add_argument("--out", required=True)
    p.add_argument("--which", default="all", choices=("all", "kept"))
    p.add_argument("--cv-module", default=str(DEFAULT_CV))
    p.add_argument("--camera-height", type=float, default=1.4, help="카메라 높이 m (거리 계산에만, I1과 같은 1.4)")
    a = p.parse_args()
    d = build(a.folder, a.which, a.cv_module, a.camera_height)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, ensure_ascii=False, allow_nan=False, indent=1), encoding="utf-8")
    print("저장:", a.out, d["run"]["counts"], "프레임", len(d["frames"]))


if __name__ == "__main__":
    main()
