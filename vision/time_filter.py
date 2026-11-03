"""시간 필터: cv_module 결과(FrameResult 목록) → 같은 형식, '최근 5프레임 중 3번 이상' 잡힌 판의 관측만 남김.

cv_module 코드는 고치지 않고 결과 파일만 읽어 바꾼다 (윤재 OCR 필터 보고의 '앞 프레임만 보는 시간 필터'와 같은 규칙).
    추적: 같은 검출 종류(label) + 박스 IoU ≥ 0.3 이면 앞 프레임의 같은 판. 마지막으로 잡힌 지 WINDOW프레임 넘으면 끊김
    판정: 지금 프레임을 포함한 최근 WINDOW(5)프레임 중 MIN_HITS(3)번 이상 잡힌 판만 관측으로 넘김 (뒤 프레임은 안 봄 → 지연 없음)
    빠진 관측은 dropped에 reason="time_filter"로 옮기고 status를 다시 매긴다. run에 "time_filter" 설정을 남긴다.
입력은 cv_module 출력 JSON({"run", "frames"}) 또는 FrameResult 한 줄씩인 JSONL.
"""
import argparse
import json
from pathlib import Path

WINDOW, MIN_HITS, IOU = 5, 3, 0.3


def load_frames(path):
    """→ (run dict 또는 None, frame dict 목록)."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return None, [json.loads(l) for l in text.splitlines() if l.strip()]
    d = json.loads(text)
    return d.get("run"), d["frames"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def apply(frames, window=WINDOW, min_hits=MIN_HITS, iou_min=IOU):
    """frames (dict, 시각 순)를 고쳐서 반환. 통과한 관측에 track id를 붙이지는 않는다 (형식 그대로)."""
    tracks = []      # {label, box, hits: [프레임 번호]}
    out = []
    for k, f in enumerate(frames):
        f = json.loads(json.dumps(f))
        boxes = {d["detection_id"]: d["box_xyxy"] for d in f.get("detections", [])}
        keep, drop = [], []
        used = set()
        for ob in f.get("observations", []):
            box = boxes[ob["detection_id"]]
            best, bi = iou_min, None
            for ti, tr in enumerate(tracks):
                if ti in used or tr["label"] != ob["label"] or k - tr["hits"][-1] > window:
                    continue
                v = iou(box, tr["box"])
                if v >= best:
                    best, bi = v, ti
            if bi is None:
                tracks.append({"label": ob["label"], "box": box, "hits": []})
                bi = len(tracks) - 1
            used.add(bi)
            tr = tracks[bi]
            tr["box"] = box
            tr["hits"].append(k)
            n = sum(1 for h in tr["hits"] if h > k - window)
            (keep if n >= min_hits else drop).append(ob)
        tracks = [t for t in tracks if k - t["hits"][-1] <= window]
        f["observations"] = keep
        f.setdefault("dropped", []).extend(
            {"detection_id": o["detection_id"], "label": o["label"], "score": o["score"], "reason": "time_filter"}
            for o in drop)
        if f["status"] != "failed":
            f["status"] = "ok" if keep else "empty"
        out.append(f)
    return out


def save(run, frames, path_json, path_jsonl=None):
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    schema = frames[0].get("schema_version") if frames else None
    Path(path_json).write_text(json.dumps({"schema_version": schema, "run": run, "frames": frames},
                                          ensure_ascii=False, allow_nan=False, indent=1), encoding="utf-8")
    if path_jsonl:
        with open(path_jsonl, "w", encoding="utf-8") as fp:
            for f in frames:
                fp.write(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="시간 필터 (최근 5프레임 중 3번)")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True, help="결과 JSON (cv_module 형식)")
    ap.add_argument("--jsonl", help="FrameResult 한 줄씩")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--min-hits", type=int, default=MIN_HITS)
    a = ap.parse_args()
    run, frames = load_frames(a.inp)
    res = apply(frames, a.window, a.min_hits)
    run = dict(run or {}, time_filter={"window_frames": a.window, "min_hits": a.min_hits, "iou": IOU,
                                       "causal": True})
    save(run, res, a.out, a.jsonl)
    n0 = sum(len(f["observations"]) for f in frames)
    n1 = sum(len(f["observations"]) for f in res)
    print(f"관측 {n0} → {n1} ({a.out})")
