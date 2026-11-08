"""시간 필터: cv_module 결과(FrameResult 목록) → 같은 형식, '최근 5프레임 중 3번 이상' 잡힌 판의 관측만 남김.

cv_module 코드는 고치지 않고 결과 파일만 읽어 바꾼다 (OCR 필터 시험의 '앞 프레임만 보는 시간 필터'와 같은 규칙).
    추적: 같은 검출 종류(label) + 박스 IoU ≥ 0.3 이면 앞 프레임의 같은 판. 마지막으로 잡힌 지 WINDOW프레임 넘으면 끊김
    판정: 지금 프레임을 포함한 최근 WINDOW(5)프레임 중 MIN_HITS(3)번 이상 잡힌 판만 관측으로 넘김 (뒤 프레임은 안 봄 → 지연 없음)
    빠진 관측은 dropped에 reason="time_filter"로 옮기고 status를 다시 매긴다. run에 "time_filter" 설정을 남긴다.
    track_id: 판마다 증가하는 번호를 관측에 붙인다(배열 위치가 아님). TrackLimiter: 같은 track은 N초에 1번만 PF에 넣음 (연결안 3절 5)
    보조 잇기 옵션(기본 꺼짐, 끄면 원래 규칙과 같은 코드):
        center_dist_max: IoU가 문턱 미만일 때 박스 중심 거리 / 두 박스 중 큰 대각선 ≤ 이 값이면 같은 판
                         (앱 Pipeline.swift match와 같은 점수: IoU 통과 1 + IoU, 중심 거리 통과 1 - dn/cd + 1e-6, 큰 점수)
        pitch_comp:      앞 박스를 프레임 자세 pitch 변화만큼 세로로 옮긴 뒤 IoU (dy = fy · Δpitch[rad], 기울기 1 고정).
                         실패 프레임·가정 자세(source가 device 아님)·pitch나 fy가 없는 프레임은 안 옮김. fy는 이미지 크기에 맞춤
입력은 cv_module 출력 JSON({"run", "frames"}) 또는 FrameResult 한 줄씩인 JSONL.
"""
import argparse
import json
import math
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


def center_dist(a, b):
    """박스 중심 거리 / 두 박스 중 큰 대각선 (앱 Pipeline.centerDist와 같음)"""
    dx, dy = (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
    da, db = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return (dx * dx + dy * dy) ** 0.5 / max(da, db, 1e-9)


def _pitch_fy(f):
    """프레임 → (pitch[rad], 이미지 크기에 맞춘 fy) 또는 None.
    None: 실패 프레임, 자세가 기기 값이 아님(source가 device 아님 = 빈 값을 0으로 채운 가정값), pitch·fy가 없거나 수가 아님.
    K 기준 크기(intrinsics.height)가 이미지 높이와 다르면 cv_module처럼 fy를 비례로 맞춤."""
    if f.get("status") == "failed":
        return None
    info = f.get("frame_info") or {}
    pose, k = info.get("pose") or {}, info.get("intrinsics") or {}
    if pose.get("source", "device") != "device":
        return None
    try:
        p, fy = float(pose["pitch_deg"]), float(k["fy"])
        kh, h = k.get("height"), info.get("image_height")
        if kh and h and float(kh) != float(h):
            fy *= float(h) / float(kh)
    except (KeyError, TypeError, ValueError):
        return None
    return (math.radians(p), fy) if math.isfinite(p) and math.isfinite(fy) and fy > 0 else None


def apply(frames, window=WINDOW, min_hits=MIN_HITS, iou_min=IOU, center_dist_max=None, pitch_comp=False):
    """frames (dict, 시각 순)를 고쳐서 반환. 관측마다 track_id를 붙인다(그 밖의 형식은 그대로).
    center_dist_max·pitch_comp: 보조 잇기 옵션(모듈 설명). 둘 다 끄면 원래 규칙(IoU 같으면 뒤 track)."""
    extra = center_dist_max is not None or pitch_comp
    tracks = []      # {id, label, box, hits: [프레임 번호], pf: 마지막 프레임 (pitch, fy)}
    out = []
    next_id = 0
    for k, f in enumerate(frames):
        f = json.loads(json.dumps(f))
        boxes = {d["detection_id"]: d["box_xyxy"] for d in f.get("detections", [])}
        pf = _pitch_fy(f) if pitch_comp else None
        keep, drop = [], []
        used = set()
        for ob in f.get("observations", []):
            box = boxes[ob["detection_id"]]
            best, bi = iou_min, None
            if extra:
                best = 0.0
            for ti, tr in enumerate(tracks):
                if ti in used or tr["label"] != ob["label"] or k - tr["hits"][-1] > window:
                    continue
                if not extra:
                    v = iou(box, tr["box"])
                    if v >= best:
                        best, bi = v, ti
                    continue
                ref = tr["box"]
                if pf is not None and tr.get("pf") is not None:
                    dy = pf[1] * (pf[0] - tr["pf"][0])
                    ref = [ref[0], ref[1] + dy, ref[2], ref[3] + dy]
                v = iou(box, ref)
                s = 1 + v if v >= iou_min else 0.0
                if s == 0.0 and center_dist_max is not None:
                    dn = center_dist(tr["box"], box)
                    if dn <= center_dist_max:
                        s = 1 - dn / max(center_dist_max, 1e-9) + 1e-6
                if s > 0.0 and s >= best:
                    best, bi = s, ti
            if bi is None:
                tracks.append({"id": next_id, "label": ob["label"], "box": box, "hits": []})
                next_id += 1
                bi = len(tracks) - 1
            used.add(bi)
            tr = tracks[bi]
            tr["box"] = box
            if pitch_comp:
                tr["pf"] = pf
            tr["hits"].append(k)
            ob["track_id"] = tr["id"]
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


def track_ids(frames, window=WINDOW, iou_min=IOU, center_dist_max=None, pitch_comp=False):
    """필터 없는 프레임들 → {detection_id: track_id}. 시간 필터와 같은 추적이라 필터 뒤 관측에도 같은 번호를 쓸 수 있다."""
    return {ob["detection_id"]: ob["track_id"]
            for f in apply(frames, window, 1, iou_min, center_dist_max, pitch_comp) for ob in f["observations"]}


class TrackLimiter:
    """같은 track은 최근 n초에 1번만 통과. 호출 순서 = 시각 순. track_id가 없는 관측은 그대로 통과."""

    def __init__(self, n):
        self.n, self.last = n, {}

    def __call__(self, t, obs):
        keep = []
        for o in obs:
            tid = o.get("track_id")
            if tid is None or t - self.last.get(tid, -1e18) >= self.n - 1e-6:
                if tid is not None:
                    self.last[tid] = t
                keep.append(o)
        return keep


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
