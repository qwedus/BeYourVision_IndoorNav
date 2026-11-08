"""아이폰 녹화본 폴더 → 프레임 이미지 + FrameInfo (형식: docs/명세.md 9절).

    녹화_YYYYMMDD_HHMM/
      video.mov      H.264, 거치 방향 버퍼 그대로(회전 정보 없음이 기본). 앱 1.2.0 세로 1080×1920, 1.3.0부터 가로 1920×1080
      frames.jsonl   프레임마다 촬영 시각 t, K, roll·pitch, 자이로 (영상 프레임 순서 = i)
      motion.csv     모션 센서 원자료 약 100 Hz (t는 frames.jsonl과 같은 시계)
      baro.csv       기압 kPa·상대 높이 m (앱 1.3.1~, 없을 수 있음. t는 같은 시계) · baro_abs.csv 절대 고도(되는 기기만)
      meta.json      기기·앱·카메라 설정

- 초당 N장(기본 5): 1/N초 창마다 한 장. 고르는 기준(pick): gyro(자이로 회전 속도 최소, 기본) · first(창의 첫 장)
  · phone(폰이 0.2초 창에서 고른 picked) · all(모든 프레임).
- K는 frames.jsonl 값(source="device"). 주점이 버퍼 중앙이 아니라 90° 돌린 기준으로 보이면 버퍼 쪽으로 돌리고,
  다른 해상도 기준으로 보이면 비율을 맞춘다(k_frame="auto"). 무엇을 했는지 run 정보에 남긴다.
- 영상 회전 정보가 있으면 적용해 "up" 이미지로 만든다(OpenCV 자동 회전, 안 되면 직접).
- 고르기·K·FrameInfo 부분은 표준 라이브러리만, 영상 풀기(extract)만 OpenCV(cv2)가 필요하다.
"""
import csv
import json
import math
import statistics
from pathlib import Path

from ..schema import FrameInfo, Intrinsics, Pose

PICKS = ("gyro", "first", "phone", "all")


def read_meta(folder):
    p = Path(folder) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_tags(folder):
    """frames.jsonl → 프레임 꼬리표 목록 (i 순서)."""
    tags = []
    with open(Path(folder) / "frames.jsonl", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tags.append(json.loads(line))
            except json.JSONDecodeError as e:   # 앱이 저장 중 끊겼으면 마지막 줄이 잘릴 수 있음
                raise ValueError(f"frames.jsonl {n}번째 줄을 읽을 수 없음: {e}") from None
    tags.sort(key=lambda t: t["i"])
    return tags


BARO_COLS = ("t", "pressure_kpa", "rel_alt_m")          # 필수 열. t_recv(받은 시각)는 있으면 읽음
BARO_ABS_COLS = ("t", "alt_m", "accuracy_m", "precision_m")


def _read_csv(path, need):
    """숫자 CSV → dict 목록 (t 순). 파일이 없거나 표본이 없으면 None. 중간에 끊긴 마지막 줄은 버림."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return None
        miss = [c for c in need if c not in r.fieldnames]
        if miss:
            raise ValueError(f"{path.name}에 열 {miss}가 없음 (있는 열: {r.fieldnames})")
        lines = list(r)
    rows = []
    for n, row in enumerate(lines, 2):
        try:
            x = {k: float(v) for k, v in row.items() if k and v not in (None, "")}
            if any(c not in x for c in need):
                raise ValueError("빈 칸")
        except ValueError:
            if n == len(lines) + 1:   # 앱이 저장 중 끊겨 잘린 마지막 줄만 허용
                break
            raise ValueError(f"{path.name} {n}번째 줄을 읽을 수 없음") from None
        rows.append(x)
    rows.sort(key=lambda x: x["t"])
    return rows or None


def read_baro(folder):
    """baro.csv → [{"t", "pressure_kpa", "rel_alt_m", ("t_recv")}] (t 순, t는 frames.jsonl·motion.csv와 같은 시계).
    파일이 없거나 비면 None (앱 1.3.0 이전 녹화본, 기압계 없음, 권한 거부 — 이유는 meta.json baro)."""
    return _read_csv(Path(folder) / "baro.csv", BARO_COLS)


ABS_CLOCK_GAP_S = 1e6   # t와 t_recv가 이보다 벌어지면 다른 시계로 봄 (호스트 시계는 부팅 후 초, 벽시계는 2001년부터 초)


def read_baro_abs(folder):
    """baro_abs.csv → [{"t", "alt_m", "accuracy_m", "precision_m", ("t_recv", "t_raw")}] 또는 None (실내에선 GPS가 없어 비기 쉬움).

    앱 1.3.1은 절대 고도 t를 호스트 시계가 아니라 벽시계(2001-01-01부터 초, CMAbsoluteAltitudeData.timestamp)로 썼다.
    t_recv(호스트 시계)와 크게 벌어지면 t += median(t_recv − t)로 호스트 시계에 맞추고 원래 값은 t_raw에 둔다
    (차이는 전달 지연만큼 늦게 잡히지만 실측 흔들림 0.03 s, 지연 약 0.1 s 이하)."""
    rows = _read_csv(Path(folder) / "baro_abs.csv", BARO_ABS_COLS)
    lags = [r["t_recv"] - r["t"] for r in rows or () if "t_recv" in r]
    if lags and abs(statistics.median(lags)) > ABS_CLOCK_GAP_S:
        off = statistics.median(lags)
        for r in rows:
            r["t_raw"] = r["t"]
            r["t"] = r["t"] + off
    return rows


def baro_summary(rows, meta=None):
    """기압 표본 → run 정보에 넣을 짧은 요약. 표본이 없으면 {"recorded": False, "reason": ...}."""
    if not rows:
        b = (meta or {}).get("baro") or {}
        return {"recorded": False, "reason": b.get("error") or ("기압 기록 없음" if b else "baro.csv 없음 (앱 1.3.1 이전?)")}
    alt = [r["rel_alt_m"] for r in rows]
    out = {"recorded": True, "samples": len(rows), "t_first": rows[0]["t"], "t_last": rows[-1]["t"],
           "rate_hz": (len(rows) - 1) / (rows[-1]["t"] - rows[0]["t"]) if rows[-1]["t"] > rows[0]["t"] else None,
           "pressure_kpa_range": [min(r["pressure_kpa"] for r in rows), max(r["pressure_kpa"] for r in rows)],
           "rel_alt_m_range": [min(alt), max(alt)]}
    lags = [r["t_recv"] - r["t"] for r in rows if "t_recv" in r]
    if lags:   # 같은 시계면 수 초 안(아이폰 17 실측 약 2.7 s: 값이 측정보다 늦게 들어옴). 수천 초 이상이면 시계가 다름
        out["recv_lag_median_s"] = statistics.median(lags)
    return out


def select(tags, rate=5.0, pick="gyro"):
    """1/rate초 창마다 한 장 고르기. 반환: 고른 꼬리표 목록 (시각 순)."""
    if pick not in PICKS:
        raise ValueError(f"pick은 {PICKS} 중 하나")
    if pick == "all":
        return list(tags)
    if pick == "phone":
        return [t for t in tags if t.get("picked")]
    if not tags:
        return []
    if not rate or rate <= 0:
        raise ValueError("rate는 양수")
    t0, win = tags[0]["t"], 1.0 / rate
    groups = {}
    for t in tags:
        groups.setdefault(math.floor((t["t"] - t0) / win + 1e-9), []).append(t)
    out = []
    for k in sorted(groups):
        g = groups[k]
        if pick == "first":
            out.append(g[0])
        else:   # 자이로가 없는 프레임은 뒤로 (같으면 앞 프레임)
            out.append(min(g, key=lambda t: (t.get("gyro_norm") is None, t.get("gyro_norm") or 0.0, t["i"])))
    return out


# 거치 방향(meta.json mount.orientation) → 기기 좌표로 본 (이미지 위, 이미지 오른쪽). 앱 Mount와 같은 값
MOUNTS = {
    "portrait": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),          # 앱 1.2.0까지 (mount 기록 없음)
    "landscape_right": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),  # 앱 1.3.0~: 화면을 볼 때 폰 윗부분이 왼쪽
}


def mount_of(meta):
    """meta.json → 거치 방향 이름. 기록이 없으면 세로(1.2.0 이전 녹화본)."""
    m = (meta or {}).get("mount") or {}
    name = m.get("orientation", "portrait") if isinstance(m, dict) else str(m)
    if name not in MOUNTS:
        raise ValueError(f"모르는 거치 방향 {name!r} (아는 것: {', '.join(MOUNTS)})")
    return name


def roll_pitch_from_gravity(gx, gy, gz, mount="portrait"):
    """중력(기기 좌표: x 오른쪽, y 위, z 화면 밖, 단위 무관) → 카메라 기준 (roll, pitch) [도].

    앱 MotionLog.rollPitch와 같은 식. 뒤 카메라, 명세 2절 부호:
      pitch = asin(g_z)               렌즈가 위를 보면 + (화면 안 회전과 무관)
      roll  = atan2(g·오른쪽, -g·위)   화면이 시계 방향으로 돌면 +
              세로: atan2(g_x, -g_y),  가로(landscape_right): atan2(-g_y, -g_x)
    """
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n == 0:
        raise ValueError("중력 벡터가 0")
    g = (gx / n, gy / n, gz / n)
    up, right = MOUNTS[mount]
    gu = sum(a * b for a, b in zip(g, up))
    gr = sum(a * b for a, b in zip(g, right))
    return math.degrees(math.atan2(gr, -gu)), math.degrees(math.asin(max(-1.0, min(1.0, g[2]))))


def k_for_tag(tag, mode="auto"):
    """꼬리표의 K → (fx, fy, cx, cy, 한 일) 버퍼(w×h, 세로든 가로든) 기준. K가 없으면 None.

    mode="asis": 그대로. mode="auto": 주점으로 기준을 판단
      - 버퍼 중앙 근처 → 그대로 ("asis"). 가로 녹화(1.3.0~)는 회전 0이라 늘 이것
      - 90° 돌린 모양(세로 버퍼에 회전 전 1920×1080 K) 중앙 근처 → 시계 방향 90° 회전 ("rotated_cw90"): fx'=fy, fy'=fx, cx'=w-cy, cy'=cx
      - 같은 모양이지만 다른 해상도 기준 → 비율로 맞춤 ("scaled x…")
    """
    K = tag.get("K")
    if not K:
        return None
    (fx, _, cx), (_, fy, cy) = K[0], K[1]
    w, h = tag["w"], tag["h"]
    if mode == "asis":
        return fx, fy, cx, cy, "asis"
    if mode != "auto":
        raise ValueError("k_frame은 auto | asis")

    def fit(W, H):   # 주점이 (s·W/2, s·H/2)라고 볼 때 s와 상대 오차
        s = (cx / (W / 2) + cy / (H / 2)) / 2
        return s, abs(cx - s * W / 2) / W + abs(cy - s * H / 2) / H

    s_p, e_p = fit(w, h)
    s_l, e_l = fit(h, w)
    rotated = e_l < e_p
    s = s_l if rotated else s_p
    if abs(s - 1) <= 0.05:   # 주점이 중앙에서 조금 벗어난 것은 비율 차이로 보지 않음
        s = 1.0
    did = []
    if rotated:
        fx, fy, cx, cy = fy, fx, s * w - cy, cx   # 회전 전 기준 이미지(s·h × s·w)를 시계 방향 90°
        did.append("rotated_cw90")
    if s != 1.0:
        fx, fy, cx, cy = fx / s, fy / s, cx / s, cy / s
        did.append(f"scaled x{1 / s:.4f}")
    return fx, fy, cx, cy, "+".join(did) or "asis"


def frame_info_for_tag(tag, image_path, name, frame_prefix="", k_frame="auto", fallback_fx=None,
                       camera_height=None):
    """꼬리표 → FrameInfo. 반환: (FrameInfo, K에 한 일)."""
    w, h = tag["w"], tag["h"]
    k = k_for_tag(tag, k_frame)
    if k is None:
        if fallback_fx is None:
            raise ValueError(f"프레임 {tag['i']}: K가 없음 (--fx로 가정값을 주면 assumed로 씀)")
        fx, fy, cx, cy, did, ksrc = fallback_fx, fallback_fx, w / 2, h / 2, "assumed", "assumed"
    else:
        fx, fy, cx, cy, did = k
        ksrc = "device"
    roll, pitch = tag.get("roll_deg"), tag.get("pitch_deg")
    psrc = "device" if roll is not None and pitch is not None else "assumed"
    fi = FrameInfo(
        frame_id=f"{frame_prefix}{name}/f{tag['i']:05d}", image_width=w, image_height=h,
        image_path=str(image_path) if image_path else None,
        intrinsics=Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, source=ksrc),
        pose=Pose(roll_deg=roll if roll is not None else 0.0, pitch_deg=pitch if pitch is not None else 0.0,
                  camera_height_m=camera_height, source=psrc),
        capture_time_s=tag["t"])
    return fi, did


# ---------------------------------------------------------------- 영상 풀기 (OpenCV)

def _write_jpg(cv2, path, img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"JPEG 만들기 실패: {path}")
    Path(path).write_bytes(buf.tobytes())   # cv2.imwrite는 윈도에서 한글 경로를 못 씀


def extract(folder, tags, out_dir, quality=95):
    """고른 프레임만 영상에서 꺼내 out_dir/f<i>.jpg로 저장. 반환: ({i: 경로}, 영상 정보)."""
    import cv2
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {t["i"]: t for t in tags}
    cap = cv2.VideoCapture(str(folder / "video.mov"))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {folder / 'video.mov'}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    rot_meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
    paths, i, last = {}, 0, max(want) if want else -1
    rotated = False
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, img = cap.retrieve()
            if not ok:
                raise ValueError(f"프레임 {i}를 풀 수 없음")
            t = want[i]
            H, W = img.shape[:2]
            if (W, H) != (t["w"], t["h"]):
                if (H, W) == (t["w"], t["h"]) and rot_meta in (90, 270, -90):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if rot_meta == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated = True
                else:
                    raise ValueError(f"프레임 {i} 크기 {W}x{H}가 frames.jsonl {t['w']}x{t['h']}와 다름 (회전 정보 {rot_meta})")
            p = out_dir / f"f{i:05d}.jpg"
            _write_jpg(cv2, p, img, quality)
            paths[i] = p
        i += 1
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    missing = sorted(set(want) - set(paths))
    return paths, {"rotation_meta": rot_meta, "rotated_by_reader": rotated, "frame_count_video": n_video,
                   "missing": missing}


def load(folder, rate=5.0, pick="gyro", out_dir=None, frame_prefix="", k_frame="auto", fallback_fx=None,
         camera_height=None, quality=95):
    """녹화본 → (FrameInfo 목록, 입력 정보 dict). 프레임은 out_dir(기본 <녹화본>/frames_<N>fps_<pick>)에 jpg로 꺼낸다."""
    folder = Path(folder)
    meta, tags = read_meta(folder), read_tags(folder)
    chosen = select(tags, rate, pick)
    if out_dir is None:
        out_dir = folder / ("frames_all" if pick == "all" else f"frames_{rate:g}fps_{pick}")
    paths, video = extract(folder, chosen, out_dir, quality)
    frames, k_did = [], {}
    for t in chosen:
        if t["i"] not in paths:
            continue
        fi, did = frame_info_for_tag(t, paths[t["i"]], folder.name, frame_prefix, k_frame, fallback_fx, camera_height)
        k_did[did] = k_did.get(did, 0) + 1
        frames.append(fi)
    info = {"kind": "recording", "name": folder.name, "rate": None if pick == "all" else rate, "pick": pick,
            "frames_in_recording": len(tags), "frames_selected": len(chosen), "frames_read": len(frames),
            "k_frame": k_frame, "k_handling": k_did, "video": video, "mount": mount_of(meta),
            "meta": {k: meta.get(k) for k in ("format", "format_version", "device", "ios", "app_version",
                                               "started_wall")}}
    info["baro"] = baro_summary(read_baro(folder), meta)
    if video["frame_count_video"] and video["frame_count_video"] != len(tags):
        info["warning"] = f"영상 프레임 {video['frame_count_video']}개 ≠ frames.jsonl {len(tags)}줄"
    return frames, info
