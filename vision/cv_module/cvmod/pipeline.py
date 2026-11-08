"""표준 Detection 목록 → 기하 Observation 목록 (FrameResult). 검출 라이브러리 없이 돈다.

순서 (한 검출마다)
    1. 박스·점수 검사 → 틀리면 dropped(invalid_detection)
    2. label_map·drop_labels 어디에도 없는 종류 → unknown_label
    3. 검출기·종류별 문턱값 미만 → below_threshold (문턱값을 먼저 봐서, 버린 광고도 '넘었던 것'만 남음)
    4. drop_labels(광고 등) → drop_label / label_map 이 빈 목록(지도에 없음) → not_in_map
    5. 박스 중심 → 방위각·앙각. 거리는 높이가 명시된 경우에만 (기본 null)
status: 입력(FrameInfo) 오류나 검출 실패 → failed / 관측 0개 → empty / 그 밖 → ok
"""
from . import geometry
from .schema import (Detection, Dropped, FrameResult, InputError, Observation,
                     validate_detection, validate_frame_info)


class Config:
    """설정 파일(config/*.json) 내용. 종류 매핑·문턱값·버릴 종류·기하 옵션."""

    def __init__(self, d):
        self.raw = d
        self.label_map = {k: list(v) for k, v in d["label_map"].items()}
        self.drop_labels = set(d.get("drop_labels", []))
        self.detectors = d.get("detectors", {})
        g = d.get("geometry", {})
        self.min_elevation_deg = float(g.get("min_elevation_deg", 2.0))
        # 거리용 랜드마크 높이: 명시한 지도 종류만. 기본은 비어 있어 거리 = null
        self.distance_heights = dict(g.get("distance_landmark_height_m") or {})

    def threshold(self, source, label):
        th = self.detectors.get(source, {}).get("thresholds", {})
        if label in th:
            return float(th[label])
        if "default" in th:
            return float(th["default"])
        raise InputError(f"검출기 '{source}'의 문턱값이 설정에 없음 (label {label})")


def process_frame(frame_info, detections, config, run_id=None, error=None):
    """FrameInfo + Detection 목록 → FrameResult. error가 있으면(검출 실패) 바로 failed."""
    res = FrameResult(frame_info=frame_info, status="failed", detections=list(detections),
                      run_id=run_id, error=error)
    if error is not None:
        return res
    try:
        fx, fy, cx, cy = validate_frame_info(frame_info)
    except InputError as e:
        res.error = f"입력 오류: {e}"
        return res
    pose = frame_info.pose
    for det in detections:
        try:
            validate_detection(det, frame_info.image_width, frame_info.image_height)
        except InputError:
            res.dropped.append(Dropped(det.detection_id, det.label, _num(det.score), "invalid_detection"))
            continue
        if det.label not in config.label_map and det.label not in config.drop_labels:
            reason = "unknown_label"
        elif det.score < config.threshold(det.source, det.label):
            reason = "below_threshold"
        elif det.label in config.drop_labels:
            reason = "drop_label"
        elif not config.label_map[det.label]:
            reason = "not_in_map"
        else:
            reason = None
        if reason:
            res.dropped.append(Dropped(det.detection_id, det.label, det.score, reason))
            continue
        res.observations.append(make_observation(det, config.label_map[det.label], (fx, fy, cx, cy),
                                                 pose, config, len(res.observations)))
    res.status = "ok" if res.observations else "empty"
    return res


def make_observation(det, map_types, K, pose, config, index):
    u, v = geometry.box_center(det.box_xyxy)
    az, el = geometry.pixel_to_angles(u, v, *K, roll_deg=pose.roll_deg, pitch_deg=pose.pitch_deg)
    dist, lm_h = None, None
    # 거리는 지도 종류 후보가 하나이고 그 높이를 설정에서 명시했을 때만
    if len(map_types) == 1 and map_types[0] in config.distance_heights:
        lm_h = float(config.distance_heights[map_types[0]])
        dist = geometry.horizontal_distance(el, pose.camera_height_m, lm_h, config.min_elevation_deg)
        if dist is None:
            lm_h = None
    return Observation(
        observation_id=f"{det.detection_id}/o{index}", detection_id=det.detection_id,
        source=det.source, label=det.label, map_types=list(map_types), score=det.score,
        point_px=[u, v], bearing_deg=az, elevation_deg=el,
        distance_m=dist, distance_landmark_height_m=lm_h,
        parent_detection_id=det.parent_detection_id)


def _num(x):
    """dropped 기록용: 숫자가 아니거나 NaN이면 0."""
    try:
        x = float(x)
        return x if x == x and abs(x) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0


def make_detection(frame_id, source, i, label, score, box):
    """검출 ID 규칙: '<frame_id>/<source>/d<번호>'."""
    return Detection(detection_id=f"{frame_id}/{source}/d{i}", source=source, label=label,
                     score=score, box_xyxy=list(box))
