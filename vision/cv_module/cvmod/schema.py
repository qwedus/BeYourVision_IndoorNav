"""데이터 형식: FrameInfo · Detection · Observation · FrameResult (명세: docs/명세.md).

- dict(JSON)로 바꾸는 to_dict / from_dict, 작은 입력 검증(validate_*)을 함께 둔다.
- JSON 규칙: 모르는 값·계산 안 한 값은 null. NaN·무한대는 절대 내보내지 않는다(dumps에서 allow_nan=False).
"""
import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

SCHEMA_VERSION = "0.1.0"
STATUS = ("ok", "empty", "failed")


class InputError(ValueError):
    """입력 계약 위반 (잘못된 K, 박스, NaN 등)."""


def _finite(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise InputError(f"{name}: 유한한 숫자가 아님 ({x!r})")
    return float(x)


def _opt_finite(x, name):
    return None if x is None else _finite(x, name)


# ---------------------------------------------------------------- FrameInfo

@dataclass
class Intrinsics:
    """카메라 내부 값 K [px]. width·height = 이 K가 대응하는 이미지 크기."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "assumed"          # assumed | arkit | calibration | ...
    distortion: str = "none"         # none(왜곡 없음·이미 폄) 만 지원


@dataclass
class Pose:
    """촬영 순간 카메라 자세 [도]. 기기 센서 → 이 약속으로 바꾸는 것은 앱(호출하는 쪽) 책임."""
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: Optional[float] = None  # 기하 계산에는 안 씀. PF에 넘기는 값
    yaw_ref: Optional[str] = None    # relative | magnetic | map | None
    camera_height_m: Optional[float] = None
    source: str = "assumed"          # assumed | device


@dataclass
class FrameInfo:
    frame_id: str
    image_width: int
    image_height: int
    intrinsics: Intrinsics
    pose: Pose = field(default_factory=Pose)
    capture_time_s: Optional[float] = None   # 촬영 시각 (PF 시간축)
    process_time_s: Optional[float] = None   # 처리한 시각 (참고)
    image_path: Optional[str] = None
    image_orientation: str = "up"            # 화면 위 = 이미지 위로 돌려 놓은 상태만 받음
    mirrored: bool = False                   # 좌우 반전 이미지는 받지 않음

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["intrinsics"] = Intrinsics(**d["intrinsics"])
        d["pose"] = Pose(**(d.get("pose") or {}))
        return FrameInfo(**d)


def validate_frame_info(fi):
    """입력 계약 검사. 문제가 있으면 InputError. 반환: 이미지 크기에 맞춘 (fx, fy, cx, cy)."""
    if not fi.frame_id:
        raise InputError("frame_id 없음")
    W, H = fi.image_width, fi.image_height
    if not (isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0):
        raise InputError(f"이미지 크기 이상: {W}x{H}")
    if fi.image_orientation != "up" or fi.mirrored:
        raise InputError("이미지는 위가 위로 돌려 놓고, 좌우 반전 없이 넣어야 함")
    k = fi.intrinsics
    if k.distortion != "none":
        raise InputError(f"왜곡 모델 '{k.distortion}'은 지원 안 함 (펴서 넣거나 none)")
    fx, fy, cx, cy = (_finite(getattr(k, n), f"K.{n}") for n in ("fx", "fy", "cx", "cy"))
    if fx <= 0 or fy <= 0:
        raise InputError(f"K 초점 거리는 양수여야 함 (fx={fx}, fy={fy})")
    if not (isinstance(k.width, int) and isinstance(k.height, int) and k.width > 0 and k.height > 0):
        raise InputError(f"K 기준 이미지 크기 이상: {k.width}x{k.height}")
    if not (0 < cx < k.width and 0 < cy < k.height):
        raise InputError(f"주점(cx, cy)=({cx}, {cy})가 K 기준 이미지({k.width}x{k.height}) 밖")
    if (k.width, k.height) != (W, H):
        # 같은 비율로 줄이거나 늘린 이미지만 K를 비례로 맞춘다 (자르기·회전은 받지 않음)
        sx, sy = W / k.width, H / k.height
        if abs(sx - sy) > 1e-3 * max(sx, sy):
            raise InputError(f"K 기준 크기 {k.width}x{k.height}와 이미지 {W}x{H}의 비율이 다름")
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    p = fi.pose
    _finite(p.roll_deg, "pose.roll_deg")
    _finite(p.pitch_deg, "pose.pitch_deg")
    _opt_finite(p.yaw_deg, "pose.yaw_deg")
    _opt_finite(p.camera_height_m, "pose.camera_height_m")
    _opt_finite(fi.capture_time_s, "capture_time_s")
    _opt_finite(fi.process_time_s, "process_time_s")
    return fx, fy, cx, cy


# ---------------------------------------------------------------- Detection

@dataclass
class Detection:
    """검출기 하나가 낸 박스 하나 (검출기 종류와 무관한 표준형)."""
    detection_id: str
    source: str                  # 검출기 이름 (yolo26n, yoloe, ocr, ...)
    label: str                   # 검출기 자기 종류 이름 (exit, navy, ...)
    score: float                 # 검출 점수 0~1 (그 검출기 안에서만 의미)
    box_xyxy: List[float]        # 원본 이미지 좌상단 원점 픽셀 [x1, y1, x2, y2]
    parent_detection_id: Optional[str] = None   # OCR 글자 박스면 그것이 속한 표지판 박스 id
    text: Optional[str] = None                  # OCR 글자 (지금은 항상 None)


BOX_MARGIN_PX = 1.0   # 박스가 이미지 밖으로 이만큼까지는 허용 (반올림 여유)


def validate_detection(d, W, H):
    """박스·점수 검사. 문제 있으면 InputError."""
    if not isinstance(d.box_xyxy, (list, tuple)) or len(d.box_xyxy) != 4:
        raise InputError(f"{d.detection_id}: 박스는 숫자 4개 [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (_finite(v, f"{d.detection_id}.box") for v in d.box_xyxy)
    if not (x2 > x1 and y2 > y1):
        raise InputError(f"{d.detection_id}: x2 > x1, y2 > y1 이어야 함 {d.box_xyxy}")
    m = BOX_MARGIN_PX
    if x1 < -m or y1 < -m or x2 > W + m or y2 > H + m:
        raise InputError(f"{d.detection_id}: 박스가 이미지({W}x{H}) 밖 {d.box_xyxy}")
    s = _finite(d.score, f"{d.detection_id}.score")
    if not 0.0 <= s <= 1.0:
        raise InputError(f"{d.detection_id}: score는 0~1 ({s})")


# ---------------------------------------------------------------- Observation

@dataclass
class Observation:
    """PF에 넘길 기하 관측 하나. 카메라 기준 (역 좌표 아님)."""
    observation_id: str
    detection_id: str
    source: str
    label: str
    map_types: List[str]         # 지도 종류 후보 (일대다 가능: map → station_map, route_map)
    score: float                 # 검출 점수 그대로 (지도 매칭 확률이 아님)
    point_px: List[float]        # 각도를 잰 점 [u, v] (지금은 박스 중심)
    bearing_deg: float
    elevation_deg: float
    distance_m: Optional[float] = None           # 기본 null. 높이를 명시했고 기하가 유효할 때만
    distance_landmark_height_m: Optional[float] = None
    parent_detection_id: Optional[str] = None


# ---------------------------------------------------------------- FrameResult

@dataclass
class Dropped:
    detection_id: str
    label: str
    score: float
    reason: str      # drop_label | not_in_map | unknown_label | below_threshold | invalid_detection


@dataclass
class FrameResult:
    frame_info: FrameInfo
    status: str                  # ok | empty | failed
    observations: List[Observation] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    dropped: List[Dropped] = field(default_factory=list)
    error: Optional[str] = None
    run_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        check_schema_version(d.get("schema_version"))
        return FrameResult(
            frame_info=FrameInfo.from_dict(d["frame_info"]),
            status=d["status"],
            observations=[Observation(**o) for o in d.get("observations", [])],
            detections=[Detection(**o) for o in d.get("detections", [])],
            dropped=[Dropped(**o) for o in d.get("dropped", [])],
            error=d.get("error"), run_id=d.get("run_id"),
            schema_version=d["schema_version"])


def check_schema_version(v):
    """주 번호(첫 숫자)가 같으면 읽는다. 다르면 InputError."""
    if not isinstance(v, str) or v.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise InputError(f"schema_version {v!r}는 읽을 수 없음 (지금 {SCHEMA_VERSION})")


def dumps(obj, **kw):
    """JSON 문자열. NaN·무한대가 있으면 ValueError (null로 바꿔서 넣어야 함)."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, **kw)
