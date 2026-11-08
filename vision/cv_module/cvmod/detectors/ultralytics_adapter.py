"""YOLO26n(광고 학습) · YOLOE-26s(예시 임베딩을 구워 넣은 것) 어댑터. 하나만 골라 실행한다.

- 박스는 ultralytics가 resize/letterbox를 되돌린 원본 이미지 픽셀 xyxy (r.boxes.xyxy)를 그대로 쓴다.
- NMS: one2many 출력 + NMS (end2end=False). V2 4단계에서 PC 예측이 이 가지였음을 확인했고,
  아이폰 CoreML도 같은 방식으로 변환했다 (cv_test/facility/pc_nms_variant.py).
- conf_floor(0.01)까지 모두 Detection으로 내고, 종류별 문턱값은 pipeline이 적용한다
  → 저장한 결과를 재생할 때 문턱값만 바꿔 다시 돌릴 수 있다.
- 시험한 버전: ultralytics 8.4.177, torch 2.6.0+cu124 (2026-10-24)
"""
import hashlib
from pathlib import Path

from ..pipeline import make_detection

TESTED_ULTRALYTICS = "8.4.177"


def file_sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class UltralyticsAdapter:
    def __init__(self, name, cfg, model_path):
        import ultralytics
        from ultralytics import YOLO, YOLOE
        if model_path is None:
            raise ValueError(f"{name}: 모델 파일 경로(--model)가 필요함 (설정의 model_file = {cfg.get('model_file')})")
        self.name, self.cfg, self.path = name, cfg, Path(model_path)
        cls = YOLOE if cfg["kind"] == "ultralytics_yoloe" else YOLO
        self.model = cls(str(self.path))
        self.model.model.end2end = bool(cfg.get("end2end", False))   # fuse 전에 정해야 가지가 남음
        self.names = dict(self.model.names)
        import torch
        self._versions = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}

    def run_info(self):
        c = self.cfg
        return {"detector": self.name, "kind": c["kind"], "model_file": self.path.name,
                "model_sha256_16": file_sha256(self.path), "class_names": [self.names[i] for i in sorted(self.names)],
                "embedding": c.get("embedding"), "imgsz": c["imgsz"], "conf_floor": c["conf_floor"],
                "nms": {"iou": c["iou"], "agnostic": c["agnostic_nms"], "end2end": c.get("end2end", False)},
                "thresholds": c["thresholds"], "versions": self._versions,
                "tested_ultralytics": TESTED_ULTRALYTICS}

    def detect(self, frame):
        c = self.cfg
        r = self.model.predict(frame.image_path, conf=c["conf_floor"], imgsz=c["imgsz"], iou=c["iou"],
                               agnostic_nms=c["agnostic_nms"], verbose=False)[0]
        h, w = r.orig_shape
        if (w, h) != (frame.image_width, frame.image_height):
            raise ValueError(f"이미지 크기 {w}x{h}가 FrameInfo {frame.image_width}x{frame.image_height}와 다름")
        return [make_detection(frame.frame_id, self.name, i, self.names[int(k)], round(float(s), 4),
                               [round(float(v), 2) for v in b])
                for i, (b, s, k) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                                  r.boxes.cls.tolist()))]
