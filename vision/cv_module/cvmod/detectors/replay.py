"""저장된 Detection JSON 재생. 모델 없이 pipeline만 다시 돌릴 때(문턱값·K·자세를 바꿔 보기, 시험) 쓴다.

받는 파일: run.py가 쓴 결과 파일 그대로 ({"schema_version", "run", "frames": [FrameResult...]}).
각 프레임의 frame_info와 detections만 읽고, observations·dropped는 다시 계산한다.
"""
import json

from ..schema import Detection, FrameInfo, check_schema_version


class ReplayAdapter:
    name = "replay"

    def __init__(self, path):
        if path is None:
            raise ValueError("replay: 재생할 JSON 경로(--replay)가 필요함")
        self.path = str(path)
        d = json.loads(open(path, encoding="utf-8").read())
        check_schema_version(d.get("schema_version"))
        self.source_run = d.get("run", {})
        self.frames = [(FrameInfo.from_dict(f["frame_info"]), [Detection(**x) for x in f.get("detections", [])],
                        f.get("error") if f.get("status") == "failed" and not f.get("detections") else None)
                       for f in d["frames"]]

    def run_info(self):
        return {"detector": "replay", "replay_file": self.path, "source_run": self.source_run}
