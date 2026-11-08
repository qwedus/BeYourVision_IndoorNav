"""검출기 어댑터. 모두 같은 약속을 지킨다:

    adapter.name            : 검출기 이름 (Detection.source, 설정 detectors 키와 같음)
    adapter.run_info()      : 실행 메타데이터 dict (모델 파일·해시·입력 크기·NMS·라이브러리 버전)
    adapter.detect(frame)   : FrameInfo(image_path 필요) → Detection 목록. 실패하면 예외

ultralytics는 UltralyticsAdapter를 만들 때만 import한다 (기하 핵심은 ultralytics 없이 돎).
"""
from .replay import ReplayAdapter  # noqa: F401


def make_adapter(name, config, model_path=None, replay_path=None):
    if name == "replay":
        return ReplayAdapter(replay_path)
    from .ultralytics_adapter import UltralyticsAdapter
    return UltralyticsAdapter(name, config.detectors[name], model_path)
