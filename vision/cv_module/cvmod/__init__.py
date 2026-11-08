"""CV 모듈 초안: 검출기 어댑터 → 표준 Detection → 기하 Observation (FrameResult).

핵심(schema, geometry, pipeline)은 표준 라이브러리만 쓴다. 검출기 어댑터는 ultralytics, 그림은 Pillow가 필요하다.
"""
from .pipeline import Config, process_frame  # noqa: F401
from .schema import (SCHEMA_VERSION, Detection, FrameInfo, FrameResult, InputError,  # noqa: F401
                     Intrinsics, Observation, Pose)

__version__ = "0.1.0"
