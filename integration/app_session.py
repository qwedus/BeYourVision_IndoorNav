"""앱 녹화(capstone-recording v0.2) → 팀장 indoornav.io.Session (연결안 3절 1).

motion.csv 열: t(호스트 시계 s), g(중력, 기기 좌표, g 단위), a(사용자 가속도, g), w(회전 속도 rad/s), q(자세), m(자기장)
    acc_user = a · 9.80665, gravity = g · 9.80665 (iOS CMDeviceMotion 중력 = 지구 방향 = 팀장 Sensor Logger와 같은 약속),
    gyro = w. 세 벡터가 같은 기기 좌표라 가로 장착용 90° 회전은 넣지 않음(팀장 PDR은 중력으로 수직축을 찾음)
시계: 100 Hz 균일 격자로 선형 보간. Session.t = 호스트 시각 - 센서 첫 시각(t0_sensor).
    센서·영상 첫 호스트 시각을 따로 돌려준다(info) → 걸음 시각 = Session.t + t0_sensor (호스트 시계)
기압: baro.csv pressure_kpa를 같은 격자로 보간(원본 단위 kPa 그대로, 팀장 Session 약속과 같음)
"""
import json

import numpy as np
import pandas as pd

import paths  # noqa: F401
from indoornav.io import Session

G = 9.80665


def load_app_session(rec=paths.REC, fs=100.0):
    """→ (Session, info). info: t0_sensor(Session.t = 0의 호스트 시각), t0_video(첫 영상 프레임), fs, n."""
    m = pd.read_csv(rec / "motion.csv")
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    t_host = m["t"].to_numpy(float)
    t0 = float(t_host[0])
    t = np.arange(0.0, t_host[-1] - t0, 1.0 / fs)

    def rs(cols):
        return np.column_stack([np.interp(t, t_host - t0, m[c].to_numpy(float)) for c in cols])

    baro = pd.read_csv(rec / "baro.csv")
    pressure = np.interp(t, baro["t"].to_numpy(float) - t0, baro["pressure_kpa"].to_numpy(float))
    s = Session(name=rec.parent.name if rec.name == "rec" else rec.name, fs=fs, t=t,
                acc_user=rs(["ax", "ay", "az"]) * G, gravity=rs(["gx", "gy", "gz"]) * G, gyro=rs(["wx", "wy", "wz"]),
                yaw=None, pressure=pressure)
    info = dict(t0_sensor=t0, t0_video=float(meta["recording"]["first_t"]), fs=fs, n=len(t),
                mount=meta["mount"]["orientation"], reference_frame=meta["motion"]["reference_frame"])
    return s, info
