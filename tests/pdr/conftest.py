"""가짜 Sensor Logger 세션 생성기 (정답을 아는 합성 보행 데이터)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

G = 9.80665


def make_walk(tmp_path, legs=((10, 0.0),), cadence_hz=1.8, step_len=0.6, fs=100.0,
              amp=1.5, stand_s=5.0, turn_s=1.5, gyro_bias=0.0, unit_g=True, seed=0):
    """legs: [(걸음 수, 시작 전 회전각 deg), ...]. 폰은 세로로 가슴에 고정(중력 = -y).

    반환: (세션 폴더 경로, 정답 걸음 수, 정답 거리, 정답 순회전 deg)
    """
    rng = np.random.default_rng(seed)
    acc_up, yaw_rate = [], []

    def stand(sec):
        n = int(sec * fs)
        acc_up.append(rng.normal(0, 0.02, n)); yaw_rate.append(np.zeros(n))

    stand(stand_s)
    total_steps, total_turn = 0, 0.0
    for n_steps, turn_deg in legs:
        if turn_deg:
            n = int(turn_s * fs)
            yaw_rate.append(np.full(n, np.radians(turn_deg) / turn_s))
            acc_up.append(rng.normal(0, 0.05, n))
            total_turn += turn_deg
        n = int(n_steps / cadence_hz * fs)
        tt = np.arange(n) / fs
        # 걸음마다 봉우리 하나: -cos 로 시작해 걸음 n_steps 개의 봉우리
        acc_up.append(amp * np.sin(2 * np.pi * cadence_hz * tt - np.pi / 2 + 1e-9)
                      * 1.0 + rng.normal(0, 0.05, n))
        yaw_rate.append(rng.normal(0, 0.01, n))
        total_steps += n_steps
    stand(stand_s)

    a_up = np.concatenate(acc_up)
    w_up = np.concatenate(yaw_rate) + gyro_bias
    N = len(a_up)
    t = np.arange(N) / fs
    # 기기 좌표계: 세로 장착 → 위쪽 = +y, 중력(지구 방향) = -y
    grav = np.tile([0.0, -G, 0.0], (N, 1))
    acc = np.column_stack([rng.normal(0, 0.05, N), a_up, rng.normal(0, 0.05, N)])
    gyro = np.column_stack([rng.normal(0, 0.005, N), w_up, rng.normal(0, 0.005, N)])

    s = 1 / G if unit_g else 1.0
    d = tmp_path / "20261012_S1_normal_01"
    d.mkdir()
    base = {"time": (t * 1e9).astype(np.int64) + 1_760_000_000_000_000_000, "seconds_elapsed": t}
    for name, v, scale in (("Accelerometer", acc, s), ("Gravity", grav, s), ("Gyroscope", gyro, 1.0)):
        pd.DataFrame({**base, "z": v[:, 2] * scale, "y": v[:, 1] * scale, "x": v[:, 0] * scale}) \
            .to_csv(d / f"{name}.csv", index=False)
    yaw = np.cumsum(w_up) / fs
    pd.DataFrame({**base, "qz": 0, "qy": 0, "qx": 0, "qw": 1, "roll": 0, "pitch": 0,
                  "yaw": np.angle(np.exp(1j * yaw))}).to_csv(d / "Orientation.csv", index=False)
    pd.DataFrame({"time": [base["time"][-1]], "seconds_elapsed": [t[-1]], "steps": [total_steps]}) \
        .to_csv(d / "Pedometer.csv", index=False)
    return d, total_steps, total_steps * step_len, total_turn


@pytest.fixture
def walk(tmp_path):
    return lambda **kw: make_walk(tmp_path, **kw)
