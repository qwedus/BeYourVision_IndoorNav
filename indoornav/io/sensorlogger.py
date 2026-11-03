"""Sensor Logger(iOS) 세션 로더.

세션 = 폴더 하나(또는 내보낸 zip 하나). 안에 센서별 CSV가 있다.
  data/20261012_S1_normal_01/Accelerometer.csv, Gyroscope.csv, Gravity.csv, ...

Sensor Logger CSV 공통 형식 (실제 파일로 확인 후 docs/spec_data_format.md에 기록할 것):
  time            : UNIX epoch [ns]
  seconds_elapsed : 기록 시작 후 경과 시간 [s]   ← 공통 시간축으로 사용
  x, y, z         : 기기 좌표계 3축 값 (파일에 따라 z,y,x 순서로 저장됨, 이름으로 읽으므로 무관)

단위·의미가 앱 설정(표준화 옵션)에 따라 달라질 수 있어서, 로더는 이름이 아니라
값으로 확인한다:
  - 중력 벡터 크기가 ~1 이면 g 단위 → m/s²로 변환
  - 가속도 평균 크기가 ~9.8 이면 중력 포함(total), ~0 이면 중력 제거(user)로 판단
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665

# 파일 이름 후보 (앱 버전에 따라 이름이 조금씩 다를 수 있어 여러 개를 허용)
_FILES = {
    "acc": ["Accelerometer.csv"],
    "acc_total": ["TotalAcceleration.csv"],
    "gravity": ["Gravity.csv"],
    "gyro": ["Gyroscope.csv"],
    "gyro_raw": ["GyroscopeUncalibrated.csv"],
    "orientation": ["Orientation.csv"],
    "mag": ["Magnetometer.csv"],
    "baro": ["Barometer.csv"],
    "pedometer": ["Pedometer.csv"],
    "annotation": ["Annotation.csv", "Annotations.csv"],
    "metadata": ["Metadata.csv"],
}


@dataclass
class Session:
    """공통 시간축(fs Hz)으로 정렬된 한 세션의 센서 데이터. 모든 벡터는 기기 좌표계."""

    name: str
    fs: float
    t: np.ndarray                      # [N] 경과 시간 [s]
    acc_user: np.ndarray               # [N,3] 중력 제거 가속도 [m/s²]
    gravity: np.ndarray                # [N,3] 중력 벡터 [m/s²] (지구 방향)
    gyro: np.ndarray                   # [N,3] 회전 속도 [rad/s]
    yaw: np.ndarray | None = None      # [N] iPhone 자세 yaw [rad] (unwrap 됨), 없으면 None
    pressure: np.ndarray | None = None # [N] 기압 [hPa 또는 kPa, 원본 단위 그대로]
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)  # seconds_elapsed, text
    annotations_raw: pd.DataFrame = field(default_factory=pd.DataFrame)  # uptime_s, text (시계 보정 전 원본)
    rec_epoch_s: float | None = None   # 기록 시작 시각 (UNIX epoch [s], 공통 시간축 t=0 지점)
    pedometer: pd.DataFrame = field(default_factory=pd.DataFrame)    # iPhone 걸음 수 원본
    raw: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def iphone_steps(self) -> int | None:
        """iPhone 내장 계보기 걸음 수 (비교 기준값). 없으면 None."""
        if self.pedometer.empty or "steps" not in self.pedometer:
            return None
        # Sensor Logger의 steps는 기록 시작 후 누적값으로 가정 (실제 파일로 확인 필요)
        s = self.pedometer["steps"].dropna()
        return int(s.max()) if len(s) else None


# ---------------------------------------------------------------- 파일 읽기
def _read_csvs(path: Path) -> dict[str, pd.DataFrame]:
    """폴더 또는 zip 안의 CSV를 {파일명: DataFrame}으로 읽는다."""
    out: dict[str, pd.DataFrame] = {}

    def add(name: str, data: bytes) -> None:
        # 기록이 없는 센서는 0바이트 파일로 내보내짐 (예: Annotation, 느린 걸음의 Pedometer)
        out[name] = pd.read_csv(io.BytesIO(data)) if data.strip() else pd.DataFrame()

    if path.is_dir():
        for f in path.glob("*.csv"):
            add(f.name, f.read_bytes())
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    add(Path(n).name, z.read(n))
    else:
        raise FileNotFoundError(f"세션 폴더/zip이 아님: {path}")
    if not out:
        raise FileNotFoundError(f"CSV가 없음: {path}")
    return out


def _pick(csvs: dict[str, pd.DataFrame], key: str) -> pd.DataFrame | None:
    for name in _FILES[key]:
        if name in csvs and not csvs[name].empty:
            return csvs[name]
    return None


def _xyz(df: pd.DataFrame) -> np.ndarray:
    return df[["x", "y", "z"]].to_numpy(dtype=float)


def _interp(t_new: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return np.interp(t_new, t, v)
    return np.column_stack([np.interp(t_new, t, v[:, i]) for i in range(v.shape[1])])


# ---------------------------------------------------------------- 메인
def load_session(path: str | Path, fs: float = 100.0) -> Session:
    """세션 폴더/zip을 읽어 fs Hz 공통 시간축으로 리샘플한 Session을 반환."""
    path = Path(path)
    csvs = _read_csvs(path)

    acc = _pick(csvs, "acc")
    acc_total = _pick(csvs, "acc_total")
    grav = _pick(csvs, "gravity")
    gyro = _pick(csvs, "gyro")
    if gyro is None or grav is None or (acc is None and acc_total is None):
        raise ValueError(
            f"{path.name}: 필수 파일 누락 (Gyroscope, Gravity, Accelerometer 필요). "
            f"있는 파일: {sorted(csvs)}"
        )

    # 공통 시간축: 자이로·중력·가속도가 모두 겹치는 구간
    srcs = [d for d in (acc if acc is not None else acc_total, grav, gyro)]
    t0 = max(d["seconds_elapsed"].iloc[0] for d in srcs)
    t1 = min(d["seconds_elapsed"].iloc[-1] for d in srcs)
    t = np.arange(t0, t1, 1.0 / fs)

    def rs(df: pd.DataFrame, v: np.ndarray) -> np.ndarray:
        return _interp(t, df["seconds_elapsed"].to_numpy(float), v)

    # 단위: 중력 벡터 크기로 판단 (~1 → g 단위, ~9.8 → m/s²). 가속도 파일도 같은 단위라고 가정.
    g_file = rs(grav, _xyz(grav))
    g_mag = float(np.median(np.linalg.norm(g_file, axis=1)))
    scale = G if g_mag < 2.0 else 1.0
    g = g_file * scale

    a_src = acc if acc is not None else acc_total
    a = rs(a_src, _xyz(a_src)) * scale
    # 중력 포함 여부: 평균 크기가 중력의 절반을 넘으면 중력 포함(total)으로 본다
    if np.median(np.linalg.norm(a, axis=1)) > 0.5 * G:
        a_user = a - g
    else:
        a_user = a

    w = rs(gyro, _xyz(gyro))

    yaw = None
    ori = _pick(csvs, "orientation")
    if ori is not None and "yaw" in ori:
        yaw = rs(ori, np.unwrap(ori["yaw"].to_numpy(float)))

    pressure = None
    baro = _pick(csvs, "baro")
    if baro is not None and "pressure" in baro:
        pressure = rs(baro, baro["pressure"].to_numpy(float))

    ann = _pick(csvs, "annotation")
    ann_raw = pd.DataFrame()
    rec_epoch = None
    first = srcs[0]
    if "time" in first:
        # t=0 지점의 epoch 시각 (센서 CSV의 time 은 epoch [ns])
        rec_epoch = float(first["time"].iloc[0]) / 1e9 - float(first["seconds_elapsed"].iloc[0]) + t0
    if ann is not None and "time" in ann:
        # 1.68(iOS)의 Annotation time 은 epoch 이 아니라 '폰 부팅 후 경과 시간' [ns] (2026-10-29 확인).
        # 같은 부팅·잠들지 않은 동안은 epoch = uptime + C (C 일정) → align_annotations() 로 변환.
        ann_raw = pd.DataFrame({"uptime_s": ann["time"].to_numpy(float) / 1e9, "text": ann["text"].to_numpy()})
    if ann is not None:
        # Sensor Logger 1.68(iOS)은 Annotation 시각을 다른 시계 기준으로 저장해서
        # seconds_elapsed가 -1.79e9 같은 값이 된다. 기록 구간 밖의 시각은 버린다.
        ann = ann[(ann["seconds_elapsed"] >= t0 - 1) & (ann["seconds_elapsed"] <= t1 + 1)]
        ann = ann if not ann.empty else None
    ped = _pick(csvs, "pedometer")

    return Session(
        name=path.stem,
        fs=fs,
        t=t - t[0],
        acc_user=a_user,
        gravity=g,
        gyro=w,
        yaw=yaw,
        pressure=pressure,
        annotations=(ann.assign(seconds_elapsed=ann["seconds_elapsed"] - t0)
                     if ann is not None else pd.DataFrame()),
        pedometer=ped if ped is not None else pd.DataFrame(),
        annotations_raw=ann_raw,
        rec_epoch_s=rec_epoch,
        raw=csvs,
    )


def describe(path: str | Path) -> pd.DataFrame:
    """세션 안 CSV 파일별 컬럼·샘플 수·실제 주파수 요약 (spec_data_format.md 채울 때 사용)."""
    rows = []
    for name, df in sorted(_read_csvs(Path(path)).items()):
        hz = np.nan
        if "seconds_elapsed" in df and len(df) > 1:
            dt = np.diff(df["seconds_elapsed"].to_numpy(float))
            hz = 1.0 / np.median(dt) if np.median(dt) > 0 else np.nan
        rows.append({"file": name, "rows": len(df), "hz": round(hz, 1),
                     "columns": ", ".join(map(str, df.columns))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Annotation 시계 보정
def align_annotations(session: Session, clock_offset_s: float) -> pd.DataFrame:
    """Annotation(부팅 후 경과 시간)을 세션 시간축 [s]으로 변환. clock_offset_s = epoch − uptime."""
    a = session.annotations_raw
    if a.empty or session.rec_epoch_s is None:
        return pd.DataFrame(columns=["t", "text"])
    return pd.DataFrame({"t": a["uptime_s"] + clock_offset_s - session.rec_epoch_s, "text": a["text"]})


def estimate_annotation_offset(session: Session, search_s: tuple[float, float] = (-5, 25)) -> tuple[float, float]:
    """버튼 누를 때 폰에 생기는 충격(가속도 급변)에 Annotation 간격 패턴을 맞춰 시계 오프셋을 찾는다.

    반환: (clock_offset_s, 신뢰도). 신뢰도 = 최고 점수 − 1 s 이상 떨어진 차순위 점수.
    신뢰도 3 미만이면 믿지 말 것 (같은 날 신뢰도 높은 세션의 오프셋을 쓰는 편이 낫다).
    """
    a = session.annotations_raw
    if a.empty or session.rec_epoch_s is None:
        return float("nan"), 0.0
    rel = a["uptime_s"].to_numpy() - a["uptime_s"].iloc[0]
    j = np.r_[np.linalg.norm(np.diff(session.acc_user, axis=0), axis=1) * session.fs, 0.0]
    fs = session.fs
    offs = np.arange(search_s[0], search_s[1], 0.01)
    sc = np.full(len(offs), -np.inf)
    for k, o in enumerate(offs):
        idx = np.round((rel + o) * fs).astype(int)
        if idx.min() < 0 or idx.max() >= len(j):
            continue
        v = 0.0
        for i in idx:
            peak = j[max(0, i - 15):i + 16].max()
            base = np.median(j[max(0, i - 200):i + 200]) + 1e-6
            v += np.log(peak / base)
        sc[k] = v
    b = int(np.argmax(sc))
    second = sc[np.abs(offs - offs[b]) > 1.0].max()
    t_first = offs[b]                                    # 첫 Annotation 의 세션 시각
    clock = t_first + session.rec_epoch_s - a["uptime_s"].iloc[0]
    return float(clock), float(sc[b] - second)
