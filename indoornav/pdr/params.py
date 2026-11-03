"""PDR 파라미터. 튜닝 대상 숫자는 전부 여기 모인다.

기본값은 configs/pdr_default.toml 에서 읽고, 실험할 때는 TOML을 복사해서 바꾼다.
코드 안에 숫자를 직접 박지 말 것 (튜닝 이력이 남지 않음).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class PreprocessParams:
    lpf_cutoff_hz: float = 3.0      # 걸음 신호 저역통과 차단 주파수
    lpf_order: int = 4
    mode: str = "offline"           # "offline": 녹화 전체를 앞뒤로 필터 (분석용)
                                    # "delayed": 앱과 같은 방식. delay_s 만큼 늦게, 최근 window_s 구간만 앞뒤로 필터
                                    # "causal" : 과거 데이터만 쓰는 단방향 필터 (지연 없음, 파형 왜곡)
    delay_s: float = 0.5            # mode="delayed" 일 때 허용 지연 [s]
    window_s: float = 3.0           # mode="delayed" 일 때 한 번에 필터하는 최근 구간 길이 [s]
    hop_s: float = 0.1              # mode="delayed" 일 때 갱신 주기 [s] (앱에서 몇 초마다 계산하는지)


@dataclass(frozen=True)
class StepParams:
    signal: str = "vertical"        # "vertical"(중력 방향 성분) | "norm"(가속도 크기)
    min_height: float = 0.5         # 봉우리 최소 높이 [m/s²]
    min_prominence: float = 0.5     # 봉우리 돌출 정도 [m/s²]
    min_interval_s: float = 0.30    # 걸음 최소 간격 [s] (빠른 걸음 ~0.4 s)
    max_interval_s: float = 2.0     # 이보다 긴 간격 뒤 첫 걸음은 보행 재시작으로 간주
    method: str = "peak"            # "peak": 수직 가속도 봉우리 (기존) | "adaptive": 속도 적응형 (step_adaptive.py)
    slow_energy: float = 0.25       # adaptive: 수직 흔들림(표준편차)이 이보다 작으면 느린 걸음 모드 [m/s²]
    win_s: float = 6.0              # adaptive: 모드·주기 판단 창 길이 [s]
    hop_s: float = 1.0              # adaptive: 창 이동 간격 [s]
    mode_smooth: int = 2            # adaptive: 모드 다수결에 쓰는 앞뒤 창 개수
    adaptive_prom: float = 0.8      # adaptive: 봉우리 돌출 기준 (최근 흔들림 크기 대비 배수)
    min_quality: float = 0.2        # adaptive: 이보다 주기성이 약하면 걷지 않는 구간으로 봄 (0~1)
    min_energy: float = 0.08        # adaptive: 흔들림(표준편차)이 이보다 작으면 걷지 않는 구간 [m/s²]
                                    #   서 있을 때 약 0.03~0.05, 매우 느린 걸음 약 0.18 (2026-10-11 측정)


@dataclass(frozen=True)
class LengthParams:
    model: str = "weinberg"         # "weinberg" | "weinberg_freq" | "constant"
    k: float = 0.48                 # Weinberg 계수 (사람별 보정 대상)
    exponent: float = 0.25          # Weinberg 지수 (보통 1/4 고정)
    k_slow: float = 0.0             # 느린 걸음 모드(step.method="adaptive") 전용 K. 0 이면 k 를 그대로 사용
    k_freq: float = 0.0             # model="weinberg_freq": 걸음 빈도 계수 [m per (steps/s)]
    offset_m: float = 0.0           # model="weinberg_freq": 상수항 [m]
    constant_m: float = 0.60        # model="constant" 일 때 보폭 [m]
    min_m: float = 0.10             # 보폭 하한·상한 (이상치 클리핑)
    max_m: float = 1.20
    min_bout_steps: int = 0         # 연속 걸음이 이 개수 미만인 묶음은 제자리 움직임으로 보고 보폭 0 (0 = 끔)
    bout_gap_s: float = 3.0         # 걸음 사이가 이보다 길면 다른 묶음으로 나눔 [s] (매우 느린 걸음 간격 최대 약 2.8 s)


@dataclass(frozen=True)
class HeadingParams:
    source: str = "gyro"            # "gyro"(자이로 수직축 적분) | "orientation"(iPhone yaw)
    zaru: bool = False              # 정지 구간 자이로 바이어스 보정
    still_acc_std: float = 0.15     # 정지 판별: 가속도 크기 표준편차 임계값 [m/s²]
    still_gyro_std: float = 0.05    # 정지 판별: 자이로 크기 표준편차 임계값 [rad/s]
    still_window_s: float = 1.0     # 정지 판별 창 길이 [s]
    step_average: bool = True       # 걸음 구간 평균 헤딩 사용 (몸 흔들림 상쇄)


@dataclass(frozen=True)
class PDRParams:
    fs: float = 100.0
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    step: StepParams = field(default_factory=StepParams)
    length: LengthParams = field(default_factory=LengthParams)
    heading: HeadingParams = field(default_factory=HeadingParams)

    # ---------------------------------------------------------- 입출력
    @classmethod
    def from_toml(cls, path: str | Path) -> "PDRParams":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "PDRParams":
        sub = {"preprocess": PreprocessParams, "step": StepParams,
               "length": LengthParams, "heading": HeadingParams}
        kw = {}
        for f_ in fields(cls):
            if f_.name in sub:
                kw[f_.name] = _build(sub[f_.name], d.get(f_.name, {}), f_.name)
            elif f_.name in d:
                kw[f_.name] = d[f_.name]
        unknown = set(d) - {f_.name for f_ in fields(cls)}
        if unknown:
            raise KeyError(f"알 수 없는 섹션: {unknown}")
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **dotted) -> "PDRParams":
        """점 표기로 일부만 바꾼 복사본. 예: p.with_(**{"step.min_height": 0.4, "length.k": 0.5})"""
        p = self
        for key, val in dotted.items():
            sec, name = key.split(".")
            p = replace(p, **{sec: replace(getattr(p, sec), **{name: val})})
        return p


def _build(cls, d: dict, sec: str):
    d = dict(d)
    if sec == "preprocess" and "causal" in d:          # 옛 설정 파일 호환 (causal = true/false)
        d.setdefault("mode", "causal" if d.pop("causal") else "offline")
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"[{sec}] 알 수 없는 파라미터: {unknown}")
    return cls(**d)
