"""브랜치 vision-integration 공통 경로·불러오기 준비.

integration/ 스크립트는 맨 위에서 `import paths`만 하면 패키지 indoornav와 비전 코어 사본(vision/)을 불러올 수 있다.
vision/ 안 파일은 localization_design이 원본이라 여기서 고치지 않는다(vision/MANIFEST.json, docs/vision_integration/README.md).

녹화별 값은 integration/recordings/<이름>.json 하나에서 읽는다(RC). 어느 녹화인지는 환경 변수 I1_RECORDING(없으면 route2_1030 = 1차 답사 보행).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
INPUTS = VISION / "inputs"            # 역사 패키지 + I1 입력 (저장소 .gitignore가 data/를 무시해서 inputs/)
HERE = ROOT / "integration"
OUT = HERE / "out"
RECORDINGS = HERE / "recordings"      # 녹화별 설정
RECORDING = os.environ.get("I1_RECORDING", "route2_1030")
RC = json.loads((RECORDINGS / f"{RECORDING}.json").read_text(encoding="utf-8"))
I1 = ROOT / RC["inputs"]              # 관측·임시 PDR 걸음·정답 (1차 녹화: vision/inputs/i1)
REC = ROOT / RC["rec"]                # 앱 녹화 motion.csv·baro.csv·meta.json·marks.jsonl (1차 녹화: 영상은 안 옮김)
REC_OUT = ROOT / RC["out"]            # indoornav 걸음·재생 결과 (1차 녹화: integration/out/i1)
ORIG_META = ROOT / "maps/dongchun/map_meta.json"
PDR_CONFIG = ROOT / "configs/pdr_tuned_v4.toml"
PF_CONFIG = ROOT / "configs/pf_home_v3.toml"
MASK_DIR = HERE / "maps"              # 방 뺀 보행 마스크 사본 + 그 마스크를 가리키는 map_meta 사본
MASK_META = MASK_DIR / "map_meta_noroom.json"
FLOOR_NAME = {-1: "B01", -2: "B02"}
VISION_PY = os.environ.get("I1_VISION_PY", r"C:\Users\kjh\AppData\Local\Programs\Python\Python312\python.exe")   # cv2·torch 있는 파이썬(브랜치 .venv엔 없음)

for p in (ROOT, VISION, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def rec_t0():
    """녹화 첫 영상 프레임 호스트 시각 (meta.json recording.first_t, 없으면 frames.jsonl 첫 줄). 재생 시각 t = 호스트 시각 - 이것.
    (localization_design truth.py의 REC_T0 = 651441.188447과 4e-8 s 차이)"""
    first = json.loads((REC / "meta.json").read_text(encoding="utf-8")).get("recording", {}).get("first_t")
    if first is not None:
        return float(first)
    with open(REC / "frames.jsonl", encoding="utf-8") as f:
        return float(json.loads(f.readline())["t"])


def excluded(t):
    """채점에서 뺄 구간(RC exclude: 에스컬레이터 등) 안인가. t 배열 → bool 배열"""
    import numpy as np
    t = np.asarray(t)
    out = np.zeros(t.shape, bool)
    for a, b in RC["exclude"]:
        out |= (t > a) & (t < b)
    return out
