"""브랜치 vision-integration 공통 경로·불러오기 준비.

integration/ 스크립트는 맨 위에서 `import paths`만 하면 팀장 패키지(indoornav)와 비전 코어 사본(vision/)을 불러올 수 있다.
vision/ 안 파일은 localization_design이 원본이라 여기서 고치지 않는다(vision/MANIFEST.json, docs/vision_integration/README.md).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
INPUTS = VISION / "inputs"            # 역사 패키지 + I1 입력 (팀장 .gitignore가 data/를 무시해서 inputs/)
I1 = INPUTS / "i1"
REC = I1 / "rec"                      # 앱 녹화 motion.csv·baro.csv·meta.json·marks.jsonl (영상은 안 옮김)
HERE = ROOT / "integration"
OUT = HERE / "out"
TEAM_META = ROOT / "maps/dongchun/map_meta.json"
PDR_CONFIG = ROOT / "configs/pdr_tuned_v4.toml"
PF_CONFIG = ROOT / "configs/pf_home_v3.toml"
MASK_DIR = HERE / "maps"              # 방 뺀 보행 마스크 사본 + 그 마스크를 가리키는 map_meta 사본
MASK_META = MASK_DIR / "map_meta_noroom.json"
FLOOR_NAME = {-1: "B01", -2: "B02"}

for p in (ROOT, VISION, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def rec_t0():
    """녹화 첫 영상 프레임 호스트 시각 (meta.json recording.first_t). 재생 시각 t = 호스트 시각 - 이것.
    (localization_design truth.py의 REC_T0 = 651441.188447과 4e-8 s 차이)"""
    return float(json.loads((REC / "meta.json").read_text(encoding="utf-8"))["recording"]["first_t"])
