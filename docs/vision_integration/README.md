# vision-integration 브랜치 — indoornav PDR·PF + 비전 관측 연결

indoornav PDR·파티클 필터(`indoornav/`)에 비전 관측(랜드마크 방위각 관측 모델)을 붙여, 1차 답사 보행 녹화(2번 출구 → 개찰구 3 → 원인재 방면 에스컬레이터 → 3-1 문, 118.6 s)를 오프라인 재생하고 **비전 끔/켬**을 비교하는 브랜치.
11-06부터 2차 답사 조끼 보행 8개를 같은 흐름으로 재생하고, 11-07부터 '그 시각까지 받은 값만 쓰는' 실시간 조건 재생(권장 조합 `rec`)도 함께 돌린다.
**브랜치만 받아서 2차 녹화 폴더로 처음부터 다시 만드는 방법은 아래 [다른 PC에서 2차 녹화로 처음부터](#다른-pc에서-2차-녹화로-처음부터).**

- `main`(9040d70)에서 갈라짐. **기존 파일(`indoornav/`, `configs/`, `maps/`, `scripts/`, `tests/`, `pyproject.toml`, `README.md` 등)은 고치지 않고 불러 쓰기만 한다.** 새 폴더 `vision/`, `integration/`, `docs/vision_integration/`만 더함
- 설계 근거: localization_design의 I1 연결안 문서 (11-04). 출발점을 아는 경우(known·drift)만 다룸
- 결과·작업 기록: [작업기록.md](작업기록.md), 비교표 `integration/out/i1/compare.md`, 채택 상태까지 누적 `integration/out/i1/compare_final.md`
- main 코드(`indoornav/`·`configs/`·지도 자료) 수정 건의: [main_수정건의.md](main_수정건의.md) (4건 + 검토했지만 뺀 것)

## 폴더 구조와 원본 쪽

| 폴더·파일 | 내용 | 원본 |
|---|---|---|
| `vision/obs_model.py` | 관측 모델 `obs_likelihood` (방위각·가시성·화각·바닥값), 계산 안 한 칸 구분 `computed_at` | **localization_design** (`cv_test/pf/obs_model.py`) |
| `vision/time_filter.py` | 시간 필터(최근 5프레임 중 3번) + cv_module 결과 읽기, `track_ids`, track별 반복 제한 `TrackLimiter` | **localization_design** (`cv_test/i1/time_filter.py`) |
| `vision/edge_faces.py` | 화면 끝 검출 → 판 끝점 비교 | **localization_design** (`cv_test/i1/edge_faces.py`) |
| `vision/bearing_fix.py` | 걸음 사이 회전 Δ 계산 | **localization_design** (`cv_test/i1/bearing_fix.py`) |
| `vision/text_check.py` | 글자 확인 결정 파일 읽기·적용 + 판정 규칙(11-02 OCR 필터 `ocr_filter.py`에서 옮김, 엔진 없음) | **localization_design** (`cv_test/i1/text_check.py`) |
| `vision/station_pkg.py` | 역사 패키지 읽기 + ENU ↔ aligned·나침반 ↔ th 변환 | **localization_design** (`cv_test/pf/station_pkg.py`) |
| `vision/inputs/dongchun_station.*` | 역사 패키지: 랜드마크 101면(11-06, 그 전 94면은 git 기록)·가시성 표·`computed`·층 높이·좌표 변환·에스컬레이터 끝(그래프 키) | **localization_design** (`export_station.py` 결과) |
| `vision/inputs/i1/` | 관측 JSONL(필터 없음·시간 필터), 정답 궤적·기준점, 임시 PDR 걸음, 시뮬 PF 옛 기준(`baseline_summary.json`, b852501)·새 기준(`baseline_newvis.json`, 가시성 다시 계산 뒤)·① (`sim_pf_inav_pdr_newvis.json`), EasyOCR 결정 파일(`text_check_easyocr.jsonl` 원래 규칙, `_all` 판의 모든 프레임 읽음, `_sure` 비교용) | **localization_design** (`cv_test/out/i1/`) |
| `vision/inputs/i1/rec/` | 앱 녹화 `motion.csv`·`baro.csv`·`meta.json`·`marks.jsonl` (영상 174 MB는 안 옮김) | 1차 답사 녹화 원본 |
| `vision/cv_module/` | `cvmod`(검출 → 방위각·앙각 관측, 표준 라이브러리만) + `config/default.json`(검출 문턱값). 앱 검출을 관측으로 바꿀 때 씀 | **localization_design** (`cv_module/`) |
| `vision/tools/session_to_replay.py` | 앱 기록 폴더(`detections.jsonl`·`frames.jsonl`·`meta.json`) → `cvmod` 재생 입력 | **ocr_test_app** (`tools/`) |
| `vision/pdr_simple.py` | 임시 PDR(정답 기준점 사이 걸음 시각 보간에 씀) | **localization_design** (`cv_test/i1/`) |
| `vision/inputs/graph/` | 길안내용 경로 그래프: `graph.json`(기본, 시각장애인용), `graph_general.json`(일반 보행 덧붙임 레이어, 테스트·디버깅용), 형식 파일 `station_graph.schema.json`·`station_graph_layer.schema.json`. 앱용 표 `destinations.json`(목적지 묶음을 노드로 풀어 둔 것, 도면 없는 층의 지점에는 도면 있는 층의 가장 가까운 노드 `plan_node`)·`gates.json`(도면 개찰구 몸체 다각형과 줄·칸. 그래프 개찰구 노드가 없는 줄은 도면에만 있는 모양이라 넣지 않음). 아래 [경로 그래프 레이어](#경로-그래프-레이어) | **capstone_graph** (`data/`) |
| `vision/MANIFEST.json` | 위 파일마다 원본 저장소·경로·출처 커밋·sha256 | 복사 도구가 만듦 |
| `vision/inputs/plan/` | 화면용 평면도 그림 `plan_B01.png`·`plan_B02.png`(글자 없음, 벽·기둥·점자블록·개찰구, 20 px/m)와 픽셀 ↔ 좌표 변환 `plan_transform.json` | **localization_design** (`landmark/out/survey/`, 답사 앱에 넣은 것과 같은 파일) |
| `integration/` | 앱 녹화 → `Session`, indoornav PDR 걸음 추출, 방 뺀 보행 마스크, `VisionFilter` 어댑터, 재생기·실시간 재생기·실행기·검산·비교표·그림 | **이 브랜치** |
| `integration/recordings/<이름>.json` | 녹화별 설정: 앱 세션 폴더 이름(`session`), 이 PC 녹화 경로(`rec`), 입력·출력 폴더, 정답 기준점, 채점 제외 구간 | **이 브랜치** |
| `integration/maps/` | 방 뺀 보행 마스크 사본·`map_meta_noroom.json`(재생 기본), 정식 도면에서 만든 보행 마스크 `B0x_walk_plan.png`·`map_meta_plan.json`(앱용, `plan_mask.py`), 그림 바탕 도면 선 `plan_lines.npz`(벽·점자블록, 출처 `plan_lines.json`) | 마스크는 이 브랜치(`walk_mask.py`), 도면 선은 **capstone_graph** `data/dongchun/plan.gpkg`에서 뽑음 |
| `integration/requirements.txt` | 가상환경 패키지(확인한 버전) + indoornav | **이 브랜치** |
| `integration/out/` | 결과(대부분 저장소에 안 넣음). 2차 녹화는 `integration/out/rec/<이름>/inputs·out` | 실행하면 생김 |
| `docs/vision_integration/` | 이 설명, 작업 기록, main 수정 건의 | **이 브랜치** |

**`vision/` 안 파일은 여기서 고치지 않는다.** localization_design에서 고치고 `cv_test/sync_to_branch.py`로 복사한다(옛 이름에서 9f8b8be로 바꿈, 상태만 볼 때 `--check`. 한 방향, 원본에 커밋 안 된 변경이 있거나 브랜치 사본이 손으로 바뀌었으면 멈춤). ocr_test_app 원본(`session_to_replay.py`)과 capstone_graph 원본(경로 그래프)도 같은 도구로 복사하고, `.py` 안의 원본 PC 경로(기본값 줄)는 복사할 때 지운다. 저장소 `.gitignore`가 `data/`를 무시해서 데이터 폴더 이름은 `inputs/`.

검출(YOLO)·OCR은 torch가 필요해 localization_design에서 돌리고, 이 브랜치는 결과 JSON만 읽는다. 앱이 저장한 검출(`detections.jsonl`)을 관측으로 바꾸는 부분은 사본(`vision/cv_module`, `vision/tools/session_to_replay.py`)이 있어 이 브랜치 가상환경만으로 된다. 영상으로 PC 검출을 할 때만 torch·ultralytics·cv2 있는 파이썬과 모델 파일이 필요하다(아래).

## 경로 그래프 레이어

앱이 역사 패키지(PF용)와 같은 묶음으로 받는 길안내용 그래프. 역사 패키지 안에는 그래프가 없다(에스컬레이터 끝 좌표만 그래프 키로 옮겨 씀).

- **기본(`graph.json`)**: 시각장애인용(점자블록·엘리베이터 우선). 혼자 완결된 그래프.
- **일반 레이어(`graph_general.json`)**: 기본 위에 덧붙이는 노드(`GN…`)·간선(`GE…`)만 담음. 혼자서는 쓰지 않는다.
- **모드**: 시각장애인 모드 = 기본만, 일반 모드 = 기본 + 일반 레이어. 레이어를 고르지 않으면 기본만 쓴다.
- **합치는 법**: 레이어 간선은 기본 노드를 ID로 가리킨다. 레이어의 `metadata.base.sha256`이 받은 `graph.json` 해시와 같으면 ID 그대로, 다르면(기본을 다시 만듦) `anchors`의 `key` 또는 층·좌표로 기본 노드를 다시 찾는다. 못 찾으면 레이어를 다시 만들어야 함.
- **`along` 조각**: 기본 간선 중간에 일반 노드를 두려고 그 간선을 나눈 조각. 기본 간선도 그대로 남는다. 위치를 간선에 붙일 때는 조각에 붙이고, 기본 간선이 막히면 조각도 막는다.
- 자세한 필드 설명과 참고 구현: capstone_graph `docs/dongchun-editing.md`, `src/capstone_graph/graph_loader.py`(`load_station_graph(..., layers=[...])`).

## integration/ 파일

| 파일 | 하는 일 |
|---|---|
| `paths.py` | 경로, `sys.path`(`indoornav`, `vision/`), 녹화 시작 시각 `rec_t0()` |
| `app_session.py` | `motion.csv` → `Session` (100 Hz, `acc_user = a·9.80665`, `gravity = g·9.80665`, `gyro = w`, 기압 kPa) |
| `pdr_steps.py` | `run_pdr`(`pdr_tuned_v4.toml`) → `out/i1/steps_indoornav.csv`(t·보폭·방향·**moving**·indoornav 부호 dheading) + 샘플별 방향 `heading_indoornav.npz` |
| `walk_mask.py` | 기존 보행 마스크에서 방·구조물 안쪽을 뺀 사본 `maps/B0x_walk_noroom.png` + 그것을 가리키는 `maps/map_meta_noroom.json` |
| `vision_filter.py` | `VisionFilter`: indoornav `ParticleFilter` + 비전 관측 모델 (아래) |
| `replay.py` | 이벤트 병합 재생, 기압 탑승·층 바뀜, PDR만 궤적, 지표 |
| `run_i1.py` | 조건 × seed 실행 → `out/i1/<이름>/summary.json`. 옵션 `--adopted`, `--bearing-fix`, `--track-limit`, `--text-check`, `--unread`, `--cls-th` |
| `check_inputs.py` | 7절 1·2단계 검산 → `out/i1/check_inputs.json` |
| `compare.py` | 시뮬 PF 기준과 나란히 비교표 → `out/i1/compare.md` |
| `compare_features.py` | 비전 추가 기능 켬/끔 비교표 → `out/i1/compare_*.md` |
| `sweep_obs.py` | 관측 설정 훑기 → `out/i1/sweep_*.md` |
| `diag_k_end.py` | 끝 k·끝 오차 진단(벽·비전·걸음·k를 하나씩 뺌) → `out/i1/diag_k_end.md` |
| `vision_options.json` | 채택한 비전 설정 (`run_i1.py --adopted`) |
| `fig_i1.py` | 시간별 오차·경로 그림 → `out/i1/<이름>/fig_error.png`·`fig_map.png` |
| `analyze_recording.py` | 재생 결과 추가 집계: 지도 방위 일치·PF 후보 일치, 판별 잔차, PDR 거리 비율, 크게 틀린 구간. `--station`으로 옛 지도 집계, `--json-out`으로 옛 결과 폴더를 안 건드리고 저장 |
| `fig_plan.py` | 정식 도면 위 경로 그림 → `<out>/<이름>/fig_plan.png`. 도면 선은 `maps/plan_lines.npz`(geopandas 필요 없음, `--plan <gpkg>`를 주면 원본 도면을 geopandas로 읽음) |
| `fig_plan_rt.py` | 실시간 권장 조합(`rec`) 경로 그림: `data`(PF 비전 끔을 `rec`로 돌림·PDR만은 실시간 걸음) → `draw` → `<out>/rt_fig/fig_plan_rec.png`, 지금 방식과 나란히 `fig_plan_compare.png` |
| `plan_lines.py`·`make_plan_lines.py` | 도면 선 읽기 / 원본 도면(gpkg)에서 한 번 뽑기(geopandas 있는 파이썬, 도면이 바뀔 때만) |
| `plan_mask.py` | 정식 도면에서 보행 마스크를 만듦(보행칸 0.5 m + 벽 단면선 + 개찰구 줄 엶) → `maps/B0x_walk_plan.png`·`map_meta_plan.json`. `--thick`·`--grow`·`--out`으로 비교용 변형. 아래 [정식 도면에서 만든 보행 마스크](#정식-도면에서-만든-보행-마스크) |
| `exp_plan_mask.py` | 지금 마스크 ↔ 정식 도면 마스크 비교 재생(기본 설정은 안 바꿈). `--make`로 비교용 마스크, 기록마다 실행, `--report`로 `out/plan_mask.md`·`.json` |
| `export_app_data.py` | 역 자료를 앱(Swift)이 읽는 형식으로 내보냄 → `out/app_data/<역>/`(저장소에 안 넣음). `--check`는 있는 묶음이 지금 입력과 같은지만 봄. 아래 [앱용 역 자료 묶음](#앱용-역-자료-묶음) |
| `run_recording.py` | 녹화 하나(또는 2차 8개 `all`)를 관측 → PDR → 정답 → 기본 재생 → 실시간 재생 → 그림까지 (아래) |
| `truth_build.py`·`truth_helper.py` | 설정의 기준점 → 정답 궤적 `truth.csv`·확인 그림 / 기준점 후보·영상 프레임 모음(영상 프레임은 cv2 있는 파이썬 `I1_VISION_PY`) |
| `rt_pdr.py`·`rt_replay.py`·`run_rt.py` | 실시간 조건 재생: indoornav PDR을 감싼 지연 판정, 기압 수신 시각 판정 / 받은 시각 순 PF 투입·지연 지표 / 단계별 실행(`--stages off,rec`, `--check-ref exp_d58_r20`) |
| `exp_*.py` | 실험(설정 하나만 바꾼 재생). `exp_cause.py`(원인 분리, 기본 재생 `exp_d58_r20`도 이것), `exp_far*`(먼 관측), `exp_ocr*`(Apple Vision 글자 확인), `exp_overconf*`(과신), `exp_main_recheck*`·`exp_map_data*`(main 건의 재확인). 기본 설정은 안 바꿈 |
| `exp_cause.py`·`exp_report.py`·`exp_signs.py` | 원인 분리 실험: 설정 하나만 바꾼 재생, k·구간·과신 집계, 판별 시선 교차. 기본 설정은 안 바꿈 |
| `provenance.py` | 결과에 남기는 입력 기록(입력 파일·소스 해시), 옛 파일 재사용 검사 |
| `compare_runs.py` | 두 조건의 같은 seed 짝 차이 표, 구간별 악화 표시 |
| `regress_replay.py` | 기본 재생을 다시 돌려 저장된 기록의 모든 필드(좌표·층·시각 포함)가 비트까지 같은지 확인. 저장 경로가 기준과 같으면 멈춤 |
| `check_tf_app.py` | 같은 입력에서 PC 시간 필터와 앱 규칙의 통과 집합 차이. 앱 설정이 없는 기록은 '대체 설정'으로 표시 |
| `run_negative.py` | 음성 대조 재생(random·all·shift ±5 s), 결과는 `compare_runs.py`로 비교 |
| `pool_guard.py` | Pool 작업자 초기화 오류를 호출자에게 전달(명령이 안 끝나는 일을 막음) |

## 실행 방법

환경: 저장소 `uv` 환경 그대로(`uv sync --frozen`). `pyproject.toml`에 Pillow가 빠져 있어(matplotlib 의존성으로 같이 깔림, 없으면 `uv pip install pillow`) `pyproject.toml`은 고치지 않음.
uv 없이 만들 때(결과를 비트까지 맞추려면 이 버전): `python -m venv .venv` (Python 3.12) 뒤 `.venv/Scripts/python -m pip install -r integration/requirements.txt`.

```bash
uv sync --frozen
```

```bash
.venv/Scripts/python integration/walk_mask.py
```

```bash
.venv/Scripts/python integration/pdr_steps.py
```

```bash
.venv/Scripts/python integration/check_inputs.py
```

```bash
.venv/Scripts/python integration/run_i1.py --steps indoornav --name inav_pdr_pf
```

```bash
.venv/Scripts/python integration/run_i1.py --steps simple --name simple_pdr_inav_pf
```

```bash
.venv/Scripts/python integration/run_i1.py --steps indoornav --adopted --name final_adopted
```

```bash
.venv/Scripts/python integration/compare.py
```

민감도(5단계)는 `--sigma-h0 1.5 --name <이름>_h15`, 처리 시간은 `--n 1000`. 10개 프로세스로 한 실행(100회) 약 40초. ①(시뮬 PF + indoornav PDR)은 localization_design에서 `cv_test/i1/check_baseline.py --steps <브랜치>/integration/out/i1/steps_indoornav.csv --first-dh --name sim_pf_inav_pdr_newvis`로 돌린 결과를 복사 도구로 가져옴. 기능을 하나씩 켜려면 `--bearing-fix theta`, `--track-limit 1`, `--text-check text_check_easyocr.jsonl`(`--unread drop`이면 못 읽은 관측 뺌), `--cls-th exit=0.3,...`. 그림: `integration/fig_i1.py --name <이름>`.

## 평가 안전장치

위치 결과를 바꾸지 않고 비교를 믿을 수 있게 하는 도구. 시험: `.venv/Scripts/python -m pytest tests/vision_integration`.

```
.venv/Scripts/python integration/compare_runs.py --rec s2_a_ex3_ex6 --a exp_base --b exp_d58_r20   # 같은 seed 짝 차이, 구간별 악화
.venv/Scripts/python integration/compare_runs.py --rec all --a <기준> --b <비교> --range 38-53       # 기록 9개 + 기록별 요약
.venv/Scripts/python integration/regress_replay.py --ref exp_d58_r20                                 # 기본 재생이 그대로인지
.venv/Scripts/python integration/check_tf_app.py                                                     # PC·앱 시간 필터 차이
.venv/Scripts/python integration/run_negative.py --rec s2_a_ex6_ex3 --cond random shift_m5 shift_p5   # 음성 대조 → neg_<조건>/
.venv/Scripts/python integration/compare_runs.py --rec s2_a_ex6_ex3 --a exp_d58_r20 --b neg_random    # 그 결과 비교
```

- **비교 기준**: 전체 p90 하나로 판단하지 않는다. 같은 seed 짝 차이와 방향 일치 수, 구간별 악화를 함께 본다. seed 범위는 같은 입력의 PF 표본 잡음일 뿐 새 보행·입력 변화의 불확실성이 아니다. 조건을 고른 기록과 평가 기록은 나눈다.
- **입력 기록**: `run_i1.py`·`exp_cause.py`·`sweep_obs.py`·`diag_k_end.py`·`run_negative.py`의 결과에 `provenance`(걸음·방향·관측·정답·기압·시간 원점 `meta.json`·`frames.jsonl`과 `rec_t0` 값·역사 패키지·`vision_options.json`·녹화 설정·PF/PDR 설정·마스크 해시, 소스 파일별 해시와 합친 값)를 남긴다. `compare_runs.py`가 두 결과의 설정 차이(한쪽에만 있는 칸, 실제 PF 값 포함)와 입력 차이를 같이 보여 준다.
- **음성 대조**: 좋아 보이는 변경은 `run_negative.py`의 random·shift와 같이 본다. 그쪽도 비슷하게 좋아지면 그 변경의 근거는 약하다. 전체 p90뿐 아니라 구간별로 본다.
- **지표 이름**: `지도 방위 일치`(종류·20 m·10°)와 `PF 후보 일치`(가시성·20 m·화각·앙각까지 + 10°)는 다른 값이다. 둘 다 정답과 카메라 기준값에 의존하는 진단값이고 PF 입력으로 안 들어간다. `match`는 옛 이름으로 `map_bearing_match`와 같다.
- **글자 확인 연결률**: 결정이 없는 관측은 그대로 들어가므로 `apply_decisions`가 연결률을 알리고, 이어진 수 ÷ min(관측 수, 결정 수)가 50% 아래면 멈춘다.

## 좌표·각도·시계 약속

| 항목 | 약속 |
|---|---|
| 지도 좌표 aligned | `aligned = Rot(−52.73°) · ENU`, **원점 그대로** (ENU = 동·북 m, 메시 원점 = `landmarks_Dongchun.json`). localization_design `en_to_st()`(역 좌표)는 이동값 T까지 빼므로 쓰지 않음. 회전각은 `maps/dongchun/map_meta.json` 값과 같은지 `walk_mask.py`가 확인 |
| 관측 모델 좌표 | ENU 그대로(랜드마크·판 끝점·가시성 표). 어댑터가 파티클을 aligned → ENU로 바꿔 넣음 |
| 방향 | indoornav th = aligned 수학각(반시계 +, rad). 나침반(북 0°, 시계 +) = `wrap360(37.27° − deg(th))` |
| 걸음 회전 | indoornav dheading(반시계 +, rad) = `−rad(나침반 부호 dheading_deg)`. indoornav PDR 방향 기준 = 녹화 시작(센서 첫 시각) → **첫 걸음 dheading도 넣음** |
| 관측 방위 bearing | 카메라 정면에서 오른쪽 + [도], 그대로 |
| 시계 | 모두 앱 호스트 시계. 재생 시각 t = 호스트 시각 − 첫 영상 프레임(`meta.json recording.first_t`). `Session.t` = 호스트 시각 − 센서 첫 시각(영상보다 0.024 s 늦음) |
| 층 | B1 = −1, B2 = −2 (지도 이름 `B01`·`B02`) |
| 에스컬레이터 끝 | 그래프 좌표(`capstone_graph` 키 `ESC_P1_B1` 등 → 역사 패키지). `maps/dongchun/map_meta.json` 아래 끝은 약 1.5 m 짧아 안 씀 |

## VisionFilter (indoornav PF 어댑터)

- **생성·초기화**: `pf_home_v3.toml` 읽고 N만 바꿈(3000). `pf.init()` 뒤 xy·th를 덮어씀 — 출발점 N(정답 t=0, 2² I)를 방 뺀 보행 마스크(`GridMap.free`)로 거절 샘플링, 방향 σ 20°, k는 `init()`의 U(0.8, 1.2). indoornav `init()`은 벽 너머 표본을 출발점 한 점에 모아 σ 2 m에서 65%가 한 점에 몰림 → 0%
- **predict**: indoornav `pf.predict(length, dheading, moving)` 그대로
- **update**: 파티클 aligned → ENU·나침반(걸음 사이 회전 보정을 켜면 이번 비교에만 방향 − Δ, 인자 `dpsi_deg`) → `obs_likelihood` → `pf.logw += log(w)` → `pf._normalize()` → `pf._maybe_resample()`. indoornav PF에 외부 점수를 넣는 공개 함수가 없어 비공개 메서드를 부름(main 커밋 고정 9040d70, 공개 `reweight` 요청은 연결안 4절)
- **탑승 중**(기압으로 찾은 타는 구간): 걸음 위치 안 넣음, 관측 안 넣음, 회전은 모아 두었다가 하차(층 바뀜) 뒤 `predict(0, Σdh)` 한 번
- **층 바뀜**: 파티클마다 가장 가까운 에스컬레이터 위 끝 → 아래 끝 + N(0, 1²)을 **목적지 보행 마스크 안에서** 다시 뽑음, 가중치 그대로(균등화 안 함), `pf.map`을 B02 `GridMap`으로 교체
- **estimate**: indoornav `estimate()`(평균·th·k·unc95·ESS·pdr_suspect) + RMS 불확실도(시뮬 기준과 같은 정의) + 상위 두 모드(위치·질량)
- 재주입 없음

## 방 뺀 보행 마스크

기존 보행 마스크는 평면도 이미지 색으로 만든 것이라 B01 직원실·매표실, B02 계단·에스컬레이터 구조물 안쪽, 그림 속 글자·지시선까지 걸을 수 있는 곳으로 들어가 있다. 규칙: **기존 보행 마스크 칸 중 역사 패키지 보행칸(가시성 표 `walk`, 0.5 m 격자)에서 0.5 m 넘게 떨어진 칸은 뺌.** B01 2758 → 2316 m², B02 2337 → 1995 m². 정답 궤적은 두 층 모두 100% 안. indoornav `GridMap` 클래스로 그대로 읽음.

## 정식 도면에서 만든 보행 마스크

방 뺀 보행 마스크는 글자·지시선이 들어간 평면도 그림에서 색으로 뽑은 것이라, 방을 빼도 안쪽에 글자·표시 흔적이 남고(B01 22.7 m²) 기둥은 테두리 점만 있다. `plan_mask.py`는 그림을 쓰지 않고 이 브랜치의 도면 자료만으로 마스크를 만든다. 픽셀 격자(축척·원점·크기)는 `maps/dongchun/map_meta.json`과 같아서 `GridMap.load`로 그대로 읽는다.

재료
- 역사 패키지 보행칸 `walk`: 메시 바닥에서 만든 0.5 m 격자. 방·기둥·계단 구조물 자리는 이미 빠져 있고, 경계는 벽 선보다 반 칸쯤 모자라거나 넘친다
- 도면 벽 단면선 `maps/plan_lines.npz`: 바닥 + 1.2 m 높이에서 자른 선(벽·기둥·키 큰 구조물). 1.2 m보다 낮은 개찰구 울타리·벤치·낮은 난간은 없다
- 개찰구 줄 `vision/inputs/graph/gates.json`, 경로 그래프 노드

만드는 순서(층마다)
1. 보행칸을 픽셀에 칠하고 0.5 m 넓힘 — 벽 선까지의 틈을 메우려는 것. 넘친 부분은 다음 단계에서 잘린다
2. 벽 선에서 띠 두께(기본 0.3 m)의 절반 안을 막음. PF는 한 걸음을 0.08 m 간격으로 검사하므로(`GridMap.crosses`), 띠가 0.2 m는 되어야 B02 픽셀(7.4 cm)에서도 건너뛰지 않는다
3. 개찰구 줄(첫 몸체 바깥 끝 ~ 마지막 몸체 바깥 끝)을 통째로 엶. 개찰구는 막지 않기로 함(칸 폭 0.4~1.0 m라 위치 오차가 있으면 파티클이 못 지나감, 11-13)
4. 경로 그래프 노드와 이어진 영역만 남김. 벽 밖으로 넘친 띠와 승강장 안전문 바깥 띠가 떨어져 나가고, 기둥은 원 모양으로 막힌다

시험: `tests/vision_integration/test_plan_mask.py` (규칙 하나씩 + 실제 자료에서 벽 선을 건너는 걸음을 `GridMap.crosses`가 모두 잡는지, 개찰구 줄이 열려 있는지, 그래프 간선이 막히지 않는지).

**비교 결과(11-13, `exp_plan_mask.py`. 기록별 표는 `--report`가 `integration/out/plan_mask.md`에 만듦, 저장소에는 안 넣음)**: 기록 9개, seed 10개, 지금 마스크 ↔ 벽 띠 0.2 m(`plan_g20`)·0.3 m(`plan_g30`). 판단 기준은 재생 전에 정함.

| | plan_g20 | plan_g30 |
|---|---|---|
| 넓이 B01 / B02 (지금 2316 / 1995 m²) | 2315 / 1888 m² | 2273 / 1827 m² |
| 시간 필터·기본: 9개 평균 오차 중앙 / 90% 차이 | −0.03 / +0.07 m | −0.03 / −0.04 m |
| 90%가 크게 나빠진 기록 | 없음 | 없음 |
| 자신 있게 틀린 seed 합 (지금 60) | 63 | 60 |
| 정답 궤적이 마스크 밖을 더 지나는 기록 | `s2_b_ex2_31` 1.8% | `s2_b_ex2_31` 1.8% |
| 비전 끔(참고): 9개 평균 오차 중앙 / 90% 차이 | −0.18 / −0.26 m | −0.20 / −0.33 m |
| 판단 | 기준 3·4번 못 지킴 | 기준 4번 못 지킴 |

- 4번에 걸린 곳은 `s2_b_ex2_31`의 92.6~95.8 s 한 군데: 정답 궤적(기준점 사이를 곧게 이은 선)이 기둥 가장자리를 최대 0.22~0.28 m 스쳐 지나간다. 지금 마스크는 기둥이 테두리 점만 있어서 걸리지 않았다
- **벽 띠 0.3 m 채택(11-13)**: 4번은 마스크가 틀린 것이 아니라 정답 선이 실제 기둥을 지나가서 걸린 것으로 보고 채택함. `maps/B0x_walk_plan.png`·`map_meta_plan.json`이 그 마스크이고 앱용 묶음에도 이것이 들어감
- **재생의 기본 마스크는 아직 방 뺀 사본**(`paths.MASK_META` 그대로). 바꾸면 기준 결과(`exp_d58_r20`)와 그것에 맞춘 회귀 시험·표가 모두 달라지므로 따로 정해서 한 번에 바꿈

## 앱용 역 자료 묶음

앱은 numpy(`npz`)·GeoPackage를 못 읽으므로, 이 브랜치의 역 자료를 JSON과 맨 바이트 파일로 바꿔 한 폴더에 모은다. 입력은 이 브랜치 안 파일뿐이고 결과는 저장소에 넣지 않는다(다시 만들면 같은 바이트, 약 15 MB). 개찰구 몸체는 보행 마스크에서 막지 않는다(칸 폭 0.4~1.0 m라 위치 오차가 있으면 파티클이 못 지나감, 11-13 결정).

```
.venv/Scripts/python integration/export_app_data.py            # → integration/out/app_data/dongchun/
.venv/Scripts/python integration/export_app_data.py --check    # 있는 묶음이 지금 입력과 같은지만
```

| 파일 | 내용 | 원본 |
|---|---|---|
| `manifest.json` | 형식·좌표 약속·파일 목록(종류·모양·sha256)·입력 해시. 앱은 이것부터 읽음 | |
| `station.json` | 층, 카메라 높이, 좌표 변환(회전각, 층별 `R`·`T`·`x0`·`y0`·`cell`), 에스컬레이터 끝, 랜드마크 | `vision/inputs/dongchun_station.*` |
| `<층>_walk.u8`·`_walk_pf.u8`·`_computed.u8` | 0.5 m 격자 (H, W) uint8 0/1 | 같음 |
| `<층>_dist.f16` | 가시성 표 (랜드마크 수, H, W) float16. `nan` = 안 보임(`computed` 1) 또는 모름(`computed` 0) | 같음 |
| `<층>_mask.u8` | PF 벽 판정용 보행 마스크 (H, W) uint8 0/1. 축척·원점은 `manifest.json`의 `masks` | `integration/maps/*_walk_plan.png` (정식 도면에서 만든 마스크) |
| `<층>_walls_xy.f64`·`_off.i32`, `<층>_tactile_…` | 도면 벽 선·점자블록: 이어 붙인 좌표 (N, 2)와 도형 시작 위치(도형 수 + 1) | `integration/maps/plan_lines.npz` |
| `graph.json`·`graph_general.json`·`destinations.json`·`gates.json` | 경로 그래프·목적지 묶음·개찰구 몸체 (그대로 복사, 줄 끝 LF) | `vision/inputs/graph/` |
| `plan_<층>.png` | 화면에 띄우는 평면도 그림 (그대로 복사). 픽셀 ↔ 동·북 변환은 `manifest.json`의 `ui_plan` (`en_to_px`·`px_to_en`, 2×3 행렬) | `vision/inputs/plan/` |

- **바이트 약속**: 숫자는 모두 little-endian, 배열은 행 우선. (H, W)에서 칸 (i, j)는 `i*W + j`번째, (L, H, W)에서 (l, i, j)는 `(l*H + i)*W + j`번째 값
- **좌표 약속**은 `manifest.json`의 `coords`에 식으로 적음(동·북 ↔ 역 좌표 격자 ↔ 지도 aligned ↔ 마스크 픽셀, 나침반 ↔ 지도 각). 위 [좌표·각도·시계 약속](#좌표각도시계-약속)과 같은 내용
- **읽기 참고 구현**: `export_app_data.load_app_data(폴더)` — `station_pkg.load_station`과 같은 필드를 돌려줌. 내보낸 뒤 자동으로 다시 읽어 원본(`npz`·`png`)과 값이 같은지 확인하고, 다르면 실패로 끝남
- **시험**: `tests/vision_integration/test_app_data.py` (numpy 없이 `struct`로 읽어 바이트 약속 확인, 마스크 찾기가 `GridMap.free`와 같은지, 목적지 노드가 보행 격자 위인지 등)
- PF 벽 판정에는 마스크를 씀(정식 도면에서 만든 것: 벽·기둥 막음, 개찰구 엶). 0.5 m 격자는 가시성 표 칸 찾기용

## 비전 설정

병합 중 고정값: 검출 문턱 0.25(모든 종류) + 시간 필터(5프레임 중 3번) + 화면 끝 판 끝점 비교 + 11-03 랜드마크 지도(94면, 11-06부터 2차 답사 반영 101면).

1차 병합 뒤 채택(근거는 [작업기록.md](작업기록.md) 3차): 가시성 표 다시 계산(`computed` = 보행칸 1칸 넓힘 ∪ 기존 보행 마스크), **걸음 사이 회전 보정**(비교할 때만 파티클 방향에 Δ를 더함, PF 상태는 안 바뀜), **track별 반복 제한 1 s**. 켜지 않음: 글자 확인(OCR, 규칙은 11-02 원래 판정 코드, 나아지지 않음, `--text-check`로 켤 수 있고 Apple Vision은 같은 형식 결정 파일이면 됨), 종류별 문턱값(후보 B를 2차 답사 평가용 녹화로 확정할 때까지). 관측 설정은 기본·중간을 나란히 비교, 잠정 기본.

**결정 파일 형식**: `vision/text_check.py` 맨 위. 검출·OCR은 localization_design에서 돌리고 이 브랜치는 결과만 읽음.

## 새 녹화 돌리기 (`integration/run_recording.py`)

앱 기록 zip이나 폴더 하나로 파일 검사 → 관측 → PDR → 정답 → PDR만·PF·PF + 비전 → 경로 그림·오차 표까지 돈다. PF 동작·관측 모델·비전 설정은 안 바꾼다(`vision_options.json` 그대로). 녹화별 값은 `integration/recordings/<이름>.json` 하나에 있다(녹화·입력·출력 경로, 기준점, 채점 제외 구간, 끝 오차 시작). 고르는 법: 환경 변수 `I1_RECORDING`(없으면 1차 녹화 `route2_1030`).

```
.venv/Scripts/python integration/run_recording.py <zip 또는 폴더> --name <이름>
```

1. 처음 돌리면 **정답 단계에서 멈춘다.** `integration/out/rec/<이름>/out/`에 시각 후보와 그림이 나온다.
   - `waypoints_draft.json`: 후보 시각(자이로 회전, 기압 층 이동, 2 s 넘는 멈춤, 앱 표시, 출발·끝)
   - `candidates.png`(회전·기압·걸음), `sheet_candidates.jpg`(후보 번호별 영상 프레임, 영상 있을 때), `plan_B01.png`·`plan_B02.png`(역 좌표 도면, 개찰구·에스컬레이터 등 node 값)
2. 쓸 후보를 골라 위치를 채워 `recordings/<이름>.json`의 `waypoints`에 넣는다. 항목: `name`, `t`, `floor`(B1 = -1, B2 = -2), `st`[x, y] 또는 `node`(예 `GATE_3`), `ref`(무엇을 보고 정했나). 에스컬레이터 탄 기준점에 `"esc": true`.
   - **`ref`에 판을 쓰지 않는다**(기둥·벽 모서리·개찰구·에스컬레이터 끝·문). 판 위치가 틀렸을 수 있어 채점이 순환된다. 판이면 경고가 나온다(옛 기록의 `basis` 칸도 검사).
   - 기둥을 지나는 순간 찾기: `I1_RECORDING=<이름> .venv/Scripts/python integration/truth_helper.py sheet --from 30 --to 40 --every 0.5`
3. 같은 명령을 다시 돌린다(`run_recording.py <이름>`도 됨). 채점 제외(층 이동 + 2 s)·끝 오차 시작(마지막 앞 기준점)은 기준점에서 채운다. 결과: `out/replay/report.md`, `fig_paths.png`, `fig_error.png`. 정답 확인: `out/truth_check.md`(걸음과 안 맞는 구간 표시)·`truth_check.png`.

관측은 `detections.jsonl`이 있으면 앱 검출(사본 `vision/tools/session_to_replay.py`(원본 `ocr_test_app/tools/session_to_replay.py`) → `vision/cv_module`에서 `cvmod.run --detector replay`, 이 가상환경만), 없고 영상이 있으면 PC 검출(C2 추가학습 후, 5 fps), 둘 다 없으면 이유를 적고 멈춘다. PC 검출에만 필요: cv2·torch 있는 파이썬(`I1_VISION_PY`, 기본 시스템 Python 3.12)과 모델(`I1_MODEL`, 기본 옆 폴더 `localization_design`의 C2 모델). 관측·PDR을 다시 만들려면 `--force`. 있는 파일을 그대로 쓸 때는 `inputs/input_record.json`에 적힌 입력(모델·설정·녹화 파일·코드 해시)과 지금 입력을 비교해 다르거나 기록이 없으면 경고하고 `report.md`에도 남긴다. 시간 필터는 매번 다시 계산하므로 해당 없음. 같은 걸음을 재생한 기록은 `--same-walk route2_1030`으로 기준점을 가져온다. 풀린 기록·결과(`out/rec/`)는 저장소에 안 넣는다.

**2차 답사 기록**(`recordings/s2_*.json`): 원본 폴더는 저장소 밖이라 `integration/out/rec/<이름>/rec`에 폴더 연결(Windows junction)로 둔다(`out/rec/`는 저장소에 안 넣음). 그 뒤 `run_recording.py <이름>`. 기준점이 이미 들어 있어 비교까지 돈다. 연결 대신 `--data-root <녹화 폴더들의 상위 폴더>`(또는 환경 변수 `I1_DATA_ROOT`)를 주면 설정의 `session`(앱 세션 폴더 이름)을 그 아래에서 찾는다(아래).

11-08부터 6단계 이후는 기본 재생 `exp_d58_r20`(= `regress_replay.py` 기준) → 실시간 재생 `rt_compare`(`off,rec`, off가 기본 재생과 비트까지 같은지 확인) → 그림(`fig_plan.py`, `fig_plan_rt.py`)이다. 결과가 있고 입력(관측·정답·걸음)보다 새것이며 필요한 seed·단계가 다 있으면 건너뛴다. 결과는 있는데 입력보다 오래됐거나 모자라면 덮어쓰지 않고 멈춘다(실시간 재생 폴더에는 비교용 단계가 더 들어 있을 수 있어서) → 다시 만들려면 `--force-replay`(그 폴더를 덮어씀). 실시간 재생 off 단계가 기본 재생과 비트까지 같지 않으면 멈춘다. 예전 6단계(`run_i1 --adopted` 재생·`report.md`)는 `--old-replay`.

## 다른 PC에서 2차 녹화로 처음부터

브랜치만 받은 PC에서 2차 보행 8개를 관측부터 그림까지 다시 만든다. 옆 저장소(localization_design·ocr_test_app·capstone_graph)는 필요 없다. 1차 녹화(`route2_1030`)는 입력이 저장소 안(`vision/inputs/i1/`)에 있어 그대로 돈다.

준비: 2차 앱 녹화 폴더들(세션 폴더마다 `motion.csv`·`baro.csv`·`meta.json`·`frames.jsonl`·`detections.jsonl`)을 한 상위 폴더 아래 아무 깊이에 둔다. 폴더 이름은 앱 세션 이름 그대로(`recordings/s2_*.json`의 `session`: `OCR_7-1_2출`, `OCR_20261105_2109` 등).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r integration/requirements.txt
```

```bash
.venv/Scripts/python integration/run_recording.py all --data-root <녹화 폴더들의 상위 폴더>
```

기록 하나만: `run_recording.py s2_b_ex2_31 --data-root <상위 폴더>`. 한 기록에 몇 분(10개 프로세스, seed 10개 × 기본 재생 3조건 + 실시간 2단계 + 그림용 재생 1번).

결과 위치(기록마다 `integration/out/rec/<이름>/`):

| 무엇 | 파일 |
|---|---|
| 관측 | `inputs/obs_nofilter.jsonl`·`obs_timefilter.jsonl` (중간 `replay_input.json`·`cv_raw.json`) |
| 걸음·정답 | `inputs/steps_simple.csv`·`truth.csv`, `out/steps_indoornav.csv`·`heading_indoornav.npz`, 정답 확인 `out/truth_check.md`·`.png` |
| 기본 재생 | `out/exp_d58_r20/summary.json`·`runs.npz`, 그림 `fig_plan.png` |
| 실시간 재생 | `out/rt_compare/summary.json`·`runs.npz`(`off_<seed>`·`rec_<seed>`), 걸음 판정 시각 `steps_*.csv` |
| 실시간 그림 | `out/rt_fig/fig_plan_rec.png`, `fig_plan_compare.png`(위 지금 방식, 아래 `rec`), 그린 배열 `rec_paths.npz` |

그다음 도구는 같은 녹화 폴더를 환경 변수로 알려 주고 돌린다(`I1_DATA_ROOT=<상위 폴더>`, 기록 고르기는 `I1_RECORDING=<이름>`).

```bash
I1_DATA_ROOT=<상위 폴더> .venv/Scripts/python integration/regress_replay.py --ref exp_d58_r20 --rec s2_a_71_ex2 s2_a_ex3_ex6 s2_a_ex6_ex3 s2_a_platform s2_b_elev_pillar s2_b_emart_ex5 s2_b_ex2_31 s2_b_pillar_map
```

```bash
I1_DATA_ROOT=<상위 폴더> .venv/Scripts/python -m pytest tests/vision_integration
```

```bash
I1_DATA_ROOT=<상위 폴더> I1_RECORDING=s2_b_ex2_31 .venv/Scripts/python integration/exp_cause.py --name exp_base
```

`compare_runs.py`, `run_rt.py --rec all`, 다른 `exp_*.py`도 같은 방식. 9개 기록을 도는 도구(`--rec all`, `regress_replay.py` 기본)는 1차 녹화 결과(`integration/out/i1/`)도 필요하므로 먼저 `I1_RECORDING=route2_1030`으로 `exp_cause.py --name exp_d58_r20`을 돌리거나 2차 8개만 고른다.

안 되는 경우
- `detections.jsonl`이 없는 녹화: 영상으로 PC 검출을 해야 해서 torch·ultralytics·cv2 있는 파이썬(`I1_VISION_PY`)과 C2 추가학습 후 모델 파일(`I1_MODEL`, 저장소에 없음)이 필요. 없으면 이유를 적고 멈춤
- 설정에 기준점이 없는 새 녹화: 정답 단계에서 멈춤(위 '새 녹화 돌리기' 1~3)
- 기준점 도우미의 영상 프레임 모음(`truth_helper.py sheet`, `sheet_candidates.jpg`)은 cv2 있는 파이썬 필요
- 기준점을 `node`(그래프 키)로 적은 새 녹화: 옆 폴더 `capstone_graph`의 `graph.json`이 있어야 정답을 만듦. 2차 8개 설정은 모두 `st` 좌표라 해당 없음
- 영상 PC 검출은 `video.mov`만 읽음(원본 cvmod 제한, `video.mp4`만 있으면 실패)
- 변환 사본을 직접 부를 때는 `vision/tools/session_to_replay.py <녹화 폴더> --out <파일> --cv-module vision/cv_module`(기본값은 원본 저장소 배치 기준이라 브랜치에서는 `--cv-module`이 필요)
- 도면이 바뀌면 `make_plan_lines.py --plan <plan.gpkg>`를 geopandas 있는 파이썬으로 다시 돌려 `maps/plan_lines.*`를 갱신(도면이 git 저장소 안에 있어야 출처를 남김, 아니면 아무것도 안 쓰고 멈춤)
