# vision-integration 브랜치 — indoornav PDR·PF + 비전 관측 연결

indoornav PDR·파티클 필터(`indoornav/`)에 비전 관측(랜드마크 방위각 관측 모델)을 붙여, 1차 답사 보행 녹화(2번 출구 → 개찰구 3 → 원인재 방면 에스컬레이터 → 3-1 문, 118.6 s)를 오프라인 재생하고 **비전 끔/켬**을 비교하는 브랜치.

- `main`(9040d70)에서 갈라짐. **기존 파일(`indoornav/`, `configs/`, `maps/`, `scripts/`, `tests/`, `pyproject.toml`, `README.md` 등)은 고치지 않고 불러 쓰기만 한다.** 새 폴더 `vision/`, `integration/`, `docs/vision_integration/`만 더함
- 설계 근거: localization_design의 I1 연결안 문서 (11-04). 출발점을 아는 경우(known·drift)만 다룸
- 결과·작업 기록: [작업기록.md](작업기록.md), 비교표 `integration/out/i1/compare.md`

## 폴더 구조와 원본 쪽

| 폴더·파일 | 내용 | 원본 |
|---|---|---|
| `vision/obs_model.py` | 관측 모델 `obs_likelihood` (방위각·가시성·화각·바닥값) | **localization_design** (`cv_test/pf/obs_model.py`) |
| `vision/time_filter.py` | 시간 필터(최근 5프레임 중 3번) + cv_module 결과 읽기 | **localization_design** (`cv_test/i1/time_filter.py`) |
| `vision/edge_faces.py` | 화면 끝 검출 → 판 끝점 비교 | **localization_design** (`cv_test/i1/edge_faces.py`) |
| `vision/bearing_fix.py` | 걸음 사이 회전 Δ 계산 | **localization_design** (`cv_test/i1/bearing_fix.py`) |
| `vision/text_check.py` | 글자 확인 결정 파일 읽기·적용 + 판정 규칙(11-02 OCR 필터 `ocr_filter.py`에서 옮김, 엔진 없음) | **localization_design** (`cv_test/i1/text_check.py`) |
| `vision/station_pkg.py` | 역사 패키지 읽기 + ENU ↔ aligned·나침반 ↔ th 변환 | **localization_design** (`cv_test/pf/station_pkg.py`) |
| `vision/inputs/dongchun_station.*` | 역사 패키지: 랜드마크 94면·가시성 표·`computed`·층 높이·좌표 변환·에스컬레이터 끝(그래프 키) | **localization_design** (`export_station.py` 결과) |
| `vision/inputs/i1/` | 관측 JSONL(필터 없음·시간 필터), 정답 궤적·기준점, 임시 PDR 걸음, 시뮬 PF 옛 기준(`baseline_summary.json`, b852501)·새 기준(`baseline_newvis.json`, 가시성 다시 계산 뒤)·① (`sim_pf_inav_pdr_newvis.json`), EasyOCR 결정 파일 | **localization_design** (`cv_test/out/i1/`) |
| `vision/inputs/i1/rec/` | 앱 녹화 `motion.csv`·`baro.csv`·`meta.json`·`marks.jsonl` (영상 174 MB는 안 옮김) | 1차 답사 녹화 원본 |
| `vision/MANIFEST.json` | 위 파일마다 원본 경로·출처 커밋·sha256 | 복사 도구가 만듦 |
| `integration/` | 앱 녹화 → `Session`, indoornav PDR 걸음 추출, 방 뺀 보행 마스크, `VisionFilter` 어댑터, 재생기·실행기·검산·비교표 | **이 브랜치** |
| `docs/vision_integration/` | 이 설명, 작업 기록 | **이 브랜치** |

**`vision/` 안 파일은 여기서 고치지 않는다.** localization_design에서 고치고 `cv_test/sync_to_branch.py`로 복사한다(한 방향, 원본에 커밋 안 된 변경이 있거나 브랜치 사본이 손으로 바뀌었으면 멈춤). 저장소 `.gitignore`가 `data/`를 무시해서 데이터 폴더 이름은 `inputs/`.

검출(YOLO)·OCR은 torch가 필요해 localization_design에서 돌리고, 이 브랜치는 결과 JSON만 읽는다.

## integration/ 파일

| 파일 | 하는 일 |
|---|---|
| `paths.py` | 경로, `sys.path`(`indoornav`, `vision/`), 녹화 시작 시각 `rec_t0()` |
| `app_session.py` | `motion.csv` → `Session` (100 Hz, `acc_user = a·9.80665`, `gravity = g·9.80665`, `gyro = w`, 기압 kPa) |
| `pdr_steps.py` | `run_pdr`(`pdr_tuned_v4.toml`) → `out/i1/steps_indoornav.csv`(t·보폭·방향·**moving**·indoornav 부호 dheading) + 샘플별 방향 `heading_indoornav.npz` |
| `walk_mask.py` | 기존 보행 마스크에서 방·구조물 안쪽을 뺀 사본 `maps/B0x_walk_noroom.png` + 그것을 가리키는 `maps/map_meta_noroom.json` |
| `vision_filter.py` | `VisionFilter`: indoornav `ParticleFilter` + 비전 관측 모델 (아래) |
| `replay.py` | 이벤트 병합 재생, 기압 탑승·층 바뀜, PDR만 궤적, 지표 |
| `run_i1.py` | 조건 × seed 실행 → `out/i1/<이름>/summary.json` |
| `check_inputs.py` | 7절 1·2단계 검산 → `out/i1/check_inputs.json` |
| `compare.py` | 시뮬 PF 기준과 나란히 비교표 → `out/i1/compare.md` |
| `compare_features.py` | 비전 추가 기능 켬/끔 비교표 → `out/i1/compare_*.md` |
| `sweep_obs.py` | 관측 설정 훑기 → `out/i1/sweep_*.md` |
| `diag_k_end.py` | 끝 k·끝 오차 진단(벽·비전·걸음·k를 하나씩 뺌) → `out/i1/diag_k_end.md` |
| `vision_options.json` | 채택한 비전 설정 (`run_i1.py --adopted`) |
| `fig_i1.py` | 시간별 오차·경로 그림 → `out/i1/<이름>/fig_error.png`·`fig_map.png` |

## 실행 방법

환경: 저장소 `uv` 환경 그대로(`uv sync --frozen`). `pyproject.toml`에 Pillow가 빠져 있어(matplotlib 의존성으로 같이 깔림, 없으면 `uv pip install pillow`) `pyproject.toml`은 고치지 않음.

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

민감도(5단계)는 `--sigma-h0 1.5 --name <이름>_h15`, 처리 시간은 `--n 1000`. 10개 프로세스로 한 실행(100회) 약 40초. ①(시뮬 PF + indoornav PDR)은 localization_design에서 `cv_test/i1/check_baseline.py --steps <브랜치>/integration/out/i1/steps_indoornav.csv --first-dh --name sim_pf_inav_pdr_newvis`로 돌린 결과를 복사 도구로 가져옴. 기능을 하나씩 켜려면 `--bearing-fix theta`, `--track-limit 1`, `--text-check text_check_easyocr.jsonl`, `--cls-th exit=0.3,...`. 그림: `integration/fig_i1.py --name <이름>`.

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
- **update**: 파티클 aligned → ENU·나침반 → `obs_likelihood` → `pf.logw += log(w)` → `pf._normalize()` → `pf._maybe_resample()`. indoornav PF에 외부 점수를 넣는 공개 함수가 없어 비공개 메서드를 부름(main 커밋 고정 9040d70, 공개 `reweight` 요청은 연결안 4절)
- **탑승 중**(기압으로 찾은 타는 구간): 걸음 위치 안 넣음, 관측 안 넣음, 회전은 모아 두었다가 하차(층 바뀜) 뒤 `predict(0, Σdh)` 한 번
- **층 바뀜**: 파티클마다 가장 가까운 에스컬레이터 위 끝 → 아래 끝 + N(0, 1²)을 **목적지 보행 마스크 안에서** 다시 뽑음, 가중치 그대로(균등화 안 함), `pf.map`을 B02 `GridMap`으로 교체
- **estimate**: indoornav `estimate()`(평균·th·k·unc95·ESS·pdr_suspect) + RMS 불확실도(시뮬 기준과 같은 정의) + 상위 두 모드(위치·질량)
- 재주입 없음

## 방 뺀 보행 마스크

기존 보행 마스크는 평면도 이미지 색으로 만든 것이라 B01 직원실·매표실, B02 계단·에스컬레이터 구조물 안쪽, 그림 속 글자·지시선까지 걸을 수 있는 곳으로 들어가 있다. 규칙: **기존 보행 마스크 칸 중 역사 패키지 보행칸(가시성 표 `walk`, 0.5 m 격자)에서 0.5 m 넘게 떨어진 칸은 뺌.** B01 2758 → 2316 m², B02 2337 → 1995 m². 정답 궤적은 두 층 모두 100% 안. indoornav `GridMap` 클래스로 그대로 읽음.

## 비전 설정

병합 중 고정값: 검출 문턱 0.25(모든 종류) + 시간 필터(5프레임 중 3번) + 화면 끝 판 끝점 비교 + 11-03 랜드마크 지도(94면).

1차 병합 뒤 채택(근거는 [작업기록.md](작업기록.md) 3차): 가시성 표 다시 계산(`computed` = 보행칸 1칸 넓힘 ∪ 기존 보행 마스크), **걸음 사이 회전 보정**(비교할 때만 파티클 방향에 Δ를 더함, PF 상태는 안 바뀜), **track별 반복 제한 1 s**. 켜지 않음: 글자 확인(OCR, 나아지지 않음, `--text-check`로 켤 수 있고 Apple Vision은 같은 형식 결정 파일이면 됨), 종류별 문턱값(후보 B를 2차 답사 평가용 녹화로 확정할 때까지). 관측 설정은 기본·중간을 나란히 비교, 잠정 기본.

**결정 파일 형식**: `vision/text_check.py` 맨 위. 검출·OCR은 localization_design에서 돌리고 이 브랜치는 결과만 읽음.
