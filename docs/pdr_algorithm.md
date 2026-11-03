# PDR 알고리즘 구조와 개발 방법

## 기준 알고리즘 (baseline)

교과서적인 PDR 조합을 `indoornav/pdr/`에 직접 구현해 두었다. 외부 코드를 복사하지 않고도 바로 튜닝을 시작할 수 있다.

| 단계 | 파일 | 방법 | 튜닝 파라미터 (`configs/*.toml`) |
|------|------|------|------|
| 전처리 | `preprocess.py` | 중력 방향으로 투영한 수직 가속도 → Butterworth 저역통과 | `[preprocess] lpf_cutoff_hz` |
| 걸음 검출 | `step_detect.py` | `method="peak"`: 수직 가속도 봉우리 (높이·돌출·최소 간격) | `[step] min_height, min_prominence, min_interval_s` |
| 걸음 검출 (적응형) | `step_adaptive.py` | `method="adaptive"`: 수직 흔들림이 작으면(느린 걸음) 가속도 크기 신호로 전환, 자기상관으로 걸음 주기 추정 후 봉우리 검출. 흔들림·주기성이 없으면 걸음 없음 | `[step] slow_energy, adaptive_prom, min_quality, min_energy` |
| 보폭 | `step_length.py` | Weinberg: `L = K·(a_max − a_min)^지수` (옵션: 걸음 빈도 항 추가 `weinberg_freq`) | `[length] k, exponent` (사람별 보정) |
| 헤딩 | `heading.py` | 자이로 수직축 회전 속도 적분, 걸음 구간 평균, (선택) ZARU | `[heading] source, zaru, still_*` |
| 위치 누적 | `pipeline.py` | `x += L·cos(h)`, `y += L·sin(h)` | – |
| 보폭 계수 온라인 보정 | `online_calib.py` | 랜드마크로 확정된 구간 거리 ÷ PDR 거리로 K를 조금씩 수정 (직선 구간·확실할 때만) | `CalibParams` (alpha, min_dist_m, max_turn_deg …) |

1차 튜닝 이후의 변경 이력과 판단 근거는 [`pdr_improvement_log.md`](pdr_improvement_log.md)에 있다.

모든 숫자는 `params.py`(코드 기본값)와 `configs/pdr_default.toml`(실행 기본값)에 있다. 두 파일은 `tests/pdr/test_pdr.py`가 일치하는지 검사한다.

### 좌표 약속
- 헤딩은 **출발 방향 = 0**, **좌회전(반시계) = +**, 단위 rad.
- iOS 중력 벡터는 지구 방향이므로 위쪽 단위벡터는 `-gravity/|gravity|`.
- 폰은 조끼 가슴에 세로로 고정 → 폰 방향 = 진행 방향으로 가정. 폰 방향을 바꾸면 이 가정부터 다시 확인한다.

## 개발 루프

```
1. 측정            Sensor Logger → zip 내보내기 → data/<session_id>/ 에 압축 해제 (zip 그대로 둬도 됨)
2. 기록            docs/sessions.csv 에 세션 한 줄 추가 (정답 거리, 직접 센 걸음 수, split)
3. 눈으로 확인     notebooks/pdr/01_explore_session.ipynb
4. 전체 평가       uv run python scripts/run_pdr.py --plots
5. 튜닝            uv run python scripts/tune_pdr.py --save configs/pdr_tuned.toml
6. 확인            uv run python scripts/run_pdr.py --config configs/pdr_tuned.toml --split eval
7. 알고리즘 수정   indoornav/pdr/*.py 고치기 → uv run pytest → 4번부터 반복
```

결과는 `reports/<날짜_시각>/`에 `metrics.csv`, `params.json`, (옵션) 세션별 그래프 PNG로 저장된다. `reports/`는 git에 올리지 않는다. 의미 있는 결과는 표로 정리해서 문서에 남긴다.

### tune / eval 분리 원칙
- `split=tune` 세션으로만 파라미터를 맞추고, `split=eval` 세션으로 확인한다.
- 매우 느린 걸음(`20261012_S1_slow_03`)은 가장 어려운 평가 케이스라 튜닝에 쓰지 않는다.

## 기능을 추가할 때

| 하고 싶은 것 | 어디에 |
|------|------|
| 새 걸음 검출 방법 (예: zero-crossing) | `step_detect.py`에 함수 추가, `StepParams`에 선택 옵션 추가 |
| 새 보폭 모델 (예: 걸음 빈도 선형 모델) | `step_length.py`의 `step_lengths()`에 `model` 분기 추가 |
| 헤딩 보정 (예: 통로 주방향 정렬 HDE) | `heading.py`에 함수 추가, `HeadingParams`에 on/off 옵션 |
| 정지·엘리베이터 판별 | `indoornav/pdr/` 아래 새 파일 (`stationary.py`, `floor.py`) |
| 새 지표 | `indoornav/eval/metrics.py`의 `evaluate()` |

새 방법은 **옵션으로 추가하고 기본값은 끈 상태**로 둔다. 같은 eval 세션에서 켰을 때와 껐을 때를 비교한 결과를 남긴 뒤 기본값을 바꾼다.

## 외부 코드를 가져올 때

GitHub 등의 PDR 구현을 참고하거나 가져올 때는 `third_party/<이름>/`에 **원본 그대로** 넣고 (`third_party/README.md` 참고), 우리 파이프라인과 연결하는 코드는 `indoornav/pdr/`에 따로 작성한다. 원본을 직접 고치지 않으면 출처·라이선스 관리와 비교 실험이 쉬워진다.

## 실시간(iOS) 포팅 시 주의
- 필터 방식 (`[preprocess] mode`)
  - `offline`: 녹화 전체를 앞뒤로 필터. 분석·튜닝용.
  - `delayed`: **앱에서 쓸 방식 (2026-10-16 결정).** 0.1초마다 최근 3초 구간을 앞뒤로 필터하고, 0.5초 전 지점까지만 확정한다. 위치가 0.5초 늦게 갱신되는 대신 정확도는 오프라인과 거의 같다.
  - `causal`: 과거 데이터만 쓰는 단방향 필터. 지연은 없지만 오차가 약 2배라 쓰지 않는다.
- 앱 이식 전에는 `mode = "delayed"`로 전체 평가를 돌려 오프라인과 차이가 없는지 확인한다.
- 걸음 구간 평균 헤딩도 걸음이 끝난 뒤 계산되므로 약 1걸음 지연이 있다.

## 튜닝 방식
- `scripts/tune_pdr.py`는 기본으로 **교차검증**(세션 하나씩 빼고 나머지로 고른 뒤 빼둔 세션으로 확인)을 쓴다. 출력되는 교차검증 오차가 '처음 보는 데이터에서의 예상 오차'다.
- 반복마다 골라진 값이 크게 흔들리면 데이터가 부족하다는 뜻이다.
- `--method split`을 주면 예전처럼 `sessions.csv`의 tune/eval 분할을 쓴다.
