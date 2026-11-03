# BeYourVision_IndoorNav
시각장애인을 위한 지하철역 실내 내비게이션 (PDR + 비전 맵매칭)

## 개발 환경

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
winget install --id=astral-sh.uv -e   # uv 설치 (최초 1회)
uv sync                               # 가상환경(.venv) 생성 + 라이브러리 설치
uv run pytest                         # 테스트 실행
```

## 폴더 구조

| 폴더 | 내용 |
|------|------|
| `indoornav/` | 코드 (`io` 데이터 로딩, `pdr` 추정, `eval` 평가) |
| `configs/` | 알고리즘 파라미터 (`pdr_default.toml` = 기준선) |
| `scripts/` | 전체 세션 평가(`run_pdr.py`), 파라미터 튜닝(`tune_pdr.py`) |
| `notebooks/` | 분석 노트북 |
| `tests/` | 테스트 |
| `docs/` | 데이터 포맷 명세, 테스트 프로토콜, 세션·경로 정답표(`sessions.csv`, `routes.csv`), 알고리즘 설명 |
| `third_party/` | 외부에서 가져온 코드 (원본 그대로) |
| `data/` | 원본 센서 데이터 (git에 올리지 않음) |
| `reports/` | 실행 결과 (git에 올리지 않음) |

## PDR 실행

```powershell
uv run python scripts/run_pdr.py --plots                      # 모든 세션 평가 + 그래프
uv run python scripts/tune_pdr.py --step-method adaptive --save configs/pdr_tuned_vN.toml --note "메모"  # 교차검증 튜닝 + 기록
uv run python scripts/sim_online_calib.py                       # 보폭 계수 온라인 보정 시뮬레이션
```

- 최신 설정: `configs/pdr_tuned_v3.toml`
- 변경 이력: [`docs/pdr_improvement_log.md`](docs/pdr_improvement_log.md), 튜닝별 상세: [`docs/tuning_log.md`](docs/tuning_log.md)

자세한 구조와 개발 순서는 [`docs/pdr_algorithm.md`](docs/pdr_algorithm.md).
