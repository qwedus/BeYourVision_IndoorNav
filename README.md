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
| `notebooks/` | 분석 노트북 |
| `tests/` | 테스트 |
| `docs/` | 데이터 포맷 명세, 테스트 프로토콜, 세션 기록표(`sessions.csv`) |
| `data/` | 원본 센서 데이터 (git에 올리지 않음) |
