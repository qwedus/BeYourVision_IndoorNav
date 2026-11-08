"""기하 핵심: 픽셀 → 카메라 기준 방위각·앙각, (높이가 주어질 때만) 수평 거리.

표준 라이브러리(math)만 쓴다. numpy·ultralytics 없이 import·시험할 수 있고, Swift로 옮기기도 쉽다.

출처: 저장소 `cv_test/bearing.py`(A2)의 `pixel_to_angles`, `box_to_obs` 거리 부분을 복사·정리했다.
    기준 커밋: ba9beea (bearing.py를 마지막으로 고친 커밋), 복사 시점 저장소 HEAD a1711e6 (2026-10-24)
    바꾼 점: numpy 행렬 → math 식으로 풀어 씀, 거리 계산을 별도 함수로 뗌. 계산 결과는 같다
             (tests/test_against_bearing.py로 저장소 안에서 확인).

카메라 좌표: x 오른쪽, y 위, -z 앞 (이미지 v는 아래로 커지므로 y = -(v - cy) / fy).
각도 약속 (bearing.py, landmark/meshrender.py와 같음)
    pitch: 카메라가 위를 보면 +                      [도]
    roll : 화면이 시계 방향으로 돌면 + (카메라 뒤에서 앞을 볼 때) [도]
    방위각: 카메라 정면 0°, 오른쪽 +   앙각: 수평 0°, 위 +
"""
import math


def pixel_to_angles(u, v, fx, fy, cx, cy, roll_deg=0.0, pitch_deg=0.0):
    """픽셀 (u, v) → (방위각, 앙각) [도].

    픽셀 방향을 '수평으로 편 카메라' 기준으로 돌린 뒤 잰다: d' = Rx(pitch) · Rz(-roll) · d
    """
    x, y, z = (u - cx) / fx, -(v - cy) / fy, -1.0
    # Rz(-roll)
    r = math.radians(-roll_deg)
    x, y = math.cos(r) * x - math.sin(r) * y, math.sin(r) * x + math.cos(r) * y
    # Rx(pitch)
    p = math.radians(pitch_deg)
    y, z = math.cos(p) * y - math.sin(p) * z, math.sin(p) * y + math.cos(p) * z
    az = math.degrees(math.atan2(x, -z))
    el = math.degrees(math.atan2(y, math.hypot(x, z)))
    return az, el


def box_center(box):
    """xyxy 박스 → 중심 (u, v)."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def horizontal_distance(elevation_deg, camera_height_m, landmark_height_m, min_elevation_deg=2.0):
    """앙각과 높이 차로 수평 거리 [m]. 못 구하면 None.

    높이 둘 다 '바닥에서' 잰 값(m). 다음이면 None:
      - 높이가 하나라도 None (매칭 전에는 표지판 높이를 모른다 — 같은 exit_sign도 2.61~3.25 m)
      - |앙각| < min_elevation_deg (표지판이 카메라 높이와 비슷해 오차가 폭발)
      - 높이 차와 앙각의 부호가 다름 (기하적으로 불가능)
    """
    if camera_height_m is None or landmark_height_m is None:
        return None
    dh = landmark_height_m - camera_height_m
    if abs(elevation_deg) < min_elevation_deg or dh == 0 or (dh > 0) != (elevation_deg > 0):
        return None
    return dh / math.tan(math.radians(elevation_deg))


def expected_elevation(horizontal_distance_m, camera_height_m, landmark_height_m):
    """후보 랜드마크의 예상 앙각 [도] — PF가 관측 앙각과 비교할 때 쓰는 역방향 식(참고용)."""
    return math.degrees(math.atan2(landmark_height_m - camera_height_m, horizontal_distance_m))
