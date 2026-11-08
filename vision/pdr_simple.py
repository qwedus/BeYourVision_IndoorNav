"""임시 PDR: 녹화본 motion.csv → 걸음 이벤트 CSV (팀장 PDR이 오면 이 파일만 바꿔 끼운다).

걸음 이벤트 형식 (steps.csv, 한 줄 = 한 걸음)
    t          걸음 시각 [s] (녹화본 호스트 시계, frames.jsonl·baro.csv t와 같음)
    length_m   보폭 [m]
    heading_deg 그 걸음 때 방향 [도], 나침반 부호(시계 +). **기준(0°)은 임의**(자세 기준틀 xArbitraryZVertical)라
               PF에는 앞 걸음과의 차이(dheading)만 넣는다
    dheading_deg 앞 걸음 대비 방향 변화 [도] (-180~180, 첫 걸음 0)
방향은 **카메라 축**(뒤 카메라 광축)의 수평 방향이다. 가로 거치·손에 든 녹화라 걷는 방향과 대개 같지만
몸을 돌리지 않고 폰만 돌리면 다르다.

방법 (단순하게)
    방향  : 자세 사원수(q)로 기기 -z(뒤 카메라 광축)를 기준틀로 돌려 수평 성분의 각도
    걸음  : 중력 방향 사용자 가속도(위 +)를 0.5~3 Hz로 걸러 꼭짓점(높이 0.05 g 이상, 간격 0.3 s 이상)
            서 있을 때(에스컬레이터 포함) 가짜 꼭짓점을 빼려고 주변 1초 가속도 표준편차 0.04 g 미만이면 버림
    보폭  : 고정값(기본 0.7 m). 정답 경로 길이로 맞춘 값은 결과 문서 참고
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FIELDS = ("t", "length_m", "heading_deg", "dheading_deg")


def read_motion(folder):
    rows = np.genfromtxt(Path(folder) / "motion.csv", delimiter=",", names=True)
    return rows


def camera_heading(m):
    """사원수 → 카메라 광축(기기 -z)의 수평 방향, 나침반 부호 [도] (기준 임의)."""
    w, x, y, z = m["qw"], m["qx"], m["qy"], m["qz"]
    # R(q) @ (0, 0, -1) = -(3번째 열)
    vx = -(2 * (x * z + w * y))
    vy = -(2 * (y * z - w * x))
    return (-np.degrees(np.arctan2(vy, vx))) % 360   # 기준틀 x에서 반시계 + → 나침반(시계 +)


def vertical_acc(m):
    g = np.c_[m["gx"], m["gy"], m["gz"]]
    a = np.c_[m["ax"], m["ay"], m["az"]]
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return -(a * g).sum(1)   # 위 + [g]


def detect_steps(t, av, min_height=0.05, min_gap=0.3, still_sd=0.04):
    fs = 1.0 / np.median(np.diff(t))
    b, a = butter(2, [0.5 / (fs / 2), 3.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, av)
    pk, _ = find_peaks(f, height=min_height, distance=max(1, int(min_gap * fs)))
    half = int(0.5 * fs)
    keep = [p for p in pk if np.std(av[max(0, p - half):p + half]) >= still_sd]
    return np.array(keep, int)


def run(folder, step_len=0.7, **kw):
    m = read_motion(folder)
    t = m["t"]
    head = camera_heading(m)
    pk = detect_steps(t, vertical_acc(m), **kw)
    # 걸음 시각의 방향: 앞뒤 0.1 s 원형 평균 (흔들림 줄임)
    out, prev = [], None
    for p in pk:
        sl = slice(max(0, p - 10), p + 11)
        a = np.radians(head[sl])
        h = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
        dh = 0.0 if prev is None else (h - prev + 180) % 360 - 180
        out.append(dict(t=float(t[p]), length_m=step_len, heading_deg=h, dheading_deg=dh))
        prev = h
    return out, (t, head)


def write_steps(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{r[k]:.6f}" if k == "t" else f"{r[k]:.3f}" for k in FIELDS})


def read_steps(path):
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="임시 PDR: motion.csv → steps.csv")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-len", type=float, default=0.7)
    a = ap.parse_args()
    rows, _ = run(a.recording, a.step_len)
    write_steps(rows, a.out)
    print(f"걸음 {len(rows)}개 → {a.out}")
