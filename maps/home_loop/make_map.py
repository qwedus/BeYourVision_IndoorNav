"""집 근처 폐루프 복도 지도 생성 (인하대역 헤리움 메트로타워 중앙 사각 복도).

좌표계: 원점 = 중정(안쪽 사각형) 중심, +x = 도면 오른쪽(동), +y = 도면 위(북), 단위 m.
실측 (2026-10-29, 타일 60 cm):
  중정 벽: 가로 27×0.6 = 16.2 m, 세로 28×0.6 = 16.8 m
  복도 폭: 동쪽 1.79 m, 남쪽 1.413 m  (서=동, 북=남 대칭 가정)
  보행선: 중정 벽에서 동쪽 0.255+0.6 = 0.855 m, 남쪽 0.283+0.6 = 0.883 m (서=동, 북=남 대칭 가정)
  출발점: 동쪽 보행선, 중정 남쪽 벽 높이. 직선 루트 = 출발점 → 중정 북쪽 벽 높이 (16.8 m)
추정 (도면 비율): 대각 복도 입구 폭, 엘리베이터 홀 크기

실행: python maps/home_loop/make_map.py  → map.json, map.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---- 실측 ----------------------------------------------------------------
INNER_W = 27 * 0.6       # 중정 벽 가로 (x)
INNER_H = 28 * 0.6       # 중정 벽 세로 (y)
CORR_EW = 1.79           # 동·서 복도 폭
CORR_NS = 1.413          # 남·북 복도 폭
WALK_EW = 0.255 + 0.6    # 동·서: 중정 벽 → 보행선
WALK_NS = 0.283 + 0.6    # 남·북: 중정 벽 → 보행선
# ---- 추정 ----------------------------------------------------------------
DIAG_WIDTH = 2.0         # 모서리 → 대각 날개 복도 폭
DIAG_LEN = 5.0           # 지도에 넣는 대각 복도 길이 (안 걸었으므로 막다른 길)
LOBBY_W = 4.0            # 동·서 엘리베이터 홀 입구 폭
LOBBY_D = 3.0            # 엘리베이터 홀 깊이

OUT = Path(__file__).parent


def build() -> dict:
    ix, iy = INNER_W / 2, INNER_H / 2            # 중정 벽 반폭
    ox, oy = ix + CORR_EW, iy + CORR_NS          # 바깥 벽 반폭
    wx, wy = ix + WALK_EW, iy + WALK_NS          # 보행선 반폭
    g = DIAG_WIDTH / np.sqrt(2)
    walls: list[list[list[float]]] = []

    def seg(p, q):
        walls.append([[round(float(p[0]), 3), round(float(p[1]), 3)],
                      [round(float(q[0]), 3), round(float(q[1]), 3)]])

    # 중정 벽 (닫힘)
    c = [(-ix, -iy), (ix, -iy), (ix, iy), (-ix, iy)]
    for i in range(4):
        seg(c[i], c[(i + 1) % 4])

    # 모서리 대각 복도 (막다른 길)
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        p1 = np.array([sx * (ox - g), sy * oy])
        p2 = np.array([sx * ox, sy * (oy - g)])
        d = np.array([sx, sy]) / np.sqrt(2) * DIAG_LEN
        seg(p1, p1 + d)
        seg(p2, p2 + d)
        seg(p1 + d, p2 + d)
    # 북·남 바깥 벽
    for sy in (1, -1):
        seg((-(ox - g), sy * oy), (ox - g, sy * oy))
    # 동·서 바깥 벽 + 엘리베이터 홀
    lw = LOBBY_W / 2
    for sx in (1, -1):
        seg((sx * ox, -(oy - g)), (sx * ox, -lw))
        seg((sx * ox, lw), (sx * ox, oy - g))
        seg((sx * ox, -lw), (sx * (ox + LOBBY_D), -lw))
        seg((sx * (ox + LOBBY_D), -lw), (sx * (ox + LOBBY_D), lw))
        seg((sx * (ox + LOBBY_D), lw), (sx * ox, lw))

    r = lambda v: round(float(v), 3)
    cps = {
        "START": [r(wx), r(-iy)],          # 모든 세션 출발점 (파란별 옆 '출발')
        "STRAIGHT_END": [r(wx), r(iy)],    # 직선 루트 도착 (위쪽 파란별)
        "NE": [r(wx), r(wy)], "NW": [r(-wx), r(wy)],   # 폐루프 회전점 (노란별)
        "SW": [r(-wx), r(-wy)], "SE": [r(wx), r(-wy)],
        # 시계 방향 폐루프 출발점: SE 모서리에서 서쪽으로 (출발점↔SE 거리)만큼 (2026-10-29 측정)
        "START_W": [r(wx - (wy - iy)), r(-wy)],
    }
    loop = ["START", "NE", "NW", "SW", "SE", "START"]   # 출발 후 북쪽 → 반시계
    pts = np.array([cps[k] for k in loop])
    return {
        "name": "home_loop",
        "frame": "origin=중정 중심, +x=동(도면 오른쪽), +y=북(도면 위), m",
        "params": {"inner_w": INNER_W, "inner_h": INNER_H, "corr_ew": CORR_EW, "corr_ns": CORR_NS,
                   "walk_ew": WALK_EW, "walk_ns": WALK_NS, "diag_width": DIAG_WIDTH,
                   "diag_len": DIAG_LEN, "lobby_w": LOBBY_W, "lobby_d": LOBBY_D},
        "estimated": ["diag_width", "lobby_w", "lobby_d", "서·북 복도는 동·남과 대칭 가정"],
        "start_heading_deg": 90.0,         # 출발 방향 = +y(북) 가정
        "walls": walls,
        "checkpoints": cps,
        "routes": {
            "straight": ["START", "STRAIGHT_END"],
            "loop_ccw": loop,
            "loop_cw": ["START_W", "SW", "NW", "NE", "SE", "START_W"],   # 서쪽 보고 출발 → 시계 방향
        },
        "route_length_m": {
            "straight": r(2 * iy),
            "loop": r(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
            "loop_cw": r(np.linalg.norm(np.diff(np.array([cps[k] for k in
                                                           ["START_W", "SW", "NW", "NE", "SE", "START_W"]]),
                                                 axis=0), axis=1).sum()),
        },
    }


def plot(m: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break

    cps = m["checkpoints"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for (x0, y0), (x1, y1) in m["walls"]:
        ax.plot([x0, x1], [y0, y1], color="black", lw=2)
    lp = np.array([cps[k] for k in m["routes"]["loop_ccw"]])
    ax.plot(lp[:, 0], lp[:, 1], "-", color="tab:red", lw=1.2, alpha=0.7,
            label=f"폐루프 {m['route_length_m']['loop']} m")
    st = np.array([cps[k] for k in m["routes"]["straight"]])
    ax.plot(st[:, 0], st[:, 1], "-", color="tab:blue", lw=3, alpha=0.5,
            label=f"직선 {m['route_length_m']['straight']} m")
    for k, (x, y) in cps.items():
        col = "tab:blue" if k in ("START", "STRAIGHT_END") else "goldenrod"
        ax.plot(x, y, "*", color=col, ms=14, mec="black", mew=0.5)
        ax.annotate(k, (x, y), textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=9)
    sx, sy = cps["START"]
    ax.annotate("", xy=(sx, sy + 2.0), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
    ax.text(0, 0.5, "중정", ha="center", color="gray")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m] (동 →)")
    ax.set_ylabel("y [m] (북 →)")
    ax.set_title("home_loop (실측 2026-10-29)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    m = build()
    (OUT / "map.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(m, OUT / "map.png")
    print(m["route_length_m"], f"walls={len(m['walls'])}")
