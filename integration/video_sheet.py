"""영상에서 시각 목록의 프레임을 꺼내 한 장으로 모음 (truth_helper가 cv2 있는 파이썬으로 부름. 브랜치 .venv엔 cv2 없음).

    python integration/video_sheet.py <작업.json>   # {video, frames(frames.jsonl), t0, times[], labels[], out}
"""
import bisect
import json
import sys

import cv2
import numpy as np

TILE_W, COLS = 400, 5


def main():
    job = json.loads(open(sys.argv[1], encoding="utf-8").read())
    ts = []
    with open(job["frames"], encoding="utf-8") as f:
        for line in f:
            ts.append(json.loads(line)["t"] - job["t0"])
    cap = cv2.VideoCapture(job["video"])
    if not cap.isOpened():
        raise SystemExit("영상 못 엶: " + job["video"])
    tiles = []
    for t, lab in zip(job["times"], job["labels"]):
        i = min(bisect.bisect_left(ts, t), len(ts) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        if not ok:
            im = np.zeros((1080, 1920, 3), np.uint8)
        h = int(im.shape[0] * TILE_W / im.shape[1])
        im = cv2.resize(im, (TILE_W, h), interpolation=cv2.INTER_AREA)
        # cv2 글꼴은 한글이 안 나와 시각만 그림에 적고, 번호 뜻은 후보 목록에서 봄
        text = lab.split()[0] + (f"  {t:.1f}s" if not lab.endswith(" s") else " s")
        cv2.rectangle(im, (0, 0), (TILE_W, 26), (0, 0, 0), -1)
        cv2.putText(im, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    rows = []
    blank = np.zeros_like(tiles[0])
    for k in range(0, len(tiles), COLS):
        row = tiles[k:k + COLS]
        rows.append(np.hstack(row + [blank] * (COLS - len(row))))
    out = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    buf.tofile(job["out"])       # 한글 경로
    print(job["out"], out.shape)


if __name__ == "__main__":
    main()
