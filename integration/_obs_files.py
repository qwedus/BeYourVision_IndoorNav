"""cv_module 결과 → 재생 입력 obs_nofilter.jsonl·obs_timefilter.jsonl + detection_id·track_id가 재생까지 이어지는지 검사 (run_recording이 부름).

    python integration/_obs_files.py <cv_raw.json> <입력 폴더>
"""
import sys
from pathlib import Path

import paths  # noqa: F401
import replay as rp
from time_filter import IOU, MIN_HITS, WINDOW, apply, load_frames, save, track_ids


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw, inputs = Path(sys.argv[1]), Path(sys.argv[2])
    run, frames = load_frames(raw)
    save(run, frames, inputs / "obs_nofilter.json", inputs / "obs_nofilter.jsonl")
    res = apply(frames)
    save(dict(run or {}, time_filter=dict(window_frames=WINDOW, min_hits=MIN_HITS, iou=IOU, causal=True)), res,
         inputs / "obs_timefilter.json", inputs / "obs_timefilter.jsonl")
    # 재생기와 같은 길로 읽어 id가 살아 있는지 본다
    t0 = paths.rec_t0()
    ids = track_ids(frames)
    obs = {"필터 없음": rp.frames_obs(frames, t0), "시간 필터": rp.frames_obs(res, t0)}
    for k, v in obs.items():
        n = sum(len(o) for _, o in v)
        miss = sum(o.get("detection_id") not in ids for _, os_ in v for o in os_)
        rp.attach_tracks(v, ids) if not miss else None
        print(f"  {k}: 프레임 {len(v)}, 관측 {n}, detection_id·track_id 못 이은 관측 {miss}")
        if miss:
            raise SystemExit("detection_id가 시간 필터 번호와 안 이어짐: 입력 형식 확인 필요")
    if not frames:
        raise SystemExit("프레임 0개")


if __name__ == "__main__":
    main()
