"""명령줄 실행: 프레임 폴더 · 아이폰 녹화본 · 재생 JSON → 결과 JSON 하나 (+ 선택: 오버레이 그림).

예 (cv_module 폴더에서):
  python -m cvmod.run --detector yolo26n --model <경로>/ft_ad_yolo26n.pt --frames <폴더> --files a.jpg b.jpg --out out.json
  python -m cvmod.run --detector yolo26n --model <경로>/ft_ad_yolo26n.pt --recording <녹화_YYYYMMDD_HHMM 폴더> --rate 5 --out out.json
  python -m cvmod.run --detector replay --replay out.json --out out2.json

카메라 값(K·자세)을 모르는 기존 영상 프레임은 --fx 등으로 가정값을 주고, 결과에 source="assumed"로 남는다.
녹화본은 frames.jsonl의 K·자세를 쓴다(source="device").
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

from .detectors import make_adapter
from .pipeline import Config, process_frame
from .schema import SCHEMA_VERSION, dumps
from .sources import images, recording

HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = HERE / "config" / "default.json"


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    return Config(json.loads(text)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main(argv=None):
    p = argparse.ArgumentParser(description="CV 모듈: 검출 → 관측 JSON")
    p.add_argument("--detector", required=True, help="yolo26n | yoloe | replay (설정 detectors 키)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", help="모델 .pt 경로 (yolo26n·yoloe)")
    p.add_argument("--replay", help="재생할 결과 JSON (replay)")
    p.add_argument("--frames", help="프레임 폴더")
    p.add_argument("--recording", help="아이폰 녹화본 폴더 (녹화_YYYYMMDD_HHMM)")
    p.add_argument("--rate", type=float, default=5.0, help="녹화본에서 초당 몇 장 (기본 5)")
    p.add_argument("--pick", default="gyro", choices=recording.PICKS, help="창 안에서 고르는 기준 (기본 gyro)")
    p.add_argument("--frames-out", help="녹화본에서 꺼낸 프레임 저장 폴더 (기본 <녹화본>/frames_<N>fps_<pick>)")
    p.add_argument("--k-frame", default="auto", choices=("auto", "asis"), help="녹화본 K 기준 판단 (기본 auto)")
    p.add_argument("--files", nargs="*", help="폴더 안에서 쓸 파일 이름 (없으면 *.jpg 전부)")
    p.add_argument("--out", required=True, help="결과 JSON 경로")
    p.add_argument("--overlay-dir", help="오버레이 그림을 저장할 폴더 (Pillow 필요)")
    p.add_argument("--fx", type=float, help="초점 거리 px (이미지 크기 기준). 녹화본은 K가 없을 때만 씀")
    p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float)
    p.add_argument("--cy", type=float)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--camera-height", type=float, default=None, help="카메라 높이 m (거리 계산에만 씀)")
    p.add_argument("--camera-source", default="assumed", help="K·자세 출처 (assumed | arkit | ...)")
    p.add_argument("--time-regex", default=None, help=r"파일 이름에서 촬영 시각(초)을 뽑는 정규식, 예: ^t(\d+\.\d+)")
    p.add_argument("--frame-prefix", default="", help="frame_id 앞에 붙일 글자 (예: dongchun/)")
    a = p.parse_args(argv)

    config, cfg_hash = load_config(a.config)
    adapter = make_adapter(a.detector, config, model_path=a.model, replay_path=a.replay)
    run_id = f"{a.detector}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run = {"run_id": run_id, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "config_file": Path(a.config).name, "config_version": config.raw.get("config_version"),
           "config_sha256_16": cfg_hash, **adapter.run_info()}

    frames = []
    if a.detector == "replay":
        for fi, dets, err in adapter.frames:
            frames.append(process_frame(fi, dets, config, run_id=run_id, error=err))
    else:
        if a.recording:
            infos, run["input"] = recording.load(
                a.recording, rate=a.rate, pick=a.pick, out_dir=a.frames_out, frame_prefix=a.frame_prefix,
                k_frame=a.k_frame, fallback_fx=a.fx, camera_height=a.camera_height)
            if "warning" in run["input"]:
                print("주의:", run["input"]["warning"])
        elif a.frames:
            if a.fx is None:
                p.error("--fx 필요 (K를 모르면 가정값을 주고 --camera-source assumed)")
            infos = [images.frame_info_for(path, a.fx, a.fy, a.cx, a.cy, a.roll, a.pitch, a.camera_height,
                                           a.camera_source, a.time_regex, a.frame_prefix)
                     for path in images.list_images(a.frames, a.files)]
        else:
            p.error("--frames 또는 --recording 필요")
        for fi in infos:
            try:
                dets, err = adapter.detect(fi), None
            except Exception as e:  # 검출 실패도 결과에 남긴다 (status=failed)
                dets, err = [], f"검출 실패: {type(e).__name__}: {e}"
            fi.process_time_s = time.time()
            frames.append(process_frame(fi, dets, config, run_id=run_id, error=err))

    out = {"schema_version": SCHEMA_VERSION, "run": run, "frames": [f.to_dict() for f in frames]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(dumps(out, indent=1), encoding="utf-8")
    counts = {s: sum(f.status == s for f in frames) for s in ("ok", "empty", "failed")}
    print(f"{len(frames)}프레임 {counts}, 관측 {sum(len(f.observations) for f in frames)}개 → {a.out}")

    if a.overlay_dir:
        from .overlay import draw_frame
        Path(a.overlay_dir).mkdir(parents=True, exist_ok=True)
        for f in frames:
            if f.frame_info.image_path:
                name = f.frame_info.frame_id.replace("/", "_")
                im = draw_frame(f)
                im.thumbnail((720, 1280))   # 저장 크기만 줄임 (좌표는 원본 기준으로 그린 뒤)
                im.save(Path(a.overlay_dir) / f"{a.detector}_{name}.jpg", quality=85)


if __name__ == "__main__":
    main()
