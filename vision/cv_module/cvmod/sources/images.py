"""사진 폴더 → FrameInfo (run.py에서 분리, 동작 같음).

카메라 값(K·자세)을 모르는 기존 영상 프레임은 fx 등 가정값을 주고, 결과에 source="assumed"로 남긴다.
"""
import re
from pathlib import Path

from ..schema import FrameInfo, Intrinsics, Pose


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def list_images(folder, files=None):
    folder = Path(folder)
    return [folder / f for f in files] if files else sorted(folder.glob("*.jpg"))


def frame_info_for(path, fx, fy=None, cx=None, cy=None, roll=0.0, pitch=0.0, camera_height=None,
                   camera_source="assumed", time_regex=None, frame_prefix=""):
    path = Path(path)
    w, h = image_size(path)
    t = None
    if time_regex:
        m = re.search(time_regex, path.name)
        t = float(m.group(1)) if m else None
    fy = fy or fx
    return FrameInfo(
        frame_id=f"{frame_prefix}{path.stem}", image_width=w, image_height=h, image_path=str(path),
        intrinsics=Intrinsics(fx=fx, fy=fy, cx=cx if cx is not None else w / 2,
                              cy=cy if cy is not None else h / 2, width=w, height=h,
                              source=camera_source),
        pose=Pose(roll_deg=roll, pitch_deg=pitch, yaw_deg=None, yaw_ref=None,
                  camera_height_m=camera_height, source=camera_source),
        capture_time_s=t)
