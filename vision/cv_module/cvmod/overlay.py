"""FrameResult → 박스 겹친 그림 (Pillow 필요). 종류·방위각만 적고 거리는 적지 않는다.

- 관측(문턱값 통과): 색 박스 + "지도 종류  방위각 / 앙각 / 점수"
- 버린 것 중 drop_label(광고)·not_in_map: 회색 가는 박스 + 이유 (문턱값 미만은 그리지 않음)
- 위 띠: 검출기, K·자세 출처(가정값이면 '가정값'이라고 적음)
"""
from PIL import Image, ImageDraw, ImageFont

COLORS = {"exit_sign": (237, 161, 0), "track_sign": (98, 80, 214), "info_monitor": (30, 150, 120),
          "station_map": (200, 60, 60), "route_map": (200, 60, 60)}
GRAY = (160, 160, 160)
KO = {"drop_label": "버림", "not_in_map": "지도에 없음"}


def _font(size, bold=True):
    for f in (["C:/Windows/Fonts/malgunbd.ttf"] if bold else []) + ["C:/Windows/Fonts/malgun.ttf",
                                                                    "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_frame(fr, scale=1.0):
    fi = fr.frame_info
    im = Image.open(fi.image_path).convert("RGB")
    W, H = im.size
    s = W / 1080 * scale
    dr = ImageDraw.Draw(im, "RGBA")
    f_lab, f_head = _font(int(30 * s)), _font(int(30 * s))
    by_id = {d.detection_id: d for d in fr.detections}
    for dp in fr.dropped:
        if dp.reason in KO and dp.detection_id in by_id:
            x1, y1, x2, y2 = by_id[dp.detection_id].box_xyxy
            dr.rectangle([x1, y1, x2, y2], outline=GRAY + (220,), width=max(2, int(3 * s)))
            dr.text((x1 + 4, y2 + 2), f"{dp.label} ({KO[dp.reason]})", font=_font(int(24 * s)), fill=GRAY + (255,))
    for o in fr.observations:
        x1, y1, x2, y2 = by_id[o.detection_id].box_xyxy
        c = COLORS.get(o.map_types[0], (46, 170, 107))
        dr.rectangle([x1, y1, x2, y2], outline=c + (255,), width=max(3, int(6 * s)))
        u, v = o.point_px
        dr.ellipse([u - 7 * s, v - 7 * s, u + 7 * s, v + 7 * s], fill=c + (255,))
        lines = ["/".join(o.map_types), f"방위 {o.bearing_deg:+.1f}°  앙각 {o.elevation_deg:+.1f}°  점수 {o.score:.2f}"]
        tw = max(dr.textlength(t, font=f_lab) for t in lines)
        lh = int(36 * s)
        tx = min(max(0, x1), W - tw - 12)
        ty = y2 + 6 if y2 + 2 * lh + 12 < H else max(0, y1 - 2 * lh - 12)
        dr.rectangle([tx, ty, tx + tw + 12, ty + 2 * lh + 6], fill=(20, 20, 20, 210))
        dr.text((tx + 6, ty + 2), lines[0], font=f_lab, fill=c + (255,))
        dr.text((tx + 6, ty + 2 + lh), lines[1], font=f_lab, fill=(255, 255, 255))
    k, p = fi.intrinsics, fi.pose
    src = "가정값" if "assumed" in (k.source, p.source) else k.source
    head = [f"{fr.observations[0].source if fr.observations else (fr.detections[0].source if fr.detections else '')}"
            f"  ·  {fi.frame_id}  ·  {fr.status}  ·  관측 {len(fr.observations)}개",
            f"K·자세 {src}: f={k.fx:.0f}px, roll={p.roll_deg:.0f}°, pitch={p.pitch_deg:.0f}°  ·  거리 표시 안 함"]
    hh = int(84 * s)
    dr.rectangle([0, 0, W, hh], fill=(31, 73, 125, 235))
    dr.text((14 * s, 4 * s), head[0], font=f_head, fill="white")
    dr.text((14 * s, 42 * s), head[1], font=_font(int(26 * s), bold=False), fill=(220, 230, 255))
    return im
