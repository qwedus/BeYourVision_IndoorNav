"""PC·앱 시간 필터 대조: 같은 입력에서 Python vision/time_filter.py와 앱 Pipeline.swift 규칙의 통과 집합 차이를 낸다. 필터는 안 고침.

    .venv/Scripts/python integration/check_tf_app.py                     # 기록 9개
    .venv/Scripts/python integration/check_tf_app.py s2_a_ex6_ex3 --list  # 다른 관측 목록까지

세 집합: PC(obs_nofilter에 time_filter.apply) / 이식(app_rule, 같은 입력에 Swift 규칙) / 앱 실제(녹화의 decisions.jsonl, 있을 때만).
비교 범위: PC 입력 관측 중 앱이 시간 필터 단계까지 판정한 것. 앱 실제가 없는 기록은 이식 규칙과만 비교.
앱 규칙에서 PC와 다른 점 (App/Sources/Filter/Pipeline.swift)
    끊김      : 프레임 5장(PC) / 마지막으로 잡힌 지 track_max_gap_s 1.0 s(앱)
    hits 정리 : 전부 보존(PC) / k ≤ k - 5 그리고 t ≤ t - 0.6인 것을 지움(앱). 세기는 양쪽 다 최근 5장
    동점      : IoU가 같으면 뒤 track(PC, >=) / 앞 track(앱, >)
    크기 상한 : 없음(PC) / size_limits(앱, 시간 필터 앞에서 빠짐). 보조 center_dist_max는 지금 null
값은 그 녹화의 rec/config.json(앱이 세션마다 복사), 없으면 APP_DEFAULT(앱 설정 파일과 같은 값).
"""
import argparse
import json
import sys
from pathlib import Path

import paths
from provenance import RECORDINGS
from time_filter import apply, iou, load_frames

APP_DEFAULT = {"time_filter": {"enabled": True, "count_by": "frames", "window_frames": 5, "window_s": 0.6, "min_hits": 3,
                               "match": {"same_label": True, "iou_min": 0.3, "center_dist_max": None}, "track_max_gap_s": 1.0},
               "size_limits": {"enabled": True, "max_height_ratio": {"area_map": 0.7}}}


def center_dist(a, b):
    dx, dy = (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
    da, db = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return (dx * dx + dy * dy) ** 0.5 / max(da, db, 1e-9)


def app_rule(frames, cfg=APP_DEFAULT):
    """Pipeline.swift process·match의 시간 필터 부분을 그대로 옮김(OCR 빼고). → (통과 id 집합, 크기 상한에서 빠진 id 집합)"""
    tf, sz = cfg["time_filter"], cfg.get("size_limits") or {}
    win, win_s, min_hits, gap = tf["window_frames"], tf["window_s"], tf["min_hits"], tf["track_max_gap_s"]
    m = tf["match"]
    tracks, passed, too_big = [], set(), set()
    for k, f in enumerate(frames):
        info = f["frame_info"]
        t, H = info["capture_time_s"], info["image_height"]
        boxes = {d["detection_id"]: d["box_xyxy"] for d in f.get("detections", [])}
        tracks = [tr for tr in tracks if t - tr["last"] <= max(gap, win_s)]
        used = set()
        for ob in f.get("observations", []):
            i, box = ob["detection_id"], boxes[ob["detection_id"]]
            limit = (sz.get("max_height_ratio") or {}).get(ob["label"]) if sz.get("enabled") else None
            if limit is not None and (box[3] - box[1]) / H > limit:
                too_big.add(i)
                continue
            best, score = None, 0.0
            for tr in tracks:
                if id(tr) in used or (m["same_label"] and tr["label"] != ob["label"]) or t - tr["last"] > gap + 1e-6:
                    continue
                v = iou(tr["box"], box)
                s = 1 + v if v >= m["iou_min"] else 0.0
                if s == 0.0 and m.get("center_dist_max") is not None:
                    dn = center_dist(tr["box"], box)
                    if dn <= m["center_dist_max"]:
                        s = 1 - dn / max(m["center_dist_max"], 1e-9) + 1e-6
                if s > score:
                    best, score = tr, s
            if best is None:
                best = dict(label=ob["label"], box=box, last=t, hits=[])
                tracks.append(best)
            used.add(id(best))
            best.update(box=box, last=t)
            best["hits"].append((k, t))
            if tf["count_by"] == "seconds":
                n = sum(ht > t - win_s - 1e-6 for _, ht in best["hits"])
            else:
                n = sum(hk > k - win for hk, _ in best["hits"])
            best["hits"] = [(hk, ht) for hk, ht in best["hits"] if not (hk <= k - win and ht <= t - win_s - 1e-6)]
            if not tf["enabled"] or n >= min_hits:
                passed.add(i)
    return passed, too_big


def app_decisions(path):
    """앱 decisions.jsonl → (시간 필터 통과 id, 판정한 id 전부)"""
    passed, judged = set(), set()
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            d = json.loads(line) if line.strip() else {}
            st = (d.get("stages") or {}).get("time_filter")
            if "detection_id" in d and st is not None:
                judged.add(d["detection_id"])
                if st.get("pass"):
                    passed.add(d["detection_id"])
    return passed, judged


def check(rec):
    """기록 하나 → 결과 dict"""
    rc = json.loads((paths.RECORDINGS / f"{rec}.json").read_text(encoding="utf-8"))
    inputs, folder = paths.ROOT / rc["inputs"], paths.ROOT / rc["rec"]
    raw = load_frames(inputs / "obs_nofilter.jsonl")[1]
    ids = {o["detection_id"] for f in raw for o in f.get("observations", [])}
    pc = {o["detection_id"] for f in apply(raw) for o in f["observations"]}
    saved = {o["detection_id"] for f in load_frames(inputs / "obs_timefilter.jsonl")[1] for o in f["observations"]}
    cfg_path = folder / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else APP_DEFAULT
    rule, too_big = app_rule(raw, cfg)
    res = dict(rec=rec, n_obs=len(ids), pc=len(pc), pc_saved_same=pc == saved, too_big=len(too_big), rule=len(rule),
               config="녹화" if cfg_path.exists() else "기본값", pc_vs_rule=len(pc ^ rule), pc_only_rule=sorted(pc - rule), rule_only=sorted(rule - pc))
    dec = folder / "decisions.jsonl"
    if dec.exists():
        app, judged = app_decisions(dec)
        scope = ids & judged
        res.update(app=len(app & scope), scope=len(scope), not_judged=len(ids - judged), pc_vs_app=len((pc ^ app) & scope),
                   pc_only=sorted((pc - app) & scope), app_only=sorted((app - pc) & scope), rule_vs_app=len((rule ^ app) & scope),
                   rule_app_diff=sorted((rule ^ app) & scope))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rec", nargs="*", help="기록 이름 (없으면 9개)")
    ap.add_argument("--list", action="store_true", help="다른 관측 id 목록까지 출력")
    ap.add_argument("--out", help="결과 JSON")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [check(r) for r in (a.rec or RECORDINGS)]
    print("| 기록 | 입력 관측 | PC 통과 | 앱 실제 통과 | PC↔앱 대칭차 | PC만 | 앱만 | 이식↔앱 대칭차 | PC↔이식 대칭차 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "app" in r:
            print(f"| {r['rec']} | {r['n_obs']} | {r['pc']} | {r['app']} | {r['pc_vs_app']}/{r['pc']} | {len(r['pc_only'])} | {len(r['app_only'])} | {r['rule_vs_app']} | {r['pc_vs_rule']} |")
        else:
            print(f"| {r['rec']} | {r['n_obs']} | {r['pc']} | (없음) | — | — | — | — | {r['pc_vs_rule']} |")
    for r in rows:
        if not r["pc_saved_same"]:
            print(f"경고 {r['rec']}: obs_timefilter.jsonl이 obs_nofilter에 필터를 다시 건 것과 다름")
        if r.get("not_judged"):
            print(f"참고 {r['rec']}: PC 입력 관측 중 앱이 시간 필터까지 판정 안 한 것 {r['not_judged']}개 (비교에서 뺌)")
        if r["too_big"]:
            print(f"참고 {r['rec']}: 앱 크기 상한에서 빠지는 관측 {r['too_big']}개 (PC 필터에는 없는 단계)")
        if a.list:
            for k in ("pc_only", "app_only", "rule_app_diff", "pc_only_rule", "rule_only"):
                if r.get(k):
                    print(f"  {r['rec']} {k}: {', '.join(r[k])}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
