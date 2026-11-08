"""앱 OCR 판정이 붙은 관측을 정답 자리에서 지도와 맞춰 본다 (exp_ocr.py 짝, PF는 안 돌림. 정답에 의존하는 진단값).

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_ocr_obs.py --station <40 m 표 패키지 폴더>
    .venv/Scripts/python integration/exp_ocr_obs.py --all --station <폴더>
→ <녹화 out>/av_obs.json

관측: PC 시간 필터 관측(obs_timefilter, 위치 재생 입력과 같음), 결정: exp_ocr.py가 만든 text_check_applevision.jsonl(앱 상태 app_state).
읽기: 앱 ocr.jsonl의 읽기마다 그 검출의 관측(obs_nofilter, 앱 시간 필터는 PC와 달라 tf에 없을 수 있음).
정답 자리 + 카메라 방향(자이로 + 곧은 구간 중앙값, analyze_recording.camera_offset)에서 같은 종류·그 칸에서 보이는 면(40 m 표)과 방위 10° 안이면 맞음.
    near  20 m 안 면과 맞음 / far  20 m 안에는 없고 20~40 m 면과 맞음 / none  40 m 안 어떤 면과도 10° 밖 / outside  채점 제외·에스컬레이터
    near·far를 가르는 거리는 PF가 쓰는 가시성 표 거리(d_tab: 정답 자리가 든 칸 중심 → 판 중심, 높이 차 포함 3차원. 끝점 행도 중심 거리)라 PF 20 m 한계와 같음.
    화면 끝 검출은 PF와 같이 판 끝점과 비교(edge_faces, mode 'end'), edge 칸에 표시. 맞은 면이 여럿이면 방위 차이가 가장 작은 면(near 우선).
    d = 정답 자리 → 맞은 행의 점(판 중심, 끝점 비교는 그 끝점) 수평 거리. 읽은 거리 표는 d로 묶음(30 m 이상은 한 칸)
버린 관측(앱 reject)이 near·far면 그 종류의 진짜 판을 버렸을 가능성이 크다(우연 일치 포함). 바로잡은 관측은 원래 종류·바뀐 종류 둘 다 봄.
구간: 큰 오차 구간마다 관측 수를 분류 × 앱 상태로.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

import paths
import replay as rp
from analyze_recording import camera_offset
from edge_faces import add_face_points, edge_obs, frame_obs
from exp_far import diffs
from exp_ocr import DEC_MAIN, S2_APP, jl
from obs_model import kind_types
from station_pkg import load_station
from text_check import load_decisions
from time_filter import load_frames

TOL = 10.0
SEGMENTS = {"s2_b_emart_ex5": [(118, 168)], "s2_a_ex3_ex6": [(203, 233)], "s2_a_ex6_ex3": [(38, 53), (75.6, 111.7)]}
BINS = (0, 5, 10, 15, 20, 30, 40)


def bin_of(d):
    """수평 거리 → 칸 이름. 마지막 칸은 30 m 이상 전부(끝점은 40 m를 조금 넘을 수 있음)"""
    for a, c in zip(BINS[:-2], BINS[1:-1]):
        if a <= d < c:
            return f"{a}-{c}"
    return f"{BINS[-2]}+"


def match(fm, kinds, r, az):
    """→ (분류, 수평 거리, 면 id, 표 거리)"""
    d, df, k = diffs(fm, kinds, r, az)
    if not len(d):
        return "none", None, None, None
    for sel, c in ((d <= 20) & (np.abs(df) <= TOL), "near"), ((d > 20) & (np.abs(df) <= TOL), "far"):
        if sel.any():
            m = np.where(sel)[0]
            j = m[int(np.argmin(np.abs(df[m])))]
            h = float(np.hypot(fm["lm_xy"][int(k[j]), 0] - r["east"], fm["lm_xy"][int(k[j]), 1] - r["north"]))
            return c, h, str(fm["lm_id"][int(k[j])]), float(d[j])
    return "none", None, None, None


class Ctx:
    def __init__(self, station):
        self.t0 = paths.rec_t0()
        self.truth = rp.load_truth(paths.I1 / "truth.csv")
        self.fmap = add_face_points(load_station(station))
        h = np.load(paths.REC_OUT / "heading_indoornav.npz")
        self.ty, self.yaw = h["t_host"] - self.t0, -np.degrees(np.unwrap(h["heading"]))
        self.off = camera_offset(self.truth, paths.RC["waypoints"], self.ty, self.yaw)

    def place(self, t):
        """→ (정답 행, 카메라 방향) 또는 None(채점 제외·에스컬레이터·녹화 밖)"""
        if paths.excluded(np.array([t]))[0] or t < 0 or t > self.truth["t"][-1]:
            return None
        r = rp.truth_at(self.truth, t)
        if r["on_esc"]:
            return None
        return r, (np.interp(t, self.ty, self.yaw) + self.off) % 360

    def classify(self, o, t):
        p = self.place(t)
        if p is None:
            return "outside", None, None, None
        r, cam = p
        return match(self.fmap[int(r["floor"])], kind_types(o), r, (cam + o["bearing"]) % 360)


def obs_index(path, t0):
    """detection_id → (t, obs dict, 박스 세로 px, 종류만 바꿔 다시 만드는 함수)"""
    out = {}
    for f in load_frames(path)[1]:
        t, obs = frame_obs(f, t0)
        box = {d["detection_id"]: d["box_xyxy"] for d in f["detections"]}
        cv = {o["detection_id"]: o for o in f["observations"]}
        for o in obs:
            b = box[o["detection_id"]]

            def redo(map_types, o=o, c=cv[o["detection_id"]], b=b, fi=f["frame_info"]):
                types, bearing, _ = edge_obs(dict(c, map_types=map_types), b, fi)
                return dict(o, types=types, type=types[0], bearing=bearing)
            out[o["detection_id"]] = (t, o, b[3] - b[1], redo)
    return out


def one(station):
    C = Ctx(station)
    dec = load_decisions(paths.I1 / DEC_MAIN)[0]
    tf = obs_index(paths.I1 / "obs_timefilter.jsonl", C.t0)
    raw = obs_index(paths.I1 / "obs_nofilter.jsonl", C.t0)
    rows = []
    for did, (t, o, h, redo) in tf.items():
        d = dec[did]
        c, dist, face, dtab = C.classify(o, t)
        row = dict(t=t, id=did, label=o["label"], edge=o["edge"], state=d["app_state"], decision=d["decision"], cls=c, d=dist, d_tab=dtab, face=face,
                   h=h, text=d.get("text", ""))
        if d["decision"] == "relabel":
            c2, d2, f2, _ = C.classify(redo(d["map_types"]), t)
            row.update(to=d["relabel_to"], cls_to=c2, d_to=d2, face_to=f2)
        rows.append(row)
    reads = []
    for x in jl(paths.REC / "ocr.jsonl"):
        q = raw.get(x["detection_id"])
        if q is None:
            reads.append(dict(t=x["t"], label=x["label"], verdict=x["verdict"], cls="no_pc_obs", d=None, h=x.get("box_height_px"), text=x.get("text", "")))
            continue
        t, o, h, _ = q
        c, dist, face, dtab = C.classify(o, t)
        reads.append(dict(t=t, label=x["label"], edge=o["edge"], verdict=x["verdict"], cls=c, d=dist, d_tab=dtab, face=face, h=h, text=x.get("text", ""),
                          in_tf=x["detection_id"] in tf, track=x["track_id"]))
    out = dict(rec=paths.RECORDING, camera_offset=C.off, obs=rows, reads=reads, summary=summarize(rows, reads))
    (paths.REC_OUT / "av_obs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    s = out["summary"]
    print(f"{paths.RECORDING}: 버림 {s['reject']}, 바로잡음 {s['relabel']}", flush=True)


def summarize(rows, reads):
    rej = Counter((r["label"], r["cls"] + ("(끝)" if r["edge"] else "")) for r in rows if r["decision"] == "reject")
    rel = Counter((r["label"], r["cls"], r["to"], r["cls_to"]) for r in rows if r["decision"] == "relabel")
    by_state = defaultdict(Counter)
    for r in rows:
        by_state[r["state"]][r["cls"]] += 1
    bins = defaultdict(lambda: defaultdict(Counter))       # 종류 → 거리대 → 판정
    for x in reads:
        if x["cls"] in ("near", "far"):
            b = bin_of(x["d"])
            bins[x["label"]][b][x["verdict"]] += 1
            bins["전체"][b][x["verdict"]] += 1
    segs = {}
    for a, b in SEGMENTS.get(paths.RECORDING, []):
        sel = [r for r in rows if a <= r["t"] < b]
        segs[f"{a}-{b}"] = dict(n=len(sel), by=dict(Counter(f"{r['label']}|{r['cls']}|{r['state']}" for r in sel)),
                                far_faces=dict(Counter(r["face"] for r in sel if r["cls"] == "far")),
                                read=[dict(t=round(x["t"], 1), label=x["label"], verdict=x["verdict"], cls=x["cls"], d=x["d"], face=x.get("face"), text=x["text"])
                                      for x in reads if a <= x["t"] < b])
    return dict(reject={f"{k[0]}|{k[1]}": v for k, v in rej.items()}, relabel={"|".join(map(str, k)): v for k, v in rel.items()},
                by_state={k: dict(v) for k, v in by_state.items()},
                read_bins={k: {b: dict(c) for b, c in v.items()} for k, v in bins.items()}, segments=segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--station", required=True)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not a.all:
        return one(a.station)
    bad = 0
    for rec in S2_APP:
        p = subprocess.run([sys.executable, __file__, "--station", a.station], env={**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8"},
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(f"{rec}: 실패 (코드 {p.returncode})\n{p.stderr[-1500:]}", flush=True)
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
