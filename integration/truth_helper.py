"""정답 기준점 도우미: 녹화에서 기준점 시각 후보를 자동으로 뽑는다. 사람은 도면 위치만 넣으면 된다.

    I1_RECORDING=<이름> .venv/Scripts/python integration/truth_helper.py candidates    후보 → <out>/waypoints_draft.json, candidates.png, plan_B01·B02.png(역 좌표 도면), 영상이 있으면 sheet_candidates.jpg
    I1_RECORDING=<이름> .venv/Scripts/python integration/truth_helper.py sheet --from 30 --to 40 --every 0.5    그 구간 프레임 모음(기둥 지나는 순간 찾기) → <out>/sheet_30_40.jpg

후보: 출발·끝, 자이로 회전 가운데(수직축 회전 합 30° 이상), 앱 표시(marks.jsonl), 기압 층 이동(마지막 걸음·다음 첫 걸음으로 시각 다듬음), 2 s 넘는 멈춤.
채울 곳: waypoints_draft.json의 st(역 좌표) 또는 node(그래프 key), ref(기둥·벽 모서리·개찰구·에스컬레이터 끝·문). 판은 쓰지 않음.
필요 없는 후보는 지우고, 기둥을 지난 시각 같은 기준점은 sheet로 찾아 더한 뒤 recordings/<이름>.json의 waypoints에 넣고 truth_build.py.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import paths

SHEET = paths.HERE / "video_sheet.py"
TURN_WIN, TURN_RATE, TURN_MIN = 3.0, 8.0, 30.0        # 창 s, 도/s, 도 (천천히 도는 회전도 잡게 창으로 잼)
STOP_GAP = 2.0


def yaw_series(t0=None):
    """수직축 회전을 쌓은 방향(나침반 부호, 시작 0) → (녹화 시작부터 초, 도). 거치 방향과 상관없음."""
    t0 = paths.rec_t0() if t0 is None else t0
    m = pd.read_csv(paths.REC / "motion.csv")
    g = m[["gx", "gy", "gz"]].to_numpy()
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    # iOS 중력 벡터 = 지구 쪽(아래). 아래축 회전 + = 위에서 볼 때 시계 방향 = 나침반 +
    rate = (m[["wx", "wy", "wz"]].to_numpy() * g).sum(1)
    t = m["t"].to_numpy()
    yaw = np.r_[0.0, np.cumsum(rate[1:] * np.diff(t))]
    return t - t0, np.degrees(yaw)


def turns(t, yaw):
    """→ [(가운데 시각, 회전 합 도, 시작, 끝)]. TURN_WIN 창 평균 회전 속도가 TURN_RATE를 넘는 구간(1 s 안 끊김은 이음)의 합이 TURN_MIN 이상.
    걸음마다 좌우로 흔들리는 몸통은 창 평균에서 지워진다."""
    i0 = np.searchsorted(t, t - TURN_WIN / 2)
    i1 = np.minimum(np.searchsorted(t, t + TURN_WIN / 2), len(t) - 1)
    on = np.abs((yaw[i1] - yaw[i0]) / np.maximum(t[i1] - t[i0], 1e-3)) > TURN_RATE
    idx = np.where(on)[0]
    if not len(idx):
        return []
    groups, a, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[p] > 1.0:
            groups.append((a, p))
            a = i
        p = i
    groups.append((a, p))
    out = []
    for a, b in groups:
        d = yaw[b] - yaw[a]
        if abs(d) < 1e-6:
            continue
        if abs(d) >= TURN_MIN:
            mid = a + int(np.argmax(np.sign(d) * (yaw[a:b + 1] - yaw[a]) >= abs(d) / 2))
            out.append((float(t[min(mid, b)]), float(d), float(t[a]), float(t[b])))
    return out


def read_marks(t0):
    p = paths.REC / "marks.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [dict(t=float(r["t"]) - t0, kind=r.get("kind", "?"), what=r.get("what") or r.get("n")) for r in rows]


def candidates():
    import replay as rp
    from pdr_steps import read_steps
    t0 = paths.rec_t0()
    t, yaw = yaw_series(t0)
    dur = float(t[-1])
    steps = np.array([s["t"] - t0 for s in read_steps(paths.I1 / "steps_simple.csv")])
    rides, floors = rp.baro_events(rp.read_baro(paths.REC / "baro.csv", t0))
    c = [dict(t=0.0, source="출발", name="출발")]
    for tm, d, a, b in turns(t, yaw):
        c.append(dict(t=round(tm, 1), source=f"자이로 회전 {d:+.0f}° ({a:.1f}~{b:.1f} s, + = 오른쪽)",
                      name=f"{'오른쪽' if d > 0 else '왼쪽'} 회전"))
    ride_t = []
    for (a, b), (_, d) in zip(rides, floors):
        # 기압은 늦으므로 그 구간을 품는 가장 긴 걸음 공백으로 시각을 다듬는다
        before, after = steps[steps <= (a + b) / 2], steps[steps > (a + b) / 2]
        ta = float(before[-1]) if len(before) else a
        tb = float(after[0]) if len(after) else b
        ride_t.append((ta, tb))
        c.append(dict(t=round(ta, 1), source=f"기압 층 이동 {d:+d}층 시작(기압 {a:.1f} s, 마지막 걸음)", name="에스컬레이터 탐",
                      esc=True, node_hint="ESC_*_B1 또는 ESC_*_B2"))
        c.append(dict(t=round(tb, 1), source=f"기압 층 이동 끝(기압 {b:.1f} s, 다음 첫 걸음)", name="에스컬레이터 내림",
                      node_hint="ESC_*_B1 또는 ESC_*_B2"))
    if len(steps) > 1:
        for a, b in zip(steps[:-1], steps[1:]):
            if b - a > STOP_GAP and not any(x <= (a + b) / 2 <= y for x, y in ride_t):
                c.append(dict(t=round(float(a), 1), source=f"멈춤 {b - a:.1f} s (걸음 공백, 개찰구·문 앞일 수 있음)", name="멈춤"))
    for m in read_marks(t0):
        c.append(dict(t=round(m["t"], 1), source=f"앱 표시 {m['kind']} {m['what'] or ''}".strip(), name=f"표시 {m['kind']}",
                      mark=True))
    c.append(dict(t=round(dur, 1), source="녹화 끝", name="끝"))
    c.sort(key=lambda x: x["t"])
    draft = []
    for x in c:
        d = dict(name=x["name"], t=x["t"], st=None, node=None, floor=None, ref="", basis=x["source"])
        if x.get("esc"):
            d["esc"] = True
        if x.get("node_hint"):
            d["node_hint"] = x["node_hint"]
        if x.get("mark"):
            d["참고만"] = "앱 표시는 늦게 누를 수 있음. 회전·기압 후보와 비교할 것"
        draft.append(d)
    out = paths.REC_OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "waypoints_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    fig_candidates(t, yaw, steps, draft, out / "candidates.png", t0)
    plan_figs(out)
    video = find_video()
    if video:
        sheet_times(video, [d["t"] for d in draft], [f"{i} {d['name']}" for i, d in enumerate(draft)],
                    out / "sheet_candidates.jpg")
    print(f"후보 {len(draft)}개 → {out / 'waypoints_draft.json'}")
    for i, d in enumerate(draft):
        print(f"  {i:2d} {d['t']:6.1f} s  {d['name']:10s} {d['basis']}")
    return draft


def fig_candidates(t, yaw, steps, draft, out, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import replay as rp
    from indoornav.pf.plots import set_korean_font
    set_korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    axs[0].plot(t, yaw, lw=1)
    axs[0].set_ylabel("수직축 회전 합 (도, + = 오른쪽)")
    b = rp.read_baro(paths.REC / "baro.csv", t0)
    axs[1].plot(b[:, 0], b[:, 1], lw=1)
    axs[1].set_ylabel("기압 높이 (m)")
    axs[2].vlines(steps, 0, 1, lw=0.7)
    axs[2].set_ylabel("걸음")
    axs[2].set_yticks([])
    axs[2].set_xlabel("녹화 시작부터 (s)")
    for ax in axs:
        for d in draft:
            ax.axvline(d["t"], c="tab:red", lw=0.6, ls=":")
    for i, d in enumerate(draft):
        axs[0].text(d["t"], axs[0].get_ylim()[1], str(i), fontsize=8, color="tab:red", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plan_figs(out):
    """역 좌표 도면(1 m 눈금, 구멍 = 기둥·벽): 위치를 읽는 그림. 개찰구·에스컬레이터·계단·문·출구 그래프 key 표시"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from indoornav.pf.plots import set_korean_font
    from station_pkg import load_station
    from truth_build import GRAPH, transforms
    set_korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fmap = load_station(paths.INPUTS)
    _, en_to_st = transforms()
    nodes = json.loads(GRAPH.read_text(encoding="utf-8"))["nodes"] if GRAPH.exists() else []
    for fl, name in paths.FLOOR_NAME.items():
        fm = fmap[fl]
        H, W = fm["walk"].shape
        ext = [fm["x0"], fm["x0"] + W * fm["cell"], fm["y0"], fm["y0"] + H * fm["cell"]]
        fig, ax = plt.subplots(figsize=(max(10, W * fm["cell"] / 5), max(6, H * fm["cell"] / 5)))
        ax.imshow(fm["walk"], origin="lower", cmap="Greys", alpha=0.4, vmin=0, vmax=2, extent=ext)
        for n in nodes:
            if n.get("floor") == fl and n.get("type") in ("gate", "escalator", "elevator", "stairs", "exit") or (
                    n.get("floor") == fl and n.get("type") == "platform_door" and str(n.get("key", "")).endswith(("_1", "_4"))):
                p = en_to_st([n["x"], n["y"]])[0]
                ax.plot(*p, "o", c="tab:blue", ms=3)
                ax.annotate(n.get("key", ""), p, fontsize=6, color="tab:blue", xytext=(2, 2), textcoords="offset points")
        ax.set_xticks(np.arange(np.floor(ext[0] / 10) * 10, ext[1], 10))
        ax.set_yticks(np.arange(np.floor(ext[2] / 10) * 10, ext[3], 10))
        ax.minorticks_on()
        ax.grid(alpha=0.5)
        ax.grid(which="minor", alpha=0.12)
        ax.set_aspect("equal")
        ax.set_title(f"{name} 역 좌표 (m) · 어두운 곳 = 걸을 수 있는 곳, 구멍 = 기둥·벽 · 파란 글자 = node 값")
        fig.tight_layout()
        fig.savefig(out / f"plan_{name}.png", dpi=110)
        plt.close(fig)


def find_video():
    for n in ("video.mov", "video.mp4"):
        if (paths.REC / n).exists():
            return paths.REC / n
    return None


def sheet_times(video, times, labels, out):
    """video_sheet.py를 cv2가 있는 파이썬으로 돌림 (브랜치 .venv엔 cv2 없음)"""
    job = out.with_suffix(".json")
    job.write_text(json.dumps(dict(video=str(video), frames=str(paths.REC / "frames.jsonl"), t0=paths.rec_t0(), times=times,
                                   labels=labels, out=str(out)), ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([paths.VISION_PY, str(SHEET), str(job)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    job.unlink(missing_ok=True)
    if r.returncode:
        print("프레임 모음 실패:", r.stderr[-400:])
    else:
        print("→", out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("candidates", "sheet"))
    ap.add_argument("--from", dest="a", type=float)
    ap.add_argument("--to", dest="b", type=float)
    ap.add_argument("--every", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "candidates":
        candidates()
    else:
        video = find_video()
        if not video:
            raise SystemExit(f"영상 없음: {paths.REC}")
        ts = [float(x) for x in np.arange(a.a, a.b + 1e-9, a.every)]
        sheet_times(video, ts, [f"{x:.1f} s" for x in ts], paths.REC_OUT / f"sheet_{a.a:g}_{a.b:g}.jpg")


if __name__ == "__main__":
    main()
