"""정답 기준점(녹화 설정 waypoints) → 정답 궤적 truth.csv + 확인 그림 (localization_design cv_test/i1/truth.py를 녹화별 설정으로 옮김).

    I1_RECORDING=<이름> .venv/Scripts/python integration/truth_build.py
→ <inputs>/truth.csv, truth_waypoints.json, <out>/truth_check.png, truth_check.md

기준점 한 개 (integration/recordings/<이름>.json의 waypoints)
    name   이름, t 녹화 시작부터 초, floor 층(B1 = -1, B2 = -2)
    위치   st [x, y] 역 좌표(승강장 길이 방향 x, 폭 방향 y) 또는 node "GATE_3"(capstone_graph graph.json key) 둘 중 하나
    ref    무엇을 보고 정했나(기둥·벽 모서리·개찰구·에스컬레이터 끝·문). **판은 쓰지 않는다**(판 위치가 틀렸을 수 있어 채점이 순환됨). 옛 기록의 basis도 같이 검사
    esc    true면 다음 기준점까지 에스컬레이터(시간 비례 보간, 채점 제외)
    face   {"until": 초, "dir_st": [x, y]} 이 구간 앞부분 방향을 진행 방향 대신 이 값으로(계단에서 다른 쪽을 볼 때)
보간: 걷는 구간은 임시 PDR(steps_simple.csv) 걸음 수 비례(멈춘 동안 제자리), 걸음이 없으면 시간 비례. 정답이 평가 대상 PDR(indoornav)의
걸음에 묶이지 않게 임시 PDR 시각을 쓴다(보폭·방향은 안 씀). 정답 방향 = 그 구간 진행 방향. 제자리 구간은 앞 방향 유지(제자리 회전은 반영 안 함).
"""
import csv
import json
import re
import sys

import numpy as np

import paths

BAD_REF = ("판", "표지", "안내도", "노선도", "모니터", "타는 곳", "나가는 곳", "표 사는 곳", "발권기")
REF_KEYS = ("ref", "basis")   # 무엇을 보고 정했는지 적는 두 칸(옛 기록은 basis)
NOT_SIGN = "판단|판정"         # '판'이 들어가지만 판이 아닌 말
GRAPH = paths.ROOT.parent / "capstone_graph" / "data" / "dongchun" / "graph.json"
STEP_OK = (0.4, 1.0)        # 구간 보폭이 이 범위 밖이면 기준점 시각·위치를 다시 볼 것


def transforms():
    from station_pkg import load_station
    fm = next(iter(load_station(paths.INPUTS).values()))
    R, T = np.asarray(fm["R"]), np.asarray(fm["T"])
    return (lambda st: np.atleast_2d(st) @ R.T + T), (lambda en: (np.atleast_2d(en) - T) @ R)


def resolve(wps, en_to_st):
    """→ 위치 (N, 2) 역 좌표. 위치가 빠졌으면 어떤 기준점인지 적어 멈춘다. 참조물이 판이면 경고 목록으로 돌려준다."""
    missing = [w["name"] for w in wps if w.get("st") is None and not w.get("node")]
    if missing:
        raise SystemExit("위치(st 또는 node)가 빠진 기준점: " + ", ".join(missing))
    nodes = None
    P, warn = [], []
    for w in wps:
        if w.get("st") is not None:
            P.append(w["st"])
        else:
            nodes = nodes or {n.get("key"): n for n in json.loads(GRAPH.read_text(encoding="utf-8"))["nodes"]}
            if w["node"] not in nodes:
                raise SystemExit(f"그래프에 없는 node: {w['node']} ({w['name']})")
            n = nodes[w["node"]]
            P.append(en_to_st([n["x"], n["y"]])[0])
        for key in REF_KEYS:
            text = w.get(key, "")
            if any(b in re.sub(NOT_SIGN, "", text) for b in BAD_REF):
                warn.append(f"{w['name']}: {key} '{text}' — 판은 정답 기준으로 쓰지 않음(채점 순환)")
    if any(b <= a for a, b in zip([w["t"] for w in wps], [w["t"] for w in wps][1:])):
        raise SystemExit("기준점 시각이 늘어나는 순서가 아님")
    return np.array(P, float), warn


def build(wps, step_times, t0, dt=0.1):
    """step_times: 녹화 시작부터 초. → (행 목록, 기준점 정보, 구간 표)"""
    st_to_en, en_to_st = transforms()
    P, warn = resolve(wps, en_to_st)
    st_t = np.asarray(step_times)
    T = np.array([w["t"] for w in wps], float)
    rows = []
    for t in np.arange(0.0, T[-1] + 1e-9, dt):
        k = min(np.searchsorted(T, t, side="right") - 1, len(T) - 2)
        k = max(k, 0)
        a, b = T[k], T[k + 1]
        esc = bool(wps[k].get("esc"))
        if esc:
            f = (t - a) / (b - a)
        else:
            n_ab = ((st_t >= a) & (st_t < b)).sum()
            f = ((st_t >= a) & (st_t < min(t, b))).sum() / n_ab if n_ab else (t - a) / (b - a)
        p = P[k] + (P[k + 1] - P[k]) * min(max(f, 0.0), 1.0)
        d = P[k + 1] - P[k]
        face = wps[k].get("face")
        if face and t < face["until"]:
            d = np.array(face["dir_st"], float)
        en_d = st_to_en(d[None])[0] - st_to_en(np.zeros((1, 2)))[0]
        # 제자리 구간(두 기준점이 같음)은 진행 방향이 없음: 앞 방향 유지, 맨 앞이면 아래에서 뒤 방향으로 채움
        head = float(np.degrees(np.arctan2(en_d[0], en_d[1])) % 360) if np.hypot(*d) > 1e-6 else None
        if head is None and rows:
            head = rows[-1]["heading_deg"]
        en = st_to_en(p[None])[0]
        rows.append(dict(t=round(float(t), 3), t_host=t0 + float(t), east=float(en[0]), north=float(en[1]),
                         floor=wps[k]["floor"], on_esc=int(esc), heading_deg=head, segment=k))
    first = next((r["heading_deg"] for r in rows if r["heading_deg"] is not None), 0.0)
    for r in rows:
        if r["heading_deg"] is None:
            r["heading_deg"] = first
    info, seg = [], []
    for i, w in enumerate(wps):
        en = st_to_en(P[i][None])[0]
        info.append(dict(name=w["name"], t=w["t"], st=[round(float(P[i][0]), 2), round(float(P[i][1]), 2)],
                         east=round(float(en[0]), 2), north=round(float(en[1]), 2), floor=w["floor"],
                         node=w.get("node"), ref=w.get("ref", ""), basis=w.get("basis", "")))
        if i + 1 < len(wps):
            L = float(np.hypot(*(P[i + 1] - P[i])))
            n = int(((st_t >= T[i]) & (st_t < T[i + 1])).sum())
            seg.append(dict(a=w["name"], b=wps[i + 1]["name"], t0=w["t"], t1=wps[i + 1]["t"], length=L, steps=n,
                            per_step=L / n if n else None, esc=bool(w.get("esc"))))
    return rows, info, seg, warn


def save(rows, info, t0, out_csv, out_json):
    import io
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    data = f.getvalue().encode("utf-8")
    if not (out_csv.exists() and out_csv.read_bytes() == data):     # 내용이 같으면 안 씀(run_recording이 파일 시각으로 재생 결과를 다시 쓸지 정함)
        out_csv.write_bytes(data)
    out_json.write_text(json.dumps({"coords": "동·북 m (메시 원점), 층 B1 = -1, B2 = -2; st = 역 좌표", "t0_host": t0,
                                    "waypoints": info}, ensure_ascii=False, indent=1), encoding="utf-8")


def derived(wps):
    """기준점에서 나오는 채점 값: 층 이동 구간(+2 s, 내린 직후 걸음 제외), 끝 오차 시작(마지막 앞 기준점)"""
    ex = [[wps[i]["t"], wps[i + 1]["t"] + 2.0] for i, w in enumerate(wps[:-1]) if w.get("esc")]
    return dict(exclude=ex, end_start=wps[-2]["t"])


def seg_table(seg, warn):
    L = ["# 정답 기준점 확인", "", "| 구간 | 시각 (s) | 도면 거리 (m) | 걸음 | m/걸음 | 확인 |", "|---|---|---|---|---|---|"]
    for s in seg:
        bad = not s["esc"] and s["steps"] and not STEP_OK[0] <= s["per_step"] <= STEP_OK[1]
        L.append(f"| {s['a']} → {s['b']} | {s['t0']:.1f}~{s['t1']:.1f} | {s['length']:.1f} | {s['steps']} | "
                 f"{'—' if not s['per_step'] else format(s['per_step'], '.2f')} | "
                 f"{'에스컬레이터' if s['esc'] else '**걸음과 안 맞음: 시각·위치 다시 보기**' if bad else ''} |")
    if warn:
        L += ["", "**경고**", *[f"- {w}" for w in warn]]
    return "\n".join(L) + "\n"


def check_fig(rows, info, t0, out):
    """지도 위 정답 궤적·기준점 번호 + 자이로 방향과 정답 방향 비교"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from indoornav.pf.plots import set_korean_font
    from station_pkg import load_station
    set_korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fmap = load_station(paths.INPUTS)
    _, en_to_st = transforms()
    r = pd.DataFrame(rows)
    fls = [fl for fl in sorted(r["floor"].unique(), reverse=True)]
    fig = plt.figure(figsize=(9 * len(fls), 11))
    gs = fig.add_gridspec(2, len(fls), height_ratios=[3, 1])
    for j, fl in enumerate(fls):
        ax = fig.add_subplot(gs[0, j])
        fm = fmap[int(fl)]
        H, W = fm["walk"].shape
        ax.imshow(fm["walk"], origin="lower", cmap="Greys", alpha=0.35, vmin=0, vmax=2.5,
                  extent=[fm["x0"], fm["x0"] + W * fm["cell"], fm["y0"], fm["y0"] + H * fm["cell"]])
        q = r[r["floor"] == fl]
        st = en_to_st(q[["east", "north"]].to_numpy())
        ax.plot(st[:, 0], st[:, 1], c="red", lw=2)
        for i, w in enumerate(info):
            if w["floor"] == fl:
                ax.plot(*w["st"], "o", c="tab:blue", ms=5)
                ax.annotate(f"{i} ({w['t']:.1f}s)", w["st"], fontsize=9, color="tab:blue", xytext=(4, 4),
                            textcoords="offset points")
        ax.set_xlim(st[:, 0].min() - 12, st[:, 0].max() + 12)
        ax.set_ylim(st[:, 1].min() - 12, st[:, 1].max() + 12)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{paths.FLOOR_NAME[int(fl)]} 정답 궤적 (역 좌표 m, 숫자 = 기준점)")
    from truth_helper import yaw_series
    ty, yaw = yaw_series(t0)
    ax = fig.add_subplot(gs[1, :])
    th = np.degrees(np.unwrap(np.radians(r["heading_deg"].to_numpy())))
    y = np.interp(r["t"], ty, yaw)
    walk = r["on_esc"].to_numpy() == 0
    off = np.median((y - th)[walk])
    ax.plot(ty, yaw - off, c="tab:gray", lw=1, label="자이로 방향(가운데 맞춤)")
    ax.plot(r["t"], th, c="red", lw=1.5, label="정답 진행 방향")
    for w in info:
        ax.axvline(w["t"], c="tab:blue", lw=0.6, ls=":")
    ax.set_xlim(0, r["t"].max())
    ax.set_xlabel("녹화 시작부터 (s)")
    ax.set_ylabel("방향 (도)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from pdr_steps import read_steps
    wps = paths.RC["waypoints"]
    t0 = paths.rec_t0()
    steps = [s["t"] - t0 for s in read_steps(paths.I1 / "steps_simple.csv")]
    rows, info, seg, warn = build(wps, steps, t0)
    save(rows, info, t0, paths.I1 / "truth.csv", paths.I1 / "truth_waypoints.json")
    md = seg_table(seg, warn)
    paths.REC_OUT.mkdir(parents=True, exist_ok=True)
    (paths.REC_OUT / "truth_check.md").write_text(md, encoding="utf-8")
    check_fig(rows, info, t0, paths.REC_OUT / "truth_check.png")
    print(md)
    print("→", paths.I1 / "truth.csv", paths.REC_OUT / "truth_check.png")


if __name__ == "__main__":
    main()
