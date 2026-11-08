"""main 건의 4번(지도 자료) 재확인: 보행 마스크(main 원래 vs 방 뺀 사본)와 에스컬레이터 끝 좌표(main map_meta vs 그래프). 기본 설정은 안 바꿈.

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_map_data.py [--seeds 10] [--procs 11]
    .venv/Scripts/python integration/exp_map_data.py --report        9개 기록 → integration/out/map_data.md·.json

조건(나머지는 exp_main_recheck 기본 = exp_d58_r20 재생과 비트까지 같음)
    mask  noroom(지금, integration/maps 사본) / orig(main maps/dongchun/map_meta.json 마스크) — 초기 다시 뽑기·벽·하차 다시 뽑기 모두
    esc   graph(지금, 역사 패키지 그래프 좌표) / main(map_meta.json escalators 위·아래 끝) — 하차 위치만, 에스컬레이터 탄 기록만
프레임마다: 파티클 질량 중 '뺀 칸'(원래 마스크는 걸을 수 있고 방 뺀 사본은 아닌 칸)에 있는 비율
기하 확인(재생 없음): 정답 궤적(탑승 중 뺌)이 뺀 칸·두 마스크 밖을 지나는 시각, 에스컬레이터 축 위 두 마스크·역사 패키지 보행칸
출력 <녹화 out>/map_data/result.json
"""
import argparse
import json
import sys

import numpy as np

import paths
import exp_main_recheck as E
from vision_filter import load_maps

NAME = "map_data"
_M = {}


def main_escalators():
    """main map_meta escalators → 역사 패키지 순서에 맞춘 (top_aligned, bottom_aligned). 위 끝이 가장 가까운 것끼리 짝."""
    mm = json.loads(paths.ORIG_META.read_text(encoding="utf-8"))["escalators"]
    out = []
    for e in E._G["smeta"]["escalators"]:
        k = int(np.argmin([np.hypot(*(np.subtract(m["top"], e["top_aligned"]))) for m in mm]))
        out.append(dict(e, top_aligned=list(mm[k]["top"]), bottom_aligned=list(mm[k]["bottom"]), main_name=mm[k]["name"]))
    return out


class VMX(E.VFX):
    """VFX + 하차 좌표 바꾸기 + 프레임마다 '뺀 칸' 질량 기록"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.c.get("esc") == "main":
            self.esc = _M["esc_main"]
        self.room = []

    def estimate(self):
        rm = _M["removed"][self.floor]
        self.room.append(float(np.exp(self.pf.logw) @ rm(self.pf.xy)))
        return super().estimate()


def _init(n):
    E._init(n)
    mo, mn = load_maps(paths.ORIG_META), load_maps()
    _M.update(maps=dict(noroom=E._G["maps"], orig=mo), esc_main=main_escalators(),
              removed={fl: (lambda xy, a=mo[fl], b=mn[fl]: a.free(xy) & ~b.free(xy)) for fl in mo})


def _job(c):
    c = dict(c, ret_f=True)
    c0, m, lg, r, F = E._job(c, VMX, _M["maps"][c["mask"]])
    room = np.array(F.room)
    walk = ~r["esc"] & ~r["riding"] & ~paths.excluded(r["t"])
    m["unc_med"] = float(np.median(r["unc"][walk]))
    lg.update(room_med=float(np.median(room)), room_max=float(room.max()), room_t_max=float(r["t"][int(room.argmax())]),
              room=room.round(4).tolist(), room_t=r["t"].round(2).tolist() if c["seed"] == 0 else None)
    c0.pop("ret_f", None)
    return c0, m, lg


def conds(seeds):
    base = dict(steps="pdr", wall="kill", res="force", init="redraw", k="free", bfix="on", set="기본", mask="noroom", esc="graph")
    C = {}
    for mk in ("noroom", "orig"):
        for v in ("none", "tf"):
            C[f"{mk}/{v}"] = dict(base, mask=mk, vis=v)
    if paths.RC["exclude"]:
        C["noroom/tf중간"] = dict(base, vis="tf", set="중간")
        for v, st in (("none", "기본"), ("tf", "기본"), ("tf중간", "중간")):
            C[f"escmain/{v}"] = dict(base, vis="none" if v == "none" else "tf", set=st, esc="main")
    return [dict(v, name=k, seed=s) for k, v in C.items() for s in range(seeds)]


def geometry():
    """정답 궤적이 뺀 칸·마스크 밖을 지나는 곳, 에스컬레이터 축 위 칸 상태"""
    from station_pkg import enu_to_aligned, aligned_to_enu
    tr = E._G["truth"]
    rot = E._G["smeta"]["aligned_from_enu_rot_deg"]
    keep = (tr["on_esc"] == 0) & ~paths.excluded(tr["t"])
    xy = enu_to_aligned(np.c_[tr["east"], tr["north"]], rot)
    out = dict(n=int(keep.sum()))
    for nm in ("removed", "out_noroom", "out_orig"):
        bad = np.zeros(len(tr), bool)
        for fl in np.unique(tr["floor"]):
            s = keep & (tr["floor"] == fl)
            fl = int(fl)
            if nm == "removed":
                bad[s] = _M["removed"][fl](xy[s])
            else:
                bad[s] = ~_M["maps"]["noroom" if nm == "out_noroom" else "orig"][fl].free(xy[s])
        t = tr["t"][bad]
        spans = []
        for x in t:
            if spans and x - spans[-1][1] <= 0.25:
                spans[-1][1] = float(x)
            else:
                spans.append([float(x), float(x)])
        out[nm] = dict(frac=float(bad[keep].mean()), spans=spans[:20])
    fm = E._G["fmap"][-2]

    def stwalk(p):
        st = (aligned_to_enu(p, rot) - fm["T"]) @ fm["R"]
        j = np.floor((st[:, 0] - fm["x0"]) / fm["cell"]).astype(int)
        i = np.floor((st[:, 1] - fm["y0"]) / fm["cell"]).astype(int)
        h, w = fm["walk"].shape
        ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        o = np.zeros(len(p), bool)
        o[ok] = fm["walk"][i[ok], j[ok]]
        return o
    esc = []
    for e, em in zip(E._G["smeta"]["escalators"], _M["esc_main"]):
        y = e["bottom_aligned"][1]
        xs = np.arange(24.0, 36.01, 0.1)
        p = np.c_[xs, np.full_like(xs, y)]
        first = lambda v: float(xs[np.argmax(v)]) if v.any() else None
        esc.append(dict(name=e["name"], main_name=em["main_name"], graph_bottom=e["bottom_aligned"], main_bottom=em["bottom_aligned"],
                        graph_top=e["top_aligned"], main_top=em["top_aligned"],
                        first_free_orig=first(_M["maps"]["orig"][-2].free(p)), first_free_noroom=first(_M["maps"]["noroom"][-2].free(p)),
                        first_free_station=first(stwalk(p))))
    out["escalators"] = esc
    return out


def main_one(a):
    from pool_guard import run_pool
    res = run_pool(a.procs, _init, (a.n,), _job, conds(a.seeds))
    _init(a.n)
    R = {}
    for c, m, lg in res:
        R.setdefault(c["name"], dict(cond={k: v for k, v in c.items() if k not in ("seed", "name")}, seeds=[]))["seeds"].append(dict(seed=c["seed"], m=m, log=lg))
    out = paths.REC_OUT / NAME
    out.mkdir(parents=True, exist_ok=True)
    info = dict(rec=paths.RECORDING, n=a.n, seeds=a.seeds, exclude=paths.RC["exclude"], floors=E._G["floors"], geometry=geometry())
    (out / "result.json").write_text(json.dumps(dict(info=info, runs=R), ensure_ascii=False), encoding="utf-8")
    print(paths.RECORDING, "→", out / "result.json", len(res), "runs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.report:
        from exp_map_data_report import report
        return report()
    main_one(a)


if __name__ == "__main__":
    main()
