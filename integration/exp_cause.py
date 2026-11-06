"""원인 분리 실험: 설정 하나만 바꿔 재생하고 k·벽 제거를 같이 기록한다. 기본 설정(configs, vision_options)은 안 바꿈, 채택 실험 아님.

    I1_RECORDING=<이름> .venv/Scripts/python integration/exp_cause.py --name exp_base
    ... --name exp_pdrv3 --pdr-config configs/pdr_tuned_v3.toml    걸음만 v3 (같은 PF·같은 관측)
    ... --name exp_kmin05 --k-min 0.5                              k 하한만 0.5
    ... --name exp_nodoor --drop-label door                        그 종류 관측만 뺌 (원인 확인용)
    ... --name exp_range40 --max-range 40                          관측 비교 거리 한계만 40 m (원인 확인용, 채택 아님)
    ... --name exp_d58_g30 --max-range 30 --far-gate 0.3           조건부 먼 후보: 20 m 넘는 후보는 판 크기 거리와 ±30% 안일 때만 (1 이상이면 조건 없음, 경로 확인용)
    ... --station <폴더>                                           다른 역사 패키지로 (가시성 표 거리가 다른 실험용)

출력 <녹화 out>/<name>/: summary.json, runs.npz(프레임별 k·오차 + 파티클 k 분포), steps_indoornav.csv·heading_indoornav.npz(걸음 바꾼 경우)
조건: 출발점 앎, 비전 끔 / 시간 필터·기본 / 시간 필터·중간, 채택 비전 옵션. 조건 정의는 run_i1과 같아서 같은 설정이면 run_i1 결과와 같아야 함.

프레임마다 추가 기록 (run_one이 estimate를 프레임당 한 번 부르는 것을 이용, 난수는 안 씀)
    k_near  가중 파티클 중 k < k_min + 0.02 비율
    k_p10·k_p50·k_p90  가중 k 분위
    walls   지금까지 벽 제거 수
    dk_pred·dk_upd  평균 k 누적 변화: 걸음 단계(벽 제거 + 클립 + 랜덤워크) / 관측 단계. 탑승·층 바뀜 처리는 뺌
"""
import argparse
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import paths
import replay as rp
import run_i1
from obs_model import obs_likelihood
from vision_filter import VisionFilter

SETTINGS = run_i1.SETTINGS
EXTRA = [("k_near", "f4"), ("k_p10", "f4"), ("k_p50", "f4"), ("k_p90", "f4"), ("walls", "i4"), ("dk_pred", "f4"), ("dk_upd", "f4")]
NEAR = 0.02


class LogFilter(VisionFilter):
    """VisionFilter + k·벽 제거 기록. 필터 동작은 그대로."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log, self.dk = [], [0.0, 0.0]

    def _km(self):
        return float(np.exp(self.pf.logw) @ self.pf.k)

    def predict(self, length, dheading_rad, moving=True):
        a = self._km()
        super().predict(length, dheading_rad, moving)
        self.dk[0] += self._km() - a

    def update(self, obs, dpsi_deg=0.0):
        a = self._km()
        super().update(obs, dpsi_deg)
        self.dk[1] += self._km() - a

    def estimate(self):
        w, k = np.exp(self.pf.logw), self.pf.k
        o = np.argsort(k)
        c = np.cumsum(w[o])
        q = k[o][np.minimum(np.searchsorted(c, [0.1, 0.5, 0.9]), len(k) - 1)]
        self.log.append((float(w @ (k < self.P.k_min + NEAR)), *q, self.pf.stats["wall_kills"], *self.dk))
        return super().estimate()


_C = {}
NEAR_M, LONG_W, EDGE_PX = 20.0, 3.0, 5.0     # 조건 안 거는 거리, 긴 판(일부만 잡힘) 폭, 위아래가 잘린 박스


class GatedDist:
    """가시성 표 행 하나의 관측용 보기: 20 m 넘는 칸은 그 판의 허용 거리 [lo, hi] 안일 때만 남김(obs_model은 그대로)"""

    def __init__(self, base, lo, hi):
        self.base, self.lo, self.hi, self.shape = base, lo, hi, base.shape

    def __getitem__(self, idx):
        v = self.base[idx].astype(float)
        k = idx[0]
        v[(v > NEAR_M) & ~((v >= self.lo[k]) & (v <= self.hi[k]))] = np.nan
        return v


def size_ok(ob):
    """크기 거리를 쓸 수 있는 관측인가: 박스 위아래가 안 잘림. 치수는 세로만 씀(폭은 보는 각도·화면 끝에 영향받음)"""
    return ob.get("f_over_h") is not None


def gated_likelihood(x, y, theta, floor, observations, fmap, params=None):
    """관측마다: 크기 거리 s = fx × 판 세로 ÷ 박스 세로. 20 m 넘는 후보는 |s/d - 1| ≤ tol일 때만. 폭 3 m 넘는 판, 세로 없는 판, 잘린 박스는 먼 후보 없음."""
    tol = _C["far_gate"]
    if tol >= 1 or not observations:
        return obs_likelihood(x, y, theta, floor, observations, fmap, params)
    fm = fmap[int(floor)]
    H, W = _C.setdefault(("sz", int(floor)), (np.array([d["size"][1] for d in fm["lm_json"]], float), np.array([d["size"][0] for d in fm["lm_json"]], float)))
    logw = np.zeros(len(x))
    for ob in observations:
        L = len(H)
        lo, hi = np.full(L, np.inf), np.full(L, -np.inf)
        if size_ok(ob):
            s = ob["f_over_h"] * H
            ok = (W < LONG_W) & (H > 0)
            lo[ok], hi[ok] = s[ok] / (1 + tol), s[ok] / (1 - tol)
        g = dict(fm, dist=GatedDist(fm["dist"], lo, hi))
        logw += np.log(np.maximum(obs_likelihood(x, y, theta, floor, [ob], {int(floor): g}, params), 1e-300))
    return np.exp(logw - logw.max())


def attach_size(obs_lists):
    """관측에 f_over_h(fx ÷ 박스 세로) 붙임. 위아래 잘린 박스는 None"""
    from time_filter import load_frames
    t = {}
    for f in load_frames(paths.I1 / "obs_timefilter.jsonl")[1]:
        fx = f["frame_info"]["intrinsics"]["fx"]
        H = f["frame_info"]["image_height"]
        for d in f["detections"]:
            b = d["box_xyxy"]
            t[d["detection_id"]] = None if b[1] < EDGE_PX or b[3] > H - EDGE_PX else fx / max(b[3] - b[1], 1.0)
    return [(tm, [dict(o, f_over_h=t.get(o["detection_id"])) for o in v]) for tm, v in obs_lists]


def _init(a_k_min, steps_dir, drop=None, max_range=None, station=None, far_gate=None):
    if station:
        paths.INPUTS = Path(station)
    if steps_dir:                     # 걸음 교체: run_i1은 REC_OUT 밑의 걸음·방향 파일을 읽음
        paths.REC_OUT = Path(steps_dir)
        run_i1.STEPS["indoornav"] = paths.REC_OUT / "steps_indoornav.csv"
    run_i1._init("indoornav", _C.get("n", 3000), None, json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8")))
    if a_k_min is not None:
        run_i1._G["P"] = replace(run_i1._G["P"], k_min=a_k_min)
    if max_range:
        for st in SETTINGS.values():
            st["obs_params"] = {**(st["obs_params"] or {}), "max_range": max_range}
    if far_gate is not None:
        import vision_filter
        _C["far_gate"] = far_gate
        vision_filter.obs_likelihood = gated_likelihood
        run_i1._G["obs"]["tf"] = attach_size(run_i1._G["obs"]["tf"])
    if drop:
        run_i1._G["obs"] = {k: [(t, [o for o in v if o["label"] != drop]) for t, v in fr] for k, fr in run_i1._G["obs"].items()}


def _job(c):
    G = run_i1._G
    r0 = G["truth"][0]
    st = SETTINGS[c["set"]]
    F = LogFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                  float(r0["heading_deg"]), c["seed"], st["obs_params"])
    obs = G["obs"]["raw" if c["obs"] == "none" else c["obs"]]
    from time_filter import TrackLimiter
    lim = TrackLimiter(G["opt"]["track_limit"]) if G["opt"].get("track_limit") else None
    r = rp.run_one(F, obs, G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"],
                   vision=c["obs"] != "none", obs_hz=st["obs_hz"], dpsi=G.get("dpsi"), limiter=lim)
    x = np.array(F.log, dtype=EXTRA)
    return c, r, x, dict(wall_kills=F.pf.stats["wall_kills"], resamples=F.pf.stats["resamples"], all_dead=F.pf.stats["all_dead"],
                         ms_update=1e3 * F.diag["t_update"] / max(F.diag["n_update"], 1), ms_predict=1e3 * F.diag["t_predict"] / max(len(G["steps"]["known"]), 1))


def path_ratio(r, truth):
    """1 s 간격으로 뽑은 PF 경로 길이 ÷ 같은 시각 정답 경로 길이 (층 이동·채점 제외 구간 뺌)"""
    tt = np.arange(r["t"][0], r["t"][-1], 1.0)
    i = np.minimum(np.searchsorted(r["t"], tt), len(r) - 1)
    j = np.minimum(np.searchsorted(truth["t"], tt), len(truth) - 1)
    ok = ~r["esc"][i] & ~r["riding"][i] & ~paths.excluded(tt)
    both = ok[1:] & ok[:-1]
    pf = np.hypot(np.diff(r["ex"][i]), np.diff(r["ey"][i]))
    tr = np.hypot(np.diff(truth["east"][j]), np.diff(truth["north"][j]))
    return float(pf[both].sum() / max(tr[both].sum(), 1e-6)), float(pf[both].sum()), float(tr[both].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pdr-config")
    ap.add_argument("--k-min", type=float)
    ap.add_argument("--drop-label")
    ap.add_argument("--max-range", type=float)
    ap.add_argument("--station")
    ap.add_argument("--far-gate", type=float)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = paths.REC_OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    steps_dir = None
    n_steps_info = None
    if a.pdr_config:
        from pdr_steps import indoornav_steps, write_steps
        rows, info = indoornav_steps(paths.ROOT / a.pdr_config)
        write_steps(rows, out / "steps_indoornav.csv")
        np.savez_compressed(out / "heading_indoornav.npz", t_host=info["t_host"], heading=info["heading"])
        steps_dir = str(out)
        n_steps_info = dict(n=len(rows), moving=sum(r["moving"] for r in rows))
    conds = [dict(obs="none", set="기본", seed=s) for s in range(a.seeds)]
    conds += [dict(obs="tf", set=st, seed=s) for st in SETTINGS for s in range(a.seeds)]
    with Pool(a.procs, _init, (a.k_min, steps_dir, a.drop_label, a.max_range, a.station, a.far_gate)) as pool:
        res = pool.map(_job, conds)
    _init(a.k_min, steps_dir, a.drop_label, a.max_range, a.station, a.far_gate)
    G = run_i1._G
    runs, ms, xs, ds, pr = {}, {}, {}, {}, {}
    for c, r, x, d in res:
        k = "known/none" if c["obs"] == "none" else f"known/tf/{c['set']}"
        runs.setdefault(k, []).append(r)
        xs.setdefault(k, []).append(x)
        ms.setdefault(k, []).append(rp.metrics(r))
        ds.setdefault(k, []).append(d)
        pr.setdefault(k, []).append(path_ratio(r, G["truth"]))
    summ = {}
    for k in ms:
        s = rp.summarize(ms[k], ds[k])
        s["path_ratio"] = float(np.median([p[0] for p in pr[k]]))
        s["n_steps_used"] = len(G["steps"]["known"])
        summ[k] = s
    dr = rp.dead_reckon(G["steps"]["known"], G["floors"], G["rides"], G["truth"], G["t0"], G["smeta"]["escalators"])
    summ["known/pdr_only"] = rp.dr_metrics(dr, G["truth"])[0]
    mv = [s for s in G["steps"]["known"]]
    info = dict(name=a.name, pdr_config=a.pdr_config or "configs/pdr_tuned_v4.toml", k_min=a.k_min if a.k_min is not None else G["P"].k_min,
                drop_label=a.drop_label, max_range=a.max_range, far_gate=a.far_gate, pf_config="configs/pf_home_v3.toml", n=G["P"].n, seeds=a.seeds, steps=n_steps_info, n_steps=len(mv))
    (out / "summary.json").write_text(json.dumps(dict(info=info, summary=summ), ensure_ascii=False, indent=1), encoding="utf-8")
    sv = {}
    for k in runs:
        for i, (r, x) in enumerate(zip(runs[k], xs[k])):
            sv[f"{k.replace('/', '_')}_{i}"] = r
            sv[f"x_{k.replace('/', '_')}_{i}"] = x
    sv["known_pdr_only_0"] = dr
    np.savez_compressed(out / "runs.npz", **sv)
    print(paths.RECORDING, a.name, {k: (round(summ[k]["med_all"], 2), round(summ[k]["p90_all"], 2)) for k in sorted(summ)}, flush=True)


if __name__ == "__main__":
    main()
