"""과신 대책 시험: 과신 원인 분리에서 나온 고칠 후보를 옵션으로 넣어 재생한다. 기본 설정·관측 모델·PF는 안 바꿈, 채택 실험 아님.

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_overconf_fix.py --cond base e3 g30 e3_g30_c5
    ... --check     base(옵션 없음)가 기본 재생 exp_d58_r20과 모든 필드 비트까지 같은지
출력 <녹화 out>/ovf_<조건>/: runs.npz(키 known_tf_기본_<seed>, compare_runs.py로 기준과 짝 비교), summary.json(입력 기록 포함)

후보 (돌리기 전에 정한 값. 조건 이름은 이 조각을 _로 이음)
    e2·e3  바닥값 0.1 → 0.2 / 0.3 (관측 모델 eps)
    g30    조건부 먼 관측: 후보 거리 한계 30 m, 20 m 넘는 후보는 판 세로로 낸 크기 거리 s와 |s/d - 1| ≤ 0.3일 때만
           (exp_cause.gated_likelihood 그대로: 폭 3 m 넘는 판·세로 없는 판·위아래 잘린 박스는 먼 후보 없음)
    c3·c5  같은 track 누적 가중: 그 track의 k번째 입력 관측(같은 판 1 s 제한을 통과한 것, 건너뛴 것도 셈) 로그 우도에 가중 exp(-(k-1)/W) (W 3 / 5).
           관측은 모두 쓰고, 한 track이 주는 증거의 합은 관측 1/(1-e^(-1/W))개(W 3: 3.53, W 5: 5.52) 분량을 넘지 않음. track 번호는 시간 필터 추적 그대로(끊기면 새로 셈)
    s50·s20 후보 없는 관측 건너뛰기: 관측마다 그 관측 종류의 PF 후보(가시성 표·거리 한계·화각·앙각)가 하나라도 있는 파티클의
           가중 비율이 0.5 / 0.2 미만이면 그 관측을 넣지 않음. 정답 안 씀. g30과 함께면 후보도 크기 조건을 건 것(우도와 같은 후보)
    l05·l10·lp 시간 필터 보조 잇기(time_filter.apply 옵션, 5장 중 3번·IoU 0.3·같은 track 1 s 제한은 그대로):
           l05·l10 IoU 0.3 미만이면 박스 중심 거리 / 큰 대각선 ≤ 0.5 / 1.0 이면 같은 판(앱 Pipeline.swift center_dist_max와 같은 식)
           lp      앞 박스를 프레임 자세 pitch 변화만큼 세로로 옮긴 뒤 IoU(dy = fy·Δpitch, 기울기 1 고정)
           obs_nofilter에 옵션을 건 시간 필터를 다시 돌려 관측과 track 번호를 만든다(옵션 없으면 obs_timefilter 그대로)
    w1·w2·w3 같은 track 관측 약하게: 그 track의 k번째 입력 관측(같은 판 1 s 제한 통과) 우도를 L^alpha (로그 우도 × alpha, 바닥값은 먼저 건 점수에)
           w1 alpha = 1/sqrt(k)   w2 k=1이면 1, 이후 (1-r)/(1+r), r = W2_R (AR(1) 유효 표본)   w3 6/((k+1)(k+2)) (합 3k/(k+2) → 3)
           alpha는 그때까지 들어온 그 track 관측 수 k로만 정함(실시간, 건너뛰기와 함께면 건너뛴 관측도 셈). track 번호 없는 관측은 1
주의: 관측마다 나눠 계산하는 경로는 관측 모델 unknown='floor'(기본)에서만 한 번 계산과 같음. runs.npz의 n_obs는 건너뛴 관측도 셈
조합: 정해 둔 것만 (COMBOS). 옵션이 하나도 없으면 VisionFilter.update 그대로(기본 재생과 비트까지 같아야 함).
옵션이 있으면 관측마다 따로 우도를 계산해 로그를 가중합(가중 1·건너뛰기 없음·한계 20 m면 한 번에 계산한 것과 상수 차이만).
"""
import argparse
import json
import sys
import time

import numpy as np

import paths
import provenance as record
import replay as rp
import run_i1
from obs_model import DEFAULT, _cells, _kind_rows, kind_types, obs_likelihood
from pool_guard import run_pool
from time_filter import TrackLimiter
from vision_filter import VisionFilter

PIECES = {"e2": ("eps", 0.2), "e3": ("eps", 0.3), "g30": ("gate", 0.3), "c3": ("cap", 3.0), "c5": ("cap", 5.0),
          "s50": ("skip", 0.5), "s20": ("skip", 0.2),
          "l05": ("link", dict(center_dist_max=0.5)), "l10": ("link", dict(center_dist_max=1.0)), "lp": ("link", dict(pitch_comp=True)),
          "w1": ("wt", "sqrt"), "w2": ("wt", "ar1"), "w3": ("wt", "sat3")}
COMBOS = ("base", "e2", "e3", "g30", "c3", "c5", "s50", "s20", "e2_g30", "e3_g30", "e3_g30_c3", "e3_g30_c5",
          "e3_g30_c5_s50", "e3_g30_c5_s20", "e3_g30_s20", "l05", "e3_g30_l05", "e3_g30_l10", "e3_g30_lp",
          "w1", "e3_g30_w1", "e3_g30_w2", "e3_g30_w3")
GATE_RANGE = 30.0
W2_R = 0.93          # 과신 원인 문서 표 2 '이웃 상관' 9개 기록 중앙 (돌리기 전에 정함, 기록별로 안 맞춤)
_W = {}


def parse(cond):
    opt = dict(eps=None, gate=None, cap=None, skip=None, link=None, wt=None)
    if cond != "base":
        for p in cond.split("_"):
            k, v = PIECES[p]
            opt[k] = v
    if opt["cap"] and opt["wt"]:
        raise ValueError(f"{cond}: 누적 상한(c)과 관측 약하게(w)는 함께 못 씀")
    return opt


def cap_weight(k, W):
    """track의 k번째(1부터) 입력 관측 가중. 무한합 1/(1-e^(-1/W)) ≈ W + 0.5"""
    return float(np.exp(-(k - 1) / W)) if W else 1.0


def wt_alpha(k, kind):
    """track의 k번째(1부터) 입력 관측 alpha. 그때까지의 k만 씀"""
    if kind == "sqrt":
        return 1.0 / np.sqrt(k)
    if kind == "ar1":
        return 1.0 if k == 1 else (1.0 - W2_R) / (1.0 + W2_R)
    if kind == "sat3":
        return 6.0 / ((k + 1) * (k + 2))
    raise ValueError(kind)


def gated_map(fm, ob, tol):
    """exp_cause.gated_likelihood가 관측 하나에 쓰는 지도와 같은 것 (20 m 넘는 후보는 크기 조건)"""
    import exp_cause as ec
    H = np.array([d["size"][1] for d in fm["lm_json"]], float)
    Wd = np.array([d["size"][0] for d in fm["lm_json"]], float)
    L = len(H)
    lo, hi = np.full(L, np.inf), np.full(L, -np.inf)
    if ec.size_ok(ob):
        sz = ob["f_over_h"] * H
        ok = (Wd < ec.LONG_W) & (H > 0)
        lo[ok], hi[ok] = sz[ok] / (1 + tol), sz[ok] / (1 - tol)
    return dict(fm, dist=ec.GatedDist(fm["dist"], lo, hi))


def cand_fraction(x, y, theta, floor, ob, fmap, params, w, gate=None):
    """관측 하나의 종류 후보가 하나라도 있는 파티클의 가중 비율 (obs_model의 후보 규칙 그대로, gate면 크기 조건 건 지도)"""
    p = dict(DEFAULT, **(params or {}))
    fm = fmap.get(int(floor))
    if fm is None:
        return 0.0
    if gate:
        fm = gated_map(fm, ob, gate)
    k = np.where(np.isin(fm["lm_type"], kind_types(ob)))[0]
    if not len(k):
        return 0.0
    i, j, inside = _cells(fm, x, y)
    cand = _kind_rows(fm, k, x, y, theta, i, j, inside, p)[0]
    return float(w @ cand.any(0))


class FixFilter(VisionFilter):
    """VisionFilter + 후보 옵션. 옵션이 없으면 부모 update 그대로"""

    def __init__(self, *a, fix=None, **kw):
        super().__init__(*a, **kw)
        self.fix = fix or {}
        self.track_n = {}
        self.log = dict(skipped=[], weights=[])          # (시각, 관측 id) / 가중

    def update(self, obs, dpsi_deg=0.0, t=None):
        f = self.fix
        if not obs:
            return
        if not (f.get("gate") or f.get("cap") or f.get("skip") or f.get("wt")):
            return super().update(obs, dpsi_deg)
        t0 = time.perf_counter()
        x, y, c = self.enu_compass()
        c = c - dpsi_deg
        m = self.miss_frac(x, y)
        self.diag["miss_sum"] += m
        self.diag["miss_max"] = max(self.diag["miss_max"], m)
        w = np.exp(self.pf.logw)
        params = dict(self.obs_params or {})
        if f.get("gate"):
            params["max_range"] = GATE_RANGE
        lik_fn = _W["gated"] if f.get("gate") else obs_likelihood
        logl = np.zeros(len(x))
        for ob in obs:
            g = 1.0
            if (f.get("cap") or f.get("wt")) and ob.get("track_id") is not None:          # 건너뛰어도 입력 순서는 셈
                k = self.track_n.get(ob["track_id"], 0) + 1
                self.track_n[ob["track_id"]] = k
                g = cap_weight(k, f["cap"]) if f.get("cap") else wt_alpha(k, f["wt"])
            if f.get("skip") and cand_fraction(x, y, c, self.floor, ob, self.fmap, params, w, f.get("gate")) < f["skip"]:
                self.log["skipped"].append((t, ob["detection_id"]))
                continue
            if (f.get("cap") or f.get("wt")) and ob.get("track_id") is not None:
                self.log["weights"].append(g)
            logl += g * np.log(np.maximum(lik_fn(x, y, c, self.floor, [ob], self.fmap, params), 1e-300))
        self.pf.logw = self.pf.logw + (logl - logl.max())
        self.pf._normalize()
        self.pf._maybe_resample()
        self.diag["t_update"] += time.perf_counter() - t0
        self.diag["n_update"] += 1


def run_one_t(F, obs_frames, steps, rides, floors, truth, t0, dpsi, limiter):
    """replay.run_one과 같고, update에 시각을 넘김(건너뛴 관측 기록용). 기본 조건은 replay.run_one을 그대로 씀"""
    cur = {"t": None}
    orig = F.update

    def upd(val, d=0.0):
        return orig(val, d, cur["t"])
    F.update = upd
    lim0 = limiter

    def lim(t, val):
        cur["t"] = t
        return lim0(t, val)
    try:
        return rp.run_one(F, obs_frames, steps, rides, floors, truth, t0, dpsi=dpsi, limiter=lim)
    finally:
        F.update = orig


def link_obs(opt, link):
    """옵션을 건 시간 필터로 다시 만든 tf 관측 (track 번호도 같은 추적). run_i1._init의 기본 경로(글자 확인·종류 문턱 없음)와 같은 순서"""
    from time_filter import apply, load_frames, track_ids
    if opt.get("text_check") or opt.get("cls_th") or opt.get("bearing_fix") == "bearing":
        raise SystemExit("보조 잇기는 글자 확인·종류 문턱·bearing 방식과 함께 안 됨")
    fr = load_frames(paths.I1 / "obs_nofilter.jsonl")[1]
    obs = rp.frames_obs(apply(fr, **link), run_i1._G["t0"])
    return rp.attach_tracks(obs, track_ids(fr, **link)) if opt.get("track_limit") else obs


def _init(opt, n, cond):
    run_i1._init("indoornav", n, None, opt)
    fx = parse(cond)
    if fx["link"]:
        run_i1._G["obs"]["tf"] = link_obs(opt, fx["link"])
    if fx["gate"]:
        import exp_cause
        exp_cause._C["far_gate"] = fx["gate"]
        _W["gated"] = exp_cause.gated_likelihood
        run_i1._G["obs"]["tf"] = exp_cause.attach_size(run_i1._G["obs"]["tf"])
    _W.update(cond=cond, fix=fx, opt=opt)


def _job(seed):
    G, fx = run_i1._G, _W["fix"]
    r0 = G["truth"][0]
    params = {"eps": fx["eps"]} if fx["eps"] is not None else None
    F = FixFilter(G["P"], G["maps"], G["fmap"], G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
                  float(r0["heading_deg"]), seed, params, fix=fx)
    lim = TrackLimiter(_W["opt"]["track_limit"])
    t = time.perf_counter()
    if fx["gate"] or fx["cap"] or fx["skip"] or fx["wt"]:
        r = run_one_t(F, G["obs"]["tf"], G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"], G.get("dpsi"), lim)
    else:
        r = rp.run_one(F, G["obs"]["tf"], G["steps"]["known"], G["rides"], G["floors"], G["truth"], G["t0"], dpsi=G.get("dpsi"), limiter=lim)
    return seed, r, dict(skipped=F.log["skipped"], w_mean=float(np.mean(F.log["weights"])) if F.log["weights"] else None,
                         sec=time.perf_counter() - t)


def run_cond(a, opt, cond):
    res = run_pool(a.procs, _init, (opt, a.n, cond), _job, list(range(a.seeds)))
    out = paths.REC_OUT / f"ovf_{cond}"
    if (out / "runs.npz").exists() and not a.check:
        raise SystemExit(f"{out.name} 결과가 이미 있음 (덮어쓰지 않음)")
    return out, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", nargs="+", choices=COMBOS, default=list(COMBOS))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    opt = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
    if a.check:
        _, res = run_cond(a, opt, "base")
        z = np.load(paths.REC_OUT / "exp_d58_r20" / "runs.npz")
        ok = sum(all(r[f].tobytes() == z[f"known_tf_기본_{s}"][f].tobytes() for f in r.dtype.names) for s, r, _ in res)
        print(f"{paths.RECORDING} base vs exp_d58_r20: 전체 필드 같음 {ok}/{len(res)} seed")
        sys.exit(0 if ok == len(res) else 1)
    run_i1._init("indoornav", a.n, None, opt)          # 입력 기록용 (PF 값·걸음 파일)
    for cond in a.cond:
        out, res = run_cond(a, opt, cond)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "runs.npz", **{f"known_tf_기본_{s}": r for s, r, _ in res})
        summ = rp.summarize([rp.metrics(r) for _, r, _ in res])
        extra = {str(s): dict(n_skipped=len(x["skipped"]), skipped=x["skipped"], w_mean=x["w_mean"]) for s, _, x in res}
        info = dict(name=out.name, cond=cond, fix=parse(cond), gate_range=GATE_RANGE if parse(cond)["gate"] else None, n=a.n,
                    seeds=a.seeds, options=opt, sec_per_run=float(np.median([x["sec"] for _, _, x in res])),
                    provenance=record.run_record(run_i1.STEPS["indoornav"], run_i1._G["P"].to_dict(), a.seeds, opt,
                                                 extra=[paths.HERE / "exp_overconf_fix.py", paths.HERE / "exp_cause.py",
                                                        paths.ROOT / "vision" / "time_filter.py"]))
        (out / "summary.json").write_text(json.dumps(dict(info=info, summary={"known/tf/기본": summ}, per_seed=extra),
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{paths.RECORDING} {cond:16s} med {summ['med_all']:.2f} p90 {summ['p90_all']:.2f} max {summ['max_all']:.2f} "
              f"cover2 {summ['cover2']:.3f} 틀림 seed {summ['wrong']}  건너뜀 {np.median([len(x['skipped']) for _, _, x in res]):.0f}",
              flush=True)


if __name__ == "__main__":
    main()
