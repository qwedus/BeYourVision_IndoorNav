"""main 수정 건의 재확인 실험: 벽 제거와 k, init()의 한 점 몰림, 걸음 사이 회전. indoornav·configs·vision 사본은 안 고침.

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_main_recheck.py [--seeds 10] [--procs 10] [--only base]
    .venv/Scripts/python integration/exp_main_recheck.py --report      9개 기록 결과 → integration/out/main_recheck.md·.json

출력: <녹화 out>/main_recheck/result.json (조건마다 seed별 지표·걸음 기록 요약). 기본 설정(configs, vision_options)은 안 바꿈.
같은 seed = 같은 초기 파티클(아래 PF 바꾸기는 첫 걸음부터만 다름).

PF 바꾸기 (PFX, indoornav ParticleFilter 상속. 기본 값이면 indoornav predict와 같은 순서로 난수를 써서 결과가 비트까지 같음)
    wall  kill   : 벽에 막힌 파티클 점수 0(지움) — indoornav 그대로
          pen    : 지우지 않고 로그 점수에 log(pen)을 더함, 위치는 그대로 (막힌 파티클이 없으므로 리샘플은 ESS 기준만)
          slide  : 막힌 파티클을 진행 방향 ±20·40·60°로 돌려 길이 × cos(각)만큼 옮겨 봄(벽 따라 미끄러짐, 방향 상태는 그대로).
                   안 되면 kill과 같음
          off    : 벽 없음(지도 None, 층 바뀐 뒤에도)
    res   force  : 막힌 파티클이 있으면 그 걸음에 바로 리샘플 — indoornav 그대로
          ess    : ESS < N·ess_frac(0.5)일 때만 리샘플(점수 0 파티클은 다음 리샘플까지 남음)
    모두 막힘: 점수가 남은 파티클이 모두 막히면(미끄러짐을 먼저 해 본 뒤) indoornav _recover로 흩음. 흩어도 산 파티클이 모두 막히면
          그 걸음은 안 옮김(점수 0 파티클이 정규화로 되살아나지 않게). force 모드에서는 indoornav와 같은 동작
    막힌 수(wall_kills): 점수가 있던 파티클 중 막힌 수. pen은 지우지 않고 벌점만 준 수, 같은 파티클이 계속 막히면 걸음마다 다시 셈
    init  redraw : 병합 코드(VisionFilter)처럼 pf.init() 뒤 걸을 수 있는 칸에서 N(xy0, 2²)·방향 N(th0, 20°) 다시 뽑기
          main   : indoornav init() 그대로(init_pos_sigma 2 m, init_h_sigma 20°). 벽 너머 표본은 출발점 한 점으로
    k     free / reset(하차 때 k = 1)
    bfix  on(채택: 걸음 사이 회전 보정) / off
    steps pdr / ideal(같은 걸음 시각, 보폭·방향은 정답 궤적 → 실제 k = 1) / drift(보폭 ×1.15, 0.4°/걸음)

걸음 기록(predict마다): 리샘플 직전 ESS, 리샘플 여부, 막힌 수, 가중 평균 k의 단계별 변화
    dk_rw   랜덤워크·클립, dk_wall  막힘 처리(점수 0·벌점) 뒤 정규화, dk_res  리샘플·지터, dk_upd  관측 단계
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import paths
import replay as rp
import run_i1
from indoornav.pf.filter import ParticleFilter
from vision_filter import VisionFilter

NAME = "main_recheck"
SLIDE_DEG = (20.0, 40.0, 60.0)
_G = run_i1._G
_X = {}


class PFX(ParticleFilter):
    def __init__(self, params, wall_map, seed=0, wall="kill", pen=0.1, res="force"):
        super().__init__(params, wall_map, seed)
        self.wall, self.pen, self.res = wall, pen, res
        self.slog = []
        self.slides = 0

    def _km(self):
        return float(np.exp(self.logw) @ self.k)

    def predict(self, length, dheading, moving=True):
        """indoornav predict와 같은 순서. 기록만 더하고 wall·res 바꾸기를 넣음."""
        p, n, r = self.p, self.p.n, self.rng
        km0 = self._km()
        self._recent_moving = (self._recent_moving + [bool(moving)])[-p.suspect_window:]
        self.k = np.clip(self.k + r.normal(0, p.sigma_k, n), p.k_min, p.k_max)
        self.th = self.th + dheading + r.normal(0, p.sigma_h0 + p.sigma_h1 * abs(dheading), n)
        km1 = self._km()
        if not moving and p.nonmoving_p > 0:
            go = r.random(n) < p.nonmoving_p
            L = np.where(go, r.uniform(p.nonmoving_lo, p.nonmoving_hi, n), 0.0)
            if not go.any():
                return self._log(km0, km1, km1, km1, np.nan, False, 0, False)
        elif length <= 0:
            return self._log(km0, km1, km1, km1, np.nan, False, 0, False)
        else:
            L = length * self.k * (1 + r.normal(0, p.sigma_len, n))
        new = self.xy + L[:, None] * np.column_stack([np.cos(self.th), np.sin(self.th)])
        nh = 0
        if self.map is not None:
            hit = self.map.crosses(self.xy, new)
            alive = np.isfinite(self.logw)       # force 모드는 걸음 시작 때 늘 전부 살아 있어 아래 조건이 indoornav의 hit.all()과 같음
            if self.wall == "slide" and hit.any():   # 미끄러짐을 먼저, 그래도 모두 막히면 흩기
                new, hit = self._slide(L, hit, new)
            if hit[alive].all():                 # 점수가 남은 파티클이 모두 막힘 → indoornav처럼 흩어서 다시 시작
                new, hit = self._recover(L)
                if hit[alive].all():             # 흩어도 산 파티클이 모두 막히면 이번 걸음은 안 옮김(죽은 파티클이 되살아나지 않게)
                    new, hit = self.xy.copy(), np.zeros(n, bool)
            nh = int((hit & alive).sum())        # 점수가 있던 파티클 중 막힌 수(이미 점수 0인 파티클은 안 셈)
            self.stats["wall_kills"] += nh
            self.logw = np.where(hit, self.logw + np.log(self.pen), self.logw) if self.wall == "pen" else np.where(hit, -np.inf, self.logw)
            new = np.where(hit[:, None], self.xy, new)
        self.xy = new
        self._normalize()
        km2, ess = self._km(), self.ess()
        force = self.res == "force" and self.map is not None and bool(np.isneginf(self.logw).any())
        n0 = self.stats["resamples"]
        self._maybe_resample(force=force)
        self._log(km0, km1, km2, self._km(), ess, self.stats["resamples"] > n0, nh, True)

    def _log(self, km0, km1, km2, km3, ess, res, nh, moved):
        self.slog.append((km1 - km0, km2 - km1, km3 - km2, ess, res, nh, moved))

    def _slide(self, L, hit, new):
        """막힌 파티클을 벽 따라 미끄러뜨림: 진행 방향 ±a 돌린 방향으로 L·cos(a). 돌리는 쪽은 파티클마다 무작위로 먼저 고름."""
        left = np.flatnonzero(hit)
        first = np.where(self.rng.random(len(left)) < 0.5, 1.0, -1.0)
        new, hit = new.copy(), hit.copy()
        for a in np.radians(SLIDE_DEG):
            for sg in (1.0, -1.0):
                if not len(left):
                    return new, hit
                ang = self.th[left] + sg * first * a
                cand = self.xy[left] + (L[left] * np.cos(a))[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
                ok = ~self.map.crosses(self.xy[left], cand)
                new[left[ok]] = cand[ok]
                hit[left[ok]] = False
                self.slides += int(ok.sum())
                left, first = left[~ok], first[~ok]
        return new, hit


class VFX(VisionFilter):
    """VisionFilter와 같은 초기화(난수 순서 같음) + PFX + 벽 끄기·하차 k = 1·init 그대로·관측 단계 k 기록."""

    def __init__(self, P, maps, fmap, smeta, floor, xy0_enu, compass0, seed, obs_params, c,
                 init_sigma=2.0, init_h_sigma_deg=20.0):
        from station_pkg import enu_to_aligned
        self.c = c
        self.P, self.maps, self.fmap, self.floor = P, maps, fmap, floor
        self.rot = smeta["aligned_from_enu_rot_deg"]
        self.esc = smeta["escalators"]
        self.obs_params = obs_params
        Pi = replace(P, init_pos_sigma=init_sigma, init_h_sigma=float(np.radians(init_h_sigma_deg))) if c["init"] == "main" else P
        if c["wall"] == "off":
            Pi = replace(Pi, use_map=False)
        self.pf = PFX(Pi, maps[floor], seed=seed, wall=c["wall"], pen=c.get("pen", 0.1), res=c["res"])
        xy0 = enu_to_aligned(xy0_enu, self.rot)[0]
        th0 = np.radians(90.0 + self.rot - compass0)
        self.pf.init(xy0, th0)
        self.collapse = float(np.all(self.pf.xy == xy0, axis=1).mean())
        if c["init"] != "main":
            r, n = self.pf.rng, P.n
            xy = np.empty((0, 2))
            while len(xy) < n:
                cc = xy0 + r.normal(0, init_sigma, (2 * n, 2))
                xy = np.vstack([xy, cc[maps[floor].free(cc)]])
            self.pf.xy = xy[:n]
            self.pf.th = th0 + r.normal(0, np.radians(init_h_sigma_deg), n)
            self.pf.logw = np.full(n, -np.log(n))
        self.diag = dict(t_predict=0.0, t_update=0.0, n_update=0, miss_sum=0.0, miss_max=0.0, snap_redraw=0)
        self.dk_upd = 0.0

    def update(self, obs, dpsi_deg=0.0):
        a = self.pf._km()
        super().update(obs, dpsi_deg)
        self.dk_upd += self.pf._km() - a

    def change_floor(self, d):
        super().change_floor(d)
        if self.c["wall"] == "off":
            self.pf.map = None
        if self.c["k"] == "reset":
            self.pf.k = np.ones_like(self.pf.k)


def _init(n):
    opt = json.loads((paths.HERE / "vision_options.json").read_text(encoding="utf-8"))
    run_i1._init("indoornav", n, None, opt)
    from diag_k_end import ideal_steps
    _G["steps"]["ideal"] = ideal_steps(_G["steps"]["known"], _G["truth"], _G["t0"], _G["smeta"]["aligned_from_enu_rot_deg"])


def _job(c, cls=None, maps=None):
    """cls·maps: 다른 실험이 VFX 상속 클래스·다른 보행 마스크로 같은 재생을 돌릴 때(없으면 VFX·기본 마스크)"""
    r0 = _G["truth"][0]
    st = run_i1.SETTINGS[c["set"]]
    F = (cls or VFX)(_G["P"], maps or _G["maps"], _G["fmap"], _G["smeta"], int(r0["floor"]), [r0["east"], r0["north"]],
            float(r0["heading_deg"]), c["seed"], st["obs_params"], c)
    from time_filter import TrackLimiter
    lim = TrackLimiter(_G["opt"]["track_limit"]) if _G["opt"].get("track_limit") else None
    obs = _G["obs"]["raw" if c["vis"] == "none" else "tf"]
    sk = {"pdr": "known", "ideal": "ideal", "drift": "drift"}[c["steps"]]
    r = rp.run_one(F, obs, _G["steps"][sk], _G["rides"], _G["floors"], _G["truth"], _G["t0"], vision=c["vis"] != "none",
                   obs_hz=st["obs_hz"], dpsi=_G.get("dpsi") if c["bfix"] == "on" else None, limiter=lim)
    m = rp.metrics(r)
    t = r["t"]
    walk = ~r["esc"] & ~r["riding"] & ~paths.excluded(t)
    cw = walk & (t > 2)
    m["cover2_walk"] = m["cover2"]
    m["ratio_walk"] = m["ratio"]
    ex = paths.RC["exclude"]
    if ex:
        b = np.searchsorted(t, ex[0][0]) - 1
        m["k_board"] = float(r["k"][max(b, 0)])
        aft = walk & (t >= ex[0][1])
        m["err_after"] = float(np.median(r["err"][aft])) if aft.any() else None
        m["p90_after"] = float(np.percentile(r["err"][aft], 90)) if aft.any() else None
        bef = walk & (t < ex[0][0])
        m["err_before"] = float(np.median(r["err"][bef])) if bef.any() else None
    S = np.array([x for x in F.pf.slog], dtype=float) if F.pf.slog else np.zeros((0, 7))
    mv = S[:, 6] > 0
    kill = mv & (S[:, 5] > 0)
    ess = S[mv, 3] / F.P.n
    lg = dict(n_pred=int(mv.sum()), n_kill_steps=int(kill.sum()), n_res_pred=int(S[mv, 4].sum()),
              res_frac=float(S[mv, 4].mean()) if mv.any() else 0.0,
              ess_pre_med=float(np.median(ess)) if mv.any() else None, ess_pre_p10=float(np.percentile(ess, 10)) if mv.any() else None,
              ess_pre_below=float(np.mean(ess < F.P.ess_frac)) if mv.any() else None,
              ess_kill_med=float(np.median(S[kill, 3] / F.P.n)) if kill.any() else None,
              kill_frac_med=float(np.median(S[kill, 5] / F.P.n)) if kill.any() else 0.0,
              dk_rw=float(S[:, 0].sum()), dk_wall=float(S[:, 1].sum()), dk_res=float(S[:, 2].sum()), dk_upd=F.dk_upd,
              wall_kills=F.pf.stats["wall_kills"], resamples=F.pf.stats["resamples"], all_dead=F.pf.stats["all_dead"],
              slides=F.pf.slides, collapse=F.collapse,
              k_mean_walk=float(np.mean(r["k"][walk])), k_min=float(np.min(r["k"][walk])),
              k_t=[float(r["k"][min(np.searchsorted(t, q), len(t) - 1)]) for q in np.linspace(0, t[-1], 11)])
    if c.get("ret_r"):
        return c, m, lg, r
    if c.get("ret_f"):
        return c, m, lg, r, F
    return c, m, lg


def conds(seeds, only=None):
    """조건 목록. 이름 → 조건(seed 없음)"""
    base = dict(steps="pdr", wall="kill", res="force", init="redraw", k="free", bfix="on", set="기본")
    C = {}

    def add(name, **kw):
        C[name] = dict(base, **kw)
    for st in ("pdr", "ideal"):
        for w in ("kill", "off"):
            for v in ("none", "tf"):
                add(f"{st}/{w}/{v}", steps=st, wall=w, vis=v)
    add("pdr/kill/tf중간", vis="tf", set="중간")
    VAR = {"pen01": dict(wall="pen", pen=0.1), "pen05": dict(wall="pen", pen=0.5), "ess": dict(res="ess"),
           "slide": dict(wall="slide"), "slide_ess": dict(wall="slide", res="ess")}
    for vn, kw in VAR.items():
        add(f"pdr/{vn}/none", vis="none", **kw)
        add(f"pdr/{vn}/tf", vis="tf", **kw)
        add(f"pdr/{vn}/tf중간", vis="tf", set="중간", **kw)
        add(f"ideal/{vn}/none", steps="ideal", vis="none", **kw)
    add("init_main/none", vis="none", init="main")
    add("init_main/tf", vis="tf", init="main")
    add("bfix_off/tf", vis="tf", bfix="off")
    add("bfix_off/tf중간", vis="tf", set="중간", bfix="off")
    if paths.RC["exclude"]:
        add("reset/none", vis="none", k="reset")
        add("reset/tf", vis="tf", k="reset")
        add("reset/tf중간", vis="tf", set="중간", k="reset")
        add("drift/free/tf", steps="drift", vis="tf")
        add("drift/reset/tf", steps="drift", vis="tf", k="reset")
        add("drift/free/none", steps="drift", vis="none")
        add("drift/reset/none", steps="drift", vis="none", k="reset")
    if only:
        C = {k: v for k, v in C.items() if any(k.startswith(o) for o in only.split(","))}
    jobs = [dict(v, name=k, seed=s) for k, v in C.items() for s in range(seeds)]
    if not only or "noise" in only:      # 잡음 기준: 기본 조건을 seed 10~19로
        jobs += [dict(C.get(k) or dict(base, vis=v), name=f"noise/{v}", seed=s) for k, v in (("pdr/kill/none", "none"), ("pdr/kill/tf", "tf")) for s in range(seeds, 2 * seeds)]
    return jobs


def stride_ratio():
    """PDR 거리 ÷ 정답 거리: 이동 걸음 보폭 합 vs 같은 시각 정답 궤적 길이(탑승·채점 제외 구간 뺌). 탑승 전만 따로."""
    tr, t0 = _G["truth"], _G["t0"]
    keep = (tr["on_esc"] == 0) & ~paths.excluded(tr["t"])
    seg = np.hypot(np.diff(tr["east"]), np.diff(tr["north"]))
    ok = keep[1:] & keep[:-1]
    ex = paths.RC["exclude"]
    out = {}
    for nm, lim in (("all", 1e18), ("before_ride", ex[0][0] if ex else None), ("after_ride", None)):
        if lim is None and nm == "before_ride":
            continue
        if nm == "after_ride":
            if not ex:
                continue
            sel_t = tr["t"][1:] >= ex[0][1]
            ss = lambda s: s["t"] - t0 >= ex[0][1]
        else:
            sel_t = tr["t"][1:] <= lim
            ss = lambda s, lim=lim: s["t"] - t0 <= lim
        d = float(seg[ok & sel_t].sum())
        L = float(sum(s["length_m"] for s in _G["steps"]["known"] if s["moving"] and ss(s)
                      and not any(a <= s["t"] - t0 <= b for a, b in _G["rides"]) and not paths.excluded(np.array([s["t"] - t0]))[0]))
        out[nm] = dict(truth_m=d, pdr_m=L, ratio=L / max(d, 1e-6))
    return out


def rotation_stats():
    """사진 찍은 순간 방향 − PF에 마지막으로 반영된 걸음 방향 |Δ| [도]: 관측이 있는 시간 필터 프레임(탑승 중 뺌),
    같은 프레임에 탑승 중 포함(1차 작업기록 4번 값 1.2·4.6·27.4°와 같은 정의), 탑승 중 뺀 모든 프레임."""
    rd = lambda t: any(a <= t <= b for a, b in _G["rides"])
    dp = run_i1._G["dpsi"]
    out = {}
    for nm, fr in (("tf_obs", [t for t, o in _G["obs"]["tf"] if o and not rd(t)]), ("tf_obs_with_ride", [t for t, o in _G["obs"]["tf"] if o]),
                   ("all_frames", [t for t, o in _G["obs"]["raw"] if not rd(t)])):
        a = np.abs([dp(t) for t in fr])
        out[nm] = dict(n=len(a), med=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max()),
                       over10=float(np.mean(a > 10)))
    return out


def init_collapse(seeds):
    """indoornav init() 그대로(위치 σ 2 m, 방향 σ 20°)에서 출발점 한 점에 모인 비율 — 출발점 앎 초기 분포."""
    from station_pkg import enu_to_aligned
    r0 = _G["truth"][0]
    rot = _G["smeta"]["aligned_from_enu_rot_deg"]
    xy0 = enu_to_aligned([r0["east"], r0["north"]], rot)[0]
    P = replace(_G["P"], init_pos_sigma=2.0, init_h_sigma=float(np.radians(20.0)))
    v = []
    for s in range(seeds):
        pf = ParticleFilter(P, _G["maps"][int(r0["floor"])], seed=s)
        pf.init(xy0, 0.0)
        v.append(float(np.all(pf.xy == xy0, axis=1).mean()))
    return dict(med=float(np.median(v)), min=float(min(v)), max=float(max(v)))


def main_one(a):
    from pool_guard import run_pool
    jobs = conds(a.seeds, a.only)
    res = run_pool(a.procs, _init, (a.n,), _job, jobs)
    _init(a.n)
    out = paths.REC_OUT / NAME
    out.mkdir(parents=True, exist_ok=True)
    R = {}
    for c, m, lg in res:
        R.setdefault(c["name"], dict(cond={k: v for k, v in c.items() if k not in ("seed", "name")}, seeds=[]))["seeds"].append(dict(seed=c["seed"], m=m, log=lg))
    info = dict(rec=paths.RECORDING, n=a.n, seeds=a.seeds, exclude=paths.RC["exclude"], rides=_G["rides"], floors=_G["floors"],
                stride=stride_ratio(), rotation=rotation_stats(), init_collapse=init_collapse(a.seeds), options=_G["opt"],
                n_steps=len(_G["steps"]["known"]))
    f = out / ("result.json" if not a.only else f"result_{a.only.replace(',', '_').replace('/', '-')}.json")
    f.write_text(json.dumps(dict(info=info, runs=R), ensure_ascii=False, indent=1), encoding="utf-8")
    print(paths.RECORDING, "→", f, len(res), "runs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--only")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rot-only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.rot_only:                       # 결과 파일의 회전 통계만 다시 계산(재생 안 함)
        _init(a.n)
        f = paths.REC_OUT / NAME / "result.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        d["info"]["rotation"] = rotation_stats()
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return print(paths.RECORDING, d["info"]["rotation"])
    if a.report:
        from exp_main_recheck_report import report
        return report()
    main_one(a)


if __name__ == "__main__":
    main()
