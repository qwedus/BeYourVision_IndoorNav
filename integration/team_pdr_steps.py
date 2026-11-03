"""팀장 PDR(run_pdr, pdr_tuned_v4.toml)을 앱 녹화에 돌려 걸음 이벤트를 뽑는다 (연결안 3절 2).

    .venv/Scripts/python integration/team_pdr_steps.py      → integration/out/i1/steps_team.csv, heading_team.npz

steps_team.csv (한 줄 = 한 걸음, 우리 재생기 steps.csv 형식 + 열 추가)
    t             걸음 시각 [s] 호스트 시계 (= Session.t + t0_sensor)
    length_m      보폭 [m] (moving=False면 팀장 PDR이 0으로 줌)
    heading_deg   나침반 부호(시계 +) 상대 방향 = -deg(팀장 heading). 기준 0°는 녹화 시작(센서 첫 시각) 방향
    dheading_deg  앞 걸음 대비 변화, 나침반 부호 = -deg(팀장 dheading). **첫 걸음도 0이 아님**(녹화 시작 → 첫 걸음 회전)
    moving        팀장 bout_mask (False = 제자리 움직임으로 본 걸음, events()에는 없어서 steps에서 직접 꺼냄)
    dheading_rad  팀장 dheading 그대로(반시계 +, rad) — 팀장 PF에 넣는 값
heading_team.npz: 샘플별 방향(팀장 heading_series, 반시계 +, rad, t=0에서 0)과 호스트 시각 → 걸음 사이 회전 보정(4-2)용
"""
import csv

import numpy as np

import paths
from app_session import load_app_session
from indoornav.pdr import PDRParams, run_pdr

FIELDS = ("t", "length_m", "heading_deg", "dheading_deg", "moving", "dheading_rad")


def team_steps(config=paths.PDR_CONFIG):
    s, info = load_app_session()
    pdr = run_pdr(s, PDRParams.from_toml(config))
    st = pdr.steps
    rows = [dict(t=float(r.t) + info["t0_sensor"], length_m=float(r.length), heading_deg=-np.degrees(r.heading),
                 dheading_deg=-np.degrees(r.dheading), moving=bool(r.moving), dheading_rad=float(r.dheading))
            for r in st.itertuples()]
    return rows, dict(info, t_host=s.t + info["t0_sensor"], heading=pdr.heading, n_steps=pdr.n_steps,
                      distance=pdr.distance, config=str(config.relative_to(paths.ROOT)))


def write_steps(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if k in ("t", "dheading_rad") else int(r[k]) if k == "moving"
                            else f"{r[k]:.4f}") for k in FIELDS})


def read_steps(path):
    with open(path, encoding="utf-8") as f:
        out = []
        for r in csv.DictReader(f):
            d = {k: float(v) for k, v in r.items()}
            d["moving"] = bool(int(d.get("moving", 1)))
            out.append(d)
    return out


if __name__ == "__main__":
    rows, info = team_steps()
    out = paths.OUT / "i1"
    write_steps(rows, out / "steps_team.csv")
    np.savez_compressed(out / "heading_team.npz", t_host=info["t_host"], heading=info["heading"])
    mv = sum(r["moving"] for r in rows)
    print(f"팀장 PDR 걸음 {len(rows)}개 (moving {mv}), 거리 {info['distance']:.1f} m, 첫 dheading "
          f"{rows[0]['dheading_deg']:.1f}° → {out / 'steps_team.csv'}")
