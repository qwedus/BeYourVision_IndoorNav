"""변(모서리 사이)별 PDR 거리 오차 분석.

모서리 시각(자이로 회전 중간)으로 경로를 변 단위로 나눠, 변마다 정답 거리와 PDR 거리를 비교한다.
회전 근처(±1.5 s) 걸음과 세션 진행 구간별 평균 보폭도 출력한다.

사용: uv run python scripts/analyze_legs.py
"""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, json

from indoornav.pf.evaluate import load_pdr_runs
from indoornav.pf.map import WallMap
from indoornav.pf.replay import route_checkpoints
ROOT = Path(__file__).resolve().parents[1]
wm=WallMap.load(ROOT / "maps/home_loop/map.json")
runs=load_pdr_runs(ROOT, ROOT / "configs/pdr_tuned_v4.toml")
rows=[]; steprows=[]
for r in runs:
    cps=route_checkpoints(r["exp"],r["pdr"],r["t"],wm); st=r["pdr"].steps
    T=st.t.values; L=st.length.values
    # make t of START slightly before first step
    times=[c.t for c in cps]; times[0]=T[0]-1e-3
    for i,(a,b) in enumerate(zip(cps[:-1],cps[1:])):
        ta,tb=times[i],times[i+1]
        m=(T>ta)&(T<=tb)
        true=float(np.linalg.norm(b.xy-a.xy))
        if true<0.01 and r["exp"]!="S2": continue
        rows.append(dict(sid=r["sid"],exp=r["exp"],leg=f"{a.name}->{b.name}",i=i,true=true,est=L[m].sum(),n=int(m.sum()),
                         dur=tb-ta,feat=st.feature.values[m].mean() if m.any() else np.nan))
    # turn proximity per step
    corner_t=[c.t for c in cps if c.name in ("NE","NW","SW","SE","NE_back") or (r["exp"]=="S2" and c.name=="STRAIGHT_END")]
    for j in range(len(st)):
        d=min([abs(T[j]-ct) for ct in corner_t]) if corner_t else 99
        steprows.append(dict(sid=r["sid"],exp=r["exp"],t=T[j],L=L[j],feat=st.feature.values[j],near_turn=d<1.5,frac=(T[j]-T[0])/(T[-1]-T[0]+1e-9)))
df=pd.DataFrame(rows); df=df[df.true>1]
df["err_pct"]=100*(df.est-df.true)/df.true; df["speed"]=df.true/df.dur
pd.set_option("display.width",200)
print(df[["sid","leg","true","est","err_pct","n","speed"]].round(2).to_string(index=False))
S=pd.DataFrame(steprows); S=S[S.L>0]
print("\n[회전 근처(±1.5s) 걸음 vs 나머지] 평균 보폭")
g=S.groupby(["sid","near_turn"]).L.mean().unstack().round(3); print(g.to_string())
print("\n[세션 진행 구간별 평균 보폭 (0~1을 4등분)]")
S["q"]=pd.cut(S.frac,[0,.25,.5,.75,1.0001],labels=["1/4","2/4","3/4","4/4"],include_lowest=True)
print(S[S.exp.isin(["L1","L2","T2","S2"])].groupby(["sid","q"],observed=True).L.mean().unstack().round(3).to_string())
