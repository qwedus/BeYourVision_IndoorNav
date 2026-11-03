"""동춘 s1 모서리별 평가: 사용자 Annotation(모서리에서 누름) 시각의 PDR·PF 위치를 그린 경로의 모서리 좌표와 비교.
사용: uv run python scripts/dongchun_corner_eval.py [출발점 픽셀 x,y]  (기본 1480,1030)
"""
import sys,json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from run_dongchun import *
from indoornav.io import estimate_annotation_offset, align_annotations
meta=json.loads(META.read_text(encoding="utf-8")); g1=GridMap.load(META,"B01"); g2=GridMap.load(META,"B02")
P=params_from_toml(ROOT/"configs/pf_home_v3.toml"); P.n=1000; P.init_pos_sigma=0.5; P.init_h_sigma=np.radians(10)
s=load_session(str(ROOT / "data/dongchun/20261030_DC_B01toB02_s1")); pdr=run_pdr(s,PDRParams.from_toml(ROOT/"configs/pdr_tuned_v4.toml"))
c,_=estimate_annotation_offset(s); ann=align_annotations(s,c).t.values
corners=[("갈림목",14.9,"B01",(1595,1035)),("통로 남쪽 끝",80.2,"B01",(1592,1300)),("대각선 시작",185.2,"B01",(1290,1310)),
         ("대각선 끝",274.1,"B01",(1100,1415)),("개찰구 아래",299.4,"B01",(1100,1550)),("에스컬레이터 위",326.7,"B01",(1204.6,1527.2)),
         ("B02 꺾는 곳",393.9,"B02",(1760,505)),("도착",422.5,"B02",(1755,434))]
start_px=[int(a) for a in sys.argv[1].split(",")] if len(sys.argv)>1 else [1480,1030]
meta["route_truth"]["start_B01"]=g1.to_world(*start_px).tolist()
def run_trace(seed):
    # run_one 을 재구성해 시각별 추정 기록
    maps={"B01":g1,"B02":g2}; esc=meta["escalators"]; tops=np.array([e["top"] for e in esc]); bots=np.array([e["bottom"] for e in esc])
    ta,tb,_=escalator_window(s); st=pdr.steps; hrel=st.heading.to_numpy()-pdr.heading[0]; dh=np.diff(np.r_[0.0,hrel])
    pf=ParticleFilter(P,g1,seed=seed); pf.init(np.array(meta["route_truth"]["start_B01"]),0.0)
    rec=[];floor="B01";rode=False
    for i in range(len(st)):
        t=float(st.t.iloc[i])
        if floor=="B01" and t>=ta and not rode: rode=True
        if rode and floor=="B01" and t>=tb:
            d=np.linalg.norm(pf.xy[:,None,:]-tops[None],axis=2); pr=np.exp(-0.5*(d/10)**2)+1e-6; pr/=pr.sum(1,keepdims=True)
            k=(pf.rng.random(len(pr))[:,None]>np.cumsum(pr,1)).sum(1); pf.xy=bots[k]+pf.rng.normal(0,0.7,(len(k),2))
            bad=~g2.free(pf.xy); pf.xy[bad]=bots[k[bad]]; pf.logw=np.full(P.n,-np.log(P.n)); pf.map=g2; floor="B02"
        if floor=="B01" and rode: pf.predict(0.0,float(dh[i]),True)
        else: pf.predict(float(st.length.iloc[i]),float(dh[i]),bool(st.moving.iloc[i]))
        e=pf.estimate(); rec.append((t,e["xy"].copy(),e["k"],e["unc95"],floor))
    return rec
pdr_xy=pdr_only(pdr,meta,0)
T=pdr.steps.t.values
res=[run_trace(sd) for sd in range(8)]
print(f"출발점 px {start_px}")
print(f"{'지점':12s} {'시각':>6s} {'PDR오차':>7s} {'PF오차(중앙/최소/최대)':>22s} {'PF k':>5s} {'95%반경':>7s}")
for name,t,fl,px in corners:
    truth=(g1 if fl=="B01" else g2).to_world(*px)
    errs=[];ks=[];us=[]
    for rec in res:
        j=max(0,np.searchsorted([x[0] for x in rec],t)-1); errs.append(np.linalg.norm(rec[j][1]-truth)); ks.append(rec[j][2]); us.append(rec[j][3])
    jp=max(0,np.searchsorted(T,t)-1); ep=np.linalg.norm(pdr_xy[jp]-truth) if fl=="B01" else np.nan
    print(f"{name:12s} {t:6.1f} {ep:7.1f} {np.median(errs):8.1f} / {min(errs):4.1f} / {max(errs):4.1f}   {np.median(ks):5.2f} {np.median(us):7.1f}")
