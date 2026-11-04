"""A4 관측 모델: 파티클마다 "이 자리·이 방향이면 지금 관측이 얼마나 그럴듯한가" (가중치).

순수 함수 하나(obs_likelihood)로 둔다. numpy만 쓰고 파일·전역 상태를 건드리지 않는다.
→ 서버(Python)나 아이폰(Swift)으로 옮길 때 이 파일만 옮기면 된다. 지도는 dict(아래 '지도 형식')로 받는다.

좌표 약속 (A1 제안)
    x, y  : 동(east)·북(north) m, 메시 원점
    theta : 카메라 정면 방향, 북 = 0°, 시계 방향 + (나침반)
    floor : 정수 층 (B1 = -1, B2 = -2)
관측 약속 (A2 box_to_obs 결과 그대로)
    bearing  : 카메라 정면에서 수평으로 몇 도 옆인지, 오른쪽 + [도]
    distance : 수평 거리 [m], 못 구하면 None
    type     : 랜드마크 종류 (지도와 같은 이름: exit_sign, track_sign, info_monitor, station_map, route_map)
               track_sign·direction_sign은 같은 종류로 봄(SAME_KIND). 대신 types=[...](후보 여러 개)도 받음
    score    : 검출 신뢰도 0~1

한 관측의 점수 = eps + (1 - eps) · score · max_후보 [ exp(-½(Δ방위/σb)²) · exp(-½(ln(d관측/d지도)/σd)²) ]
    후보 = 그 파티클 자리에서 가시성 표로 보이고(앞면·구조물 가림은 표에 이미 들어 있음),
          거리 max_range 안, 화각(± half_fov + fov_margin, 위로 max_elev) 안인 **같은 종류** 랜드마크
    - 가시성 표를 계산 안 한 칸(지도 computed 밖, 격자 밖)은 '안 보임'이 아니라 '모름'. unknown='floor'(기본)는 계산한 칸의
      '후보 없음'과 같게 바닥값을 주고, unknown='median'(진단용)은 그 층 계산한 파티클들의 중앙값을 준다
    - 비고유 랜드마크(같은 종류 여러 개)는 후보 중 최댓값 (설계서)
    - eps = 바닥값(이상치 혼합): 광고 오인 같은 틀린 검출 하나가 맞는 파티클을 0으로 만들지 않게
    - 거리 오차는 비율(σ 30%)이라 로그 비로 잰다. 거리가 None이면 방위각만 쓴다
파티클 가중치 = 관측 점수들의 곱.
**보여야 할 표지판이 안 보인 것은 점수에 넣지 않는다**(계획 6절 대책 2): 사람에 가려 못 본 것을
"그 자리가 아니다"로 오해하지 않게. 관측이 없으면 가중치는 모두 1(변화 없음).

지도 형식 (층마다 하나, fmap[floor])
    x0, y0, cell        보행 격자 원점·칸 크기 (격자 좌표 = '역 좌표'; 동·북에서 회전 R, 이동 T)
    R (2,2), T (2,)     격자 좌표 = (동북 - T) @ R
    dist (L, H, W)      가시성 표: 칸에서 랜드마크까지 거리, 안 보이면 nan (앞면·구조물 가림 반영)
    computed (H, W)     가시성을 계산한 칸 (없으면 격자 안 전부로 봄)
    lm_xy (L, 2)        랜드마크 중심 동·북
    lm_dz (L,)          랜드마크 중심 높이 - 카메라 높이 [m] (앙각 조건용)
    lm_type (L,)        종류 문자열
"""
import numpy as np

DEFAULT = dict(
    sigma_bearing=5.0,      # 도
    sigma_dist=0.30,        # 비율 (ln 기준)
    eps=0.1,                # 바닥값: 관측 하나의 최저 점수 (최고는 1)
    max_range=20.0,         # m. 인식 거리 15 m + 거리 오차 여유
    half_fov=30.0,          # 도
    fov_margin=5.0,         # 도. 파티클 방향이 조금 틀려도 화각 끝 표지판을 놓치지 않게
    max_elev=25.0,          # 도. 이보다 위는 화면 밖
    unknown='floor',        # 가시성 계산 안 한 칸: 'floor' 바닥값 / 'median' 중앙값(진단용)
)

# 같은 검출 종류로 보는 지도 종류 (10-31: 남색 판은 한 종류로 검출 → 타는 곳·방면 판 둘 다 후보, 층은 기압으로 구분)
SAME_KIND = {'track_sign': ('track_sign', 'direction_sign'), 'direction_sign': ('track_sign', 'direction_sign')}


def kind_types(ob):
    """관측 하나가 맞을 수 있는 지도 종류들. ob['types'](cv_module map_types) 또는 ob['type'], SAME_KIND로 넓힘."""
    types = ob.get('types') or [ob['type']]
    return sorted({u for t in types for u in SAME_KIND.get(t, (t,))})


def _cells(fm, x, y):
    """동·북 → 격자 (i, j), 격자 밖이면 inside=False"""
    st = (np.c_[x, y] - fm['T']) @ fm['R']
    j = np.floor((st[:, 0] - fm['x0']) / fm['cell']).astype(int)
    i = np.floor((st[:, 1] - fm['y0']) / fm['cell']).astype(int)
    H, W = fm['dist'].shape[1:]
    inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    return np.clip(i, 0, H - 1), np.clip(j, 0, W - 1), inside


def computed_at(fm, x, y):
    """동·북 점들이 가시성을 계산한 칸 위인가 (False = 그 자리의 nan은 '모름')"""
    i, j, inside = _cells(fm, x, y)
    return inside & fm['computed'][i, j] if 'computed' in fm else inside


def _kind_rows(fm, k, x, y, theta, i, j, inside, p):
    """랜드마크 행 k만 (len(k), n) : 후보(보임·거리·화각 안), 수평 거리, 상대 방위(오른쪽 +)"""
    d_tab = fm['dist'][k[:, None], i[None], j[None]].astype(float)     # nan = 안 보임
    vis = np.isfinite(d_tab) & (d_tab <= p['max_range']) & inside[None]
    dx = fm['lm_xy'][k, 0:1] - x[None]
    dy = fm['lm_xy'][k, 1:2] - y[None]
    hd = np.hypot(dx, dy)
    rel = (np.degrees(np.arctan2(dx, dy)) - theta[None] + 180) % 360 - 180
    elev = np.degrees(np.arctan2(fm['lm_dz'][k, None], np.maximum(hd, 1e-6)))
    return vis & (np.abs(rel) <= p['half_fov'] + p['fov_margin']) & (elev <= p['max_elev']), hd, rel


def obs_likelihood(x, y, theta, floor, observations, fmap, params=None):
    """파티클 N개의 관측 가중치 (N,). 관측이 없으면 모두 1.

    x, y, theta : (N,) 동·북 m, 나침반 도
    floor       : (N,) 정수 층, 또는 스칼라
    observations: [{type, bearing, distance(None 가능), score}, ...]  한 프레임 분
    fmap        : {floor: 지도 dict}
    """
    p = dict(DEFAULT, **(params or {}))
    x, y, theta = (np.asarray(a, float) for a in (x, y, theta))
    N = len(x)
    floor = np.broadcast_to(np.asarray(floor), (N,))
    logw = np.zeros(N)
    if not observations:
        return np.ones(N)
    for fl in np.unique(floor):
        sel = np.where(floor == fl)[0]
        fm = fmap.get(int(fl))
        if fm is None:                       # 지도 없는 층: 관측마다 바닥값
            logw[sel] += len(observations) * np.log(p['eps'])
            continue
        i, j, inside = _cells(fm, x[sel], y[sel])
        rows = {}       # 종류 묶음 → (행 번호, 후보, 수평 거리, 상대 방위). 관측에 나온 종류의 행만 계산(판 끝점·면 점이 많아서)
        for ob in observations:
            ks = tuple(kind_types(ob))
            if ks not in rows:
                k = np.where(np.isin(fm['lm_type'], ks))[0]
                rows[ks] = (k, *_kind_rows(fm, k, x[sel], y[sel], theta[sel], i, j, inside, p)) if len(k) else None
            if rows[ks] is None:
                logw[sel] += np.log(p['eps'])
                continue
            k, cand, hd, rel = rows[ks]
            e = ((ob['bearing'] - rel + 180) % 360 - 180) / p['sigma_bearing']
            lik = -0.5 * e ** 2
            if ob.get('distance') is not None:
                r = np.log(max(ob['distance'], 0.1) / np.maximum(hd, 0.1)) / p['sigma_dist']
                lik = lik - 0.5 * r ** 2
            lik = np.where(cand, np.exp(lik), 0.0).max(0)             # 비고유: 후보 중 최댓값
            s = float(ob.get('score', 1.0))
            logw[sel] += np.log(p['eps'] + (1 - p['eps']) * s * lik)
        if p['unknown'] == 'median':
            known = computed_at(fm, x[sel], y[sel])
            if known.any():
                logw[sel[~known]] = np.median(logw[sel[known]])
    return np.exp(logw - logw.max()) if N else logw   # 상대 가중치 (최댓값 1)
