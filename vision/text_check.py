"""글자 확인 결과 읽기·적용 (비전 코어, I1 연결안 3절 4-1, 11-04). 엔진과 분리한 자리: 재생은 결정 파일만 읽는다.

결정 파일 (JSONL, 관측 한 줄)
    detection_id, track_id, t(촬영 시각), label(검출 종류),
    decision      keep / reject / relabel
    map_types     relabel일 때 바꾼 지도 종류
    text_status   confirmed(그 종류 글자 읽음) / unread(뜻 있는 글자 못 읽음) / mismatch(다른 글자) / skipped(글자 확인 안 하는 종류)
    text, conf    판정에 쓴 글자·신뢰도
    engine        엔진 이름·버전 (easyocr 1.7.2 / 나중에 apple_vision)
엔진을 바꿀 때는 같은 형식 파일을 만들면 된다(EasyOCR은 text_check_easyocr.py가 만듦, torch가 필요해 localization_design에서 돌림).
못 읽은 관측: unread='keep'(기본, 넣음) / 'drop'(뺌). 버린 관측은 지우지 않고 dropped에 reason='text_check'로 남긴다.

판정 규칙(judge): 11-02 OCR 필터 보고의 기본안 = 다른 글자가 확실할 때만 버림 + 종류 바로잡기, 못 읽음은 남김.
    '확실히 다른 글자' = 다른 종류의 글자가 읽힘(종류 바로잡기) 또는 표지판 글자와 전혀 안 닮은 한글 3자 이상(광고 등, 자음·모음 단위 비교).
    표지판 글자와 닮았으면("나가논곳", "주는곳" = 흐린 "표 사는 곳") 못 읽음으로 봄. 깨진 영문 조각("isfor")으로는 버리지 않음.
track_decide: 판 하나를
앞 프레임만 보고 판정(뒤 프레임 안 봄). 판마다 최대 max_reads번 gap초 간격으로 읽고, 확인·다른 글자가 나오면 그 뒤로 판정을 유지.
표준 라이브러리만 쓴다(vision-integration 브랜치 vision/에 그대로 복사됨 — 원본은 localization_design).
"""
import json
import re
import unicodedata
from pathlib import Path

CONF_MIN = 0.5
SKIP = {"toilet", "ticket_machine", "gate", "door", "station_name"}     # 그림 위주·글자 확인 안 하는 종류
MAP_TYPES = {"exit": ["exit_sign"], "navy": ["track_sign", "direction_sign"], "ticket_sign": ["ticket_sign"],
             "monitor": ["info_monitor"], "map": ["station_map", "route_map"], "area_map": ["area_map"]}
STATIONS = "원인재|인천시청|부평|계양|동막|송도|검단호수공원|검단|동춘"
# 종류별 글자. 공백을 없앤 소문자 문자열에 찾는다. 역 이름은 앞뒤에 다른 한글이 붙으면 인정 안 함('행'·'역'·'방면'은 허용)
WORDS = {
    "exit": re.compile(r"나가는곳|wayout|exit|出口"),
    "navy": re.compile(rf"타는곳|tracks|(?<![가-힣])(?:{STATIONS})(?:행|역|방면)?(?![가-힣])"),
    "ticket_sign": re.compile(r"표사는곳|tickets"),
    "monitor": re.compile(r"thistrain|전역출발|열차|출발|도착|[가-힣]{2,}행(?![가-힣])"),
    "map": re.compile(r"종합안내도|노선도|안내도|map|information"),
    "area_map": re.compile(r"주변지역|vicinity|안내도|map"),
}


# 흐리게 읽힌 글자를 알아보려고 비교하는 표지판 글자들
SIGN_WORDS = ["나가는곳", "타는곳", "표사는곳", "종합안내도", "노선도", "안내도", "주변지역", "역이용", "이번열차", "전역출발", "열차", "출발", "도착",
              "원인재", "인천시청", "부평", "계양", "동막", "송도달빛축제공원", "검단호수공원", "동춘"]
NEAR = 0.34      # 자음·모음 편집 거리 ÷ 표지판 글자 자음·모음 수


def _jamo(s):
    return [c for c in unicodedata.normalize("NFD", s)]


def _dist(a, b):
    """b 전체가 a의 일부와 얼마나 다른가 (a 앞뒤 글자는 공짜로 건너뜀)"""
    prev = [0] * (len(a) + 1)
    for j, y in enumerate(b, 1):
        cur = [j]
        for i, x in enumerate(a, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (x != y)))
        prev = cur
    return min(prev)


def near_sign(s):
    """표지판 글자와 닮은 글자인가: 어떤 표지판 글자의 일부(앞뒤로 잘린 조각)여도 닮은 것으로 봄"""
    a = _jamo(s)
    return any(_dist(a, w) / len(w) <= NEAR or _dist(w, a) / max(len(a), 1) <= NEAR for w in map(_jamo, SIGN_WORDS))


def foreign_text(s):
    """표지판 글자와 전혀 안 닮은 한글 3자 이상"""
    return len(re.findall(r"[가-힣]", s)) >= 3 and not near_sign(s)


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s)).lower()


def meaningful(s):
    """글자(한글·영문) 2개 이상. 숫자·기호 조각은 판정에 안 씀."""
    return len(re.findall(r"[가-힣a-z]", s)) >= 2


def judge(label, texts):
    """읽은 조각 [(글자, 신뢰도)] → (text_status, 바로잡을 종류 또는 None, 근거 글자, 신뢰도).
    신뢰도 CONF_MIN 미만이거나 뜻 없는 조각은 안 씀. 조각 안 공백만 지우고 조각끼리 잇지 않음(옆 글자와 붙어 역 이름이 깨지지 않게)."""
    if label in SKIP or label not in WORDS:
        return "skipped", None, "", 0.0
    good = [(norm(t), c) for t, c in texts if c >= CONF_MIN and meaningful(norm(t))]
    if not good:
        return "unread", None, "", 0.0
    for s, c in good:
        if WORDS[label].search(s):
            return "confirmed", None, s, c
    best = None
    for s, c in good:
        for cls, rx in WORDS.items():
            if cls != label and rx.search(s) and (best is None or c > best[3]):
                best = ("mismatch", cls, s, c)
    if best:
        return best
    foreign = [(s, c) for s, c in good if foreign_text(s)]
    if foreign:
        s, c = max(foreign, key=lambda x: x[1])
        return "mismatch", None, s, c
    return "unread", None, "", 0.0


def track_decide(obs_rows, reads, max_reads=3, gap=0.5):
    """obs_rows: 시각 순 [{detection_id, track_id, t, label}]. reads: {detection_id: [(글자, 신뢰도), ...]} (읽을 수 있는 모든 관측).
    → 결정 행 목록. 판(track)마다 앞 프레임만 보고 상태를 갱신한다."""
    state, out = {}, []
    for r in obs_rows:
        st = state.setdefault(r["track_id"], dict(n=0, last=-1e18, status="unread", to=None, text="", conf=0.0))
        decided = st["status"] in ("confirmed", "mismatch", "skipped")
        if not decided and st["n"] < max_reads and r["t"] - st["last"] >= gap - 1e-6:
            st["n"] += 1
            st["last"] = r["t"]
            st["status"], st["to"], st["text"], st["conf"] = judge(r["label"], reads.get(r["detection_id"], []))
        if st["status"] == "mismatch":
            decision = "relabel" if st["to"] else "reject"
        else:
            decision = "keep"
        row = dict(detection_id=r["detection_id"], track_id=r["track_id"], t=r["t"], label=r["label"], decision=decision,
                   text_status=st["status"], text=st["text"], conf=round(st["conf"], 3))
        if decision == "relabel":
            row["map_types"] = MAP_TYPES[st["to"]]
        out.append(row)
    return out


def load_decisions(path):
    """→ ({detection_id: 행}, 엔진). 첫 줄은 {"engine": ...} 머리."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:] if x.strip()]
    return {r["detection_id"]: r for r in rows}, head["engine"]


def apply_decisions(frames, decisions, unread="keep"):
    """cv_module 프레임들에 결정 적용(사본 반환). reject·(unread='drop'이면 못 읽음)은 dropped로, relabel은 map_types만 바꿈."""
    out = []
    for f in frames:
        f = json.loads(json.dumps(f))
        keep, drop = [], []
        for ob in f.get("observations", []):
            d = decisions.get(ob["detection_id"])
            if d is None:
                keep.append(ob)
            elif d["decision"] == "reject" or (unread == "drop" and d["text_status"] == "unread"):
                drop.append(ob)
            else:
                if d["decision"] == "relabel":
                    ob["map_types"] = d["map_types"]
                ob["text_status"] = d["text_status"]
                keep.append(ob)
        f["observations"] = keep
        f.setdefault("dropped", []).extend(
            {"detection_id": o["detection_id"], "label": o["label"], "score": o["score"], "reason": "text_check"} for o in drop)
        out.append(f)
    return out
