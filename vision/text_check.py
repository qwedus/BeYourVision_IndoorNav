"""글자 확인 결과 읽기·적용 (비전 코어, I1 연결안 3절 4-1, 11-04). 엔진과 분리한 자리: 재생은 결정 파일만 읽는다.

결정 파일 (JSONL, 첫 줄은 {"engine": ...}, 다음부터 관측 한 줄)
    detection_id, track_id, t(촬영 시각), label(검출 종류),
    decision      keep / reject / relabel
    map_types     relabel일 때 바꾼 지도 종류
    text_status   confirmed(그 종류 글자 읽음) / unread(뜻 있는 글자 못 읽음) / mismatch(다른 글자) / skipped(글자 확인 안 하는 종류)
    text, reason  판정에 쓴 글자·이유
엔진을 바꿀 때는 같은 형식 파일을 만들면 된다(EasyOCR은 text_check_easyocr.py가 만듦, torch가 필요해 localization_design에서 돌림).
못 읽은 관측: unread='keep'(기본, 넣음) / 'drop'(뺌). 버린 관측은 지우지 않고 dropped에 reason='text_check'로 남긴다.
결정이 없는 관측은 그대로 넣으므로 결정 파일 ID가 관측과 안 이어지면 글자 확인이 조용히 꺼진다 → 연결률을 알리고, MIN_LINK 아래면 멈춘다(link_stats).

판정 규칙(decide 등)은 11-02 OCR 필터의 ocr_filter.py를 그대로 옮긴 것(완화 규칙 loose 포함, 기본 켬 = 그 보고의 기본안).
sure=True(11-04 비교용 추가): 다른 종류의 글자 없이 "뜻 있는 글자"라는 이유만으로는 버리지 않고, 표지판 글자와 안 닮은 한글(ad_like)일 때만 버림.
    I1 녹화에서 흐린 진짜 판("티는곳", "가능곳", "Thvalnblor" = This train is bound for)이 원래 규칙으로 버려져서.
track_decide는 yolo_ocr_live.py의 판 단위 처리와 같음: 앞 프레임만 봄, 최근 5프레임 중 3번 잡힌 판만 읽음, 못 읽으면 gap초 간격으로 최대 max_reads번,
판정이 나면 그 뒤로 유지. 표준 라이브러리만 쓴다(vision-integration 브랜치 vision/에 그대로 복사됨 — 원본은 localization_design).
"""
import difflib
import json
import multiprocessing
import re
from collections import defaultdict
from pathlib import Path

MIN_LINK = 0.5      # 이어진 수 ÷ min(관측 수, 결정 수)가 이보다 낮으면 다른 기록의 결정 파일로 보고 멈춤
TEXT_CLASSES = ("exit", "navy", "ticket_sign", "monitor", "map", "area_map")
MAP_TYPES = {"exit": ["exit_sign"], "navy": ["track_sign", "direction_sign"], "ticket_sign": ["ticket_sign"],
             "monitor": ["info_monitor"], "map": ["station_map", "route_map"], "area_map": ["area_map"]}
STATIONS = ("원인재", "인천시청", "부평", "계양", "동막", "송도", "박촌", "귤현", "작전", "갈산", "부평구청",
            "동수", "간석오거리", "예술회관", "인천터미널", "문학경기장", "선학", "신연수", "캠퍼스타운",
            "테크노파크", "지식정보단지", "인천대입구", "센트럴파크", "국제업무지구", "검단", "임학", "경인교대",
            "검단호수공원", "호수공원", "아라", "신검단")
SIGN_WORDS = ("나가는 곳", "Way Out", "타는 곳", "Tracks", "표 사는 곳", "Tickets", "종합안내도", "노선도", "안내도",
              "주변지역", "전역출발", "열차", "도착", "출발")   # 역 이름은 짧아 우연히 닮기 쉬워 정확히 맞을 때만 인정(station_in)


def norm(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def similar(target, text):
    t, s = norm(target), norm(text)
    if not t or not s:
        return 0.0
    if t in s:
        return 1.0
    best = difflib.SequenceMatcher(None, t, s).ratio()
    for k in range(max(1, len(s) - len(t) + 1)):
        best = max(best, difflib.SequenceMatcher(None, t, s[k:k + len(t)]).ratio())
    return best


def station_in(text):
    """역 이름 (앞뒤에 다른 한글이 붙은 '부평학원' 등은 제외, '계양행'의 '행'은 허용)."""
    tokens = re.split(r"\s+", text)
    if len(norm(text)) <= 4:
        tokens.append(text)
    for token in tokens:
        n = norm(token)
        for s in sorted(STATIONS, key=len, reverse=True):
            i = n.find(s)
            if i < 0:
                continue
            before = i > 0 and re.match(r"[가-힣]", n[i - 1])
            after = n[i + len(s):]
            if before or (after and re.match(r"[가-힣]", after[0]) and not after.startswith("행")):
                continue
            return s
    return None


def word(text, w):
    return re.search(rf"(?<![a-z]){w}(?![a-z])", text.lower()) is not None


def kinds_of(text, th=0.75):
    """글자 하나 → {종류: 강도}  강도 2 = 그 종류만의 글자, 1 = 여러 종류에 공통인 글자"""
    k = {}
    if max(similar("나가는 곳", text), similar("Way Out", text)) >= th or word(text, "exit") or "出口" in text:
        k["exit"] = 2
    if max(similar("타는 곳", text), similar("Tracks", text)) >= th:
        k["navy"] = 2
    if max(similar("표 사는 곳", text), similar("Tickets", text)) >= th or word(text, "ticket"):
        k["ticket_sign"] = 2
    if (word(text, "train") or any(w in text for w in ("전역출발", "전역", "열차", "접근", "출발", "도착"))
            or re.search(r"[가-힣]{2,}행\b", text) and station_in(text)):
        k["monitor"] = 2
    if similar("종합안내도", text) >= th or "노선도" in text or word(text, "route"):
        k["map"] = 2
    if similar("주변지역", text) >= th or word(text, "vicinity"):
        k["area_map"] = 2
    if "안내도" in text or word(text, "map") or word(text, "information"):
        k.setdefault("map", 1)
        k.setdefault("area_map", 1)
    if station_in(text):                       # 역 이름: 방면 판과 도착 안내 모니터에 공통
        k.setdefault("navy", 1)
        k.setdefault("monitor", 1)
    return k


def meaningful(text, conf):
    """뜻 있는 글자인지: 신뢰도 0.5 이상이고 글자 2개 이상."""
    return conf >= 0.5 and len(norm(text)) >= 2


def jamo(s):
    """한글을 자음·모음으로 풀기 ('티논곳' ≈ '타는곳'처럼 흐리게 읽힌 글자도 닮게 보이도록)."""
    out = []
    for ch in norm(s):
        o = ord(ch) - 0xAC00
        out.append(chr(0x1100 + o // 588) + chr(0x1161 + o % 588 // 28) + (chr(0x11A7 + o % 28) if o % 28 else "")
                   if 0 <= o < 11172 else ch)
    return "".join(out)


def ad_like(text, extra=()):
    """한글 3자 이상인데 어떤 표지판 글자와도 자음·모음 기준 절반 넘게 닮지 않음 → 광고 의심 (예: '점심사간', '키스슬')."""
    if len(re.findall(r"[가-힣]", text)) < 3 or station_in(text):
        return False
    s = jamo(text)

    def near(words):
        best = 0.0
        for w in words:
            t = jamo(w)
            for k in range(max(1, len(s) - len(t) + 1)):      # 긴 글자 안에서 표지판 글자 길이만큼 잘라 비교
                best = max(best, difflib.SequenceMatcher(None, t, s[k:k + len(t)]).ratio())
        return best
    # 표지판 글자는 절반만 닮아도, 역 이름(3자 이상)은 70% 이상 닮아야 표지판으로 봄 ('원인저' ≈ 원인재)
    return near(SIGN_WORDS + extra) < 0.5 and near([x for x in STATIONS if len(x) >= 3]) < 0.7


def decide(cls, reads, loose=True, sure=False):
    """reads: [(글자, 신뢰도)] → (decision, suggested_class, 맞은 글자, 이유). decision: keep / reject / unknown / skip"""
    if cls not in TEXT_CLASSES:
        return "skip", "", "", "글자로 확인하지 않는 종류"
    pieces = [(t.strip(), c) for t, c in reads if t.strip()]
    joined = " ".join(t for t, _ in pieces)
    score = defaultdict(int)
    hit = {}
    for t, c in pieces + ([(joined, 1.0)] if len(pieces) > 1 else []):
        for k, s in kinds_of(t).items():
            if s > score[k]:
                score[k], hit[k] = s, t
    if cls in score:
        return "keep", "", hit[cls], "종류와 맞는 글자"
    strong = [k for k, s in score.items() if s == 2]
    if strong:
        return "reject", strong[0], hit[strong[0]], "다른 종류의 글자"
    if "navy" in score:                       # 역 이름만 읽힘 → 방면 안내판일 가능성이 큼 (약한 근거)
        return "reject", "navy", hit["navy"], "역 이름이 읽힘 (방면 안내판일 가능성)"
    if not sure and any(meaningful(t, c) for t, c in pieces):
        return "reject", "", joined, "종류와 안 맞는 글자 (광고 등)"
    if loose and any(ad_like(t, ("이번열차",) if sure else ()) for t, _ in pieces):
        return "reject", "", joined, "표지판 글자와 안 닮은 한글 (광고 의심, 완화 규칙)"
    return "unknown", "", "", "뜻 있는 글자를 못 읽음"


def track_decide(obs_rows, reads, max_reads=3, gap=0.5, loose=True, sure=False, window=5, min_hits=3):
    """obs_rows: 시각 순 [{detection_id, track_id, t, label, frame}] (frame = 프레임 번호).
    reads: {detection_id: [(글자, 신뢰도), ...]} (읽을 수 있는 모든 관측). → 결정 행 목록. 판(track)마다 앞 프레임만 보고 상태를 갱신."""
    state, out = {}, []
    for r in obs_rows:
        st = state.setdefault(r["track_id"], dict(hits=[], n=0, last=-1e18, dec=None, to="", text="", why=""))
        st["hits"].append(r["frame"])
        confirmed = sum(h > r["frame"] - window for h in st["hits"]) >= min_hits
        if confirmed and r["label"] in TEXT_CLASSES and st["dec"] in (None, "unknown") and st["n"] < max_reads \
                and r["t"] - st["last"] >= gap - 1e-6:
            st["n"] += 1
            st["last"] = r["t"]
            st["dec"], st["to"], st["text"], st["why"] = decide(r["label"], reads.get(r["detection_id"], []), loose, sure)
        if r["label"] not in TEXT_CLASSES:
            status, decision = "skipped", "keep"
        elif st["dec"] == "keep":
            status, decision = "confirmed", "keep"
        elif st["dec"] == "reject":
            status, decision = "mismatch", "relabel" if st["to"] else "reject"
        else:
            status, decision = "unread", "keep"
        row = dict(detection_id=r["detection_id"], track_id=r["track_id"], t=r["t"], label=r["label"], decision=decision,
                   text_status=status, text=st["text"], reason=st["why"])
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


def link_stats(frames, decisions):
    """결정 파일과 관측이 얼마나 이어지나. 시간 필터 뒤 관측에 전체 관측용 결정을 쓰거나 그 반대면 한쪽 비율만 낮으므로 작은 쪽 기준 overlap으로 판단."""
    ids = {ob["detection_id"] for f in frames for ob in f.get("observations", [])}
    hit = len(ids & set(decisions))
    return dict(n_obs=len(ids), n_dec=len(decisions), linked=hit, obs_rate=hit / len(ids) if ids else 1.0,
                dec_rate=hit / len(decisions) if decisions else 1.0, overlap=hit / min(len(ids), len(decisions)) if ids and decisions else float(not ids))


def apply_decisions(frames, decisions, unread="keep", min_link=MIN_LINK):
    """cv_module 프레임들에 결정 적용(사본 반환). reject·(unread='drop'이면 못 읽음)은 dropped로, relabel은 map_types만 바꿈.
    결정이 없는 관측은 그대로 둠. 연결률이 min_link 아래면 ValueError."""
    st = link_stats(frames, decisions)
    if st["overlap"] < min_link:
        raise ValueError(f"글자 확인 결정이 관측과 안 이어짐: 관측 {st['n_obs']}개, 결정 {st['n_dec']}개 중 이어진 것 {st['linked']}개 "
                         f"(연결률 {st['overlap']:.0%} < {min_link:.0%}). 다른 기록의 결정 파일인지 확인할 것")
    if st["obs_rate"] < 1 and multiprocessing.parent_process() is None:
        print(f"글자 확인 연결률: 관측 {st['n_obs']}개 중 {st['linked']}개({st['obs_rate']:.0%}), 결정 {st['n_dec']}개 중 {st['dec_rate']:.0%}. 결정 없는 관측은 그대로 넣음")
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
