"""앱 OCR '다른 글자' 판정을 PC에서 다른 규칙으로 다시 판정해 결정 파일을 만든다 (exp_ocr.py 짝, 기본 설정 안 바꿈, 채택 실험 아님).

    I1_RECORDING=<기록> .venv/Scripts/python integration/exp_ocr_rules.py
    .venv/Scripts/python integration/exp_ocr_rules.py --all
→ <입력 폴더>/text_check_applevision_<규칙>.jsonl, <녹화 out>/av_rules.json

앱이 그 시각 'other_text'(다른 글자)였던 검출만 다시 판정함. 확인·못 읽음·대기·비대상 행은 기본 결정 파일(text_check_applevision.jsonl)과 같음.
판정에 쓰는 글자 = 그 판(앱 track)에서 그 시각까지 마지막 other_text 읽기의 text(앱 ocr.jsonl). 앱은 other_text 뒤로 다시 읽지 않으므로
새 규칙에서 '넣음'이 되어도 다시 읽기는 없음(그 뒤로 계속 넣음, text_status = unread).
종류 글자 = 앱 config ocr.keywords[종류](fuzzy·words·contains) + common_keywords에서 그 종류에 붙은 words·contains (역 이름 목록은 뺌).
    a1   종류 글자의 한글 음절 집합과 읽은 글자의 한글 음절이 하나라도 겹치면 넣음(원래 종류 그대로). 안 겹치면 앱 판정 그대로(버림 또는 앱 바로잡기)
    a1d  앱이 바꾼 종류 T가 있고, 읽은 글자에 T만의 글자(T 한글 음절 − 원래 종류 한글 음절, 또는 T의 영어 낱말)가 있으면 바로잡기.
         아니면 원래 종류와 한글 음절이 겹치면 넣음, 안 겹치면 버림 (공통 글자 '는·곳'으로는 바로잡지 않음)
    a2   바로잡기는 a1d와 같음. 그 밖에는 판이 아니라는 근거가 있을 때만 버림(글자 겹침과 무관): 광고 의심 = PC text_check.ad_like
         (한글 3자 이상인데 표지판 글자·역 이름과 안 닮음, 합친 글자와 ' | ' 조각마다 검사). 앱 완화 규칙과 취지는 같으나 같은 함수는 아님
         (PC = SequenceMatcher + 고정 낱말 목록, 앱 = LCS + 녹화 설정 목록)
규칙은 결과를 보기 전에 정한 것(사용자 제안 a1 + 바로잡기 조건 a1d + 대안 a2). 값을 고르는 단계 없음.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

import paths
from exp_ocr import DEC_MAIN, S2_APP, jl, sha
from text_check import ad_like

RULES = ("a1", "a1d", "a2")
HANGUL = re.compile(r"[가-힣]")


def kind_words(cfg):
    """종류 → (한글 음절 집합, 영어 낱말 목록)"""
    o = cfg["ocr"]
    words = {k: [*v.get("fuzzy", []), *v.get("words", []), *v.get("contains", [])] for k, v in o["keywords"].items()}
    for c in o.get("common_keywords", []):
        for lab in c["labels"]:
            words.setdefault(lab, []).extend([*c.get("words", []), *c.get("contains", [])])
    return {k: (set(HANGUL.findall("".join(v))), [w.lower() for w in v if re.fullmatch(r"[A-Za-z ]+", w)]) for k, v in words.items()}


def has_target(text, target, source, kw):
    """text에 target만의 글자가 있나: target 한글 음절 − source 한글 음절, 또는 target 영어 낱말"""
    hs = set(HANGUL.findall(text))
    only = kw[target][0] - kw.get(source, (set(), []))[0]
    low = text.lower()
    return bool(hs & only) or any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low) for w in kw[target][1])


def overlap(text, source, kw):
    return bool(set(HANGUL.findall(text)) & kw.get(source, (set(), []))[0])


def judge(rule, text, source, app_dec, app_to, kw):
    """→ (decision, 바꿀 종류 또는 None, 이유)"""
    if rule == "a1":
        if overlap(text, source, kw):
            return "keep", None, "원래 종류와 한글 음절 겹침"
        return app_dec, app_to, "앱 판정 그대로"
    if app_to and has_target(text, app_to, source, kw):
        return "relabel", app_to, "바뀐 종류만의 글자"
    if rule == "a2":
        if any(ad_like(t) for t in [text, *text.split(" | ")]):
            return "reject", None, "광고 의심(ad_like)"
        return "keep", None, "판이 아니라는 근거 없음"
    if overlap(text, source, kw):
        return "keep", None, "원래 종류와 한글 음절 겹침"
    return "reject", None, "겹치는 글자 없음"


def one():
    cfg = json.loads((paths.REC / "config.json").read_text(encoding="utf-8"))
    kw = kind_words(cfg)
    lm = cfg["label_map"]
    lines = (paths.I1 / DEC_MAIN).read_text(encoding="utf-8").splitlines()
    head, rows = json.loads(lines[0]), [json.loads(x) for x in lines[1:] if x.strip()]
    reads = {}
    for x in jl(paths.REC / "ocr.jsonl"):
        if x.get("verdict") == "other_text":
            reads.setdefault(x["track_id"], []).append(x)
    tf_ids = {o["detection_id"] for f in jl(paths.I1 / "obs_timefilter.jsonl") for o in f["observations"]}
    info, per_row = {}, []
    out = {r: [] for r in RULES}
    for d in rows:
        if d["app_state"] != "other_text":
            for r in RULES:
                out[r].append(d)
            continue
        prev = [x for x in reads.get(d["track_id"], []) if x["t"] <= d["t"] + 1e-9]
        text = prev[-1]["text"] if prev else d.get("text", "")
        app_to = d.get("relabel_to")
        rec = dict(id=d["detection_id"], track=d["track_id"], t=d["t"], label=d["label"], text=text, app=d["decision"], app_to=app_to,
                   in_tf=d["detection_id"] in tf_ids)
        for r in RULES:
            dec, to, why = judge(r, text, d["label"], d["decision"], app_to, kw)
            row = {k: v for k, v in d.items() if k not in ("map_types", "relabel_to")}
            row.update(decision=dec, text_status="mismatch" if dec != "keep" else "unread", reason=f"{r}: {why}", text=text)
            if dec == "relabel":
                row.update(map_types=lm[to], relabel_to=to)
            out[r].append(row)
            rec[r] = dec if dec != "relabel" else f"relabel:{to}"
        per_row.append(rec)
    for r in RULES:
        h = dict(head, variant=r, rule_doc="exp_ocr_rules.py")
        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in [h, *out[r]]) + "\n"
        p = paths.I1 / f"text_check_applevision_{r}.jsonl"
        if p.exists() and p.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{p.name}가 이미 있고 내용이 다름")
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        tf = [x for x in per_row if x["in_tf"]]
        info[r] = dict(sha256=sha(p), tf=dict(Counter(f"{x['app']}->{x[r]}" for x in tf)))
    (paths.REC_OUT / "av_rules.json").write_text(json.dumps(dict(kind_hangul={k: "".join(sorted(v[0])) for k, v in kw.items()}, rules=info, rows=per_row),
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{paths.RECORDING}: " + "; ".join(f"{r} {info[r]['tf']}" for r in RULES), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not a.all:
        return one()
    bad = 0
    for rec in S2_APP:
        p = subprocess.run([sys.executable, __file__], env={**os.environ, "I1_RECORDING": rec, "PYTHONIOENCODING": "utf-8"},
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(f"{rec}: 실패 (코드 {p.returncode})\n{p.stderr[-1500:]}", flush=True)
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
