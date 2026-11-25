"""Verify the authored question bank against Wikipedia and freeze it.

For every question we download the plain text of its source article (pinned to the current
revision id) and look for ONE sentence that contains a gold alias AND every evidence keyword.
That sentence is stored as `evidence_sentence` so a human can confirm it in seconds.

Status values:
  auto_verified  - such a sentence was found
  needs_manual   - article fetched, but no single sentence matched (human must check / fix)
  fetch_failed   - article could not be retrieved
Automatic evidence is necessary but not sufficient: every row must also be confirmed by a
person in data/verification_report.csv (human_verified = Y) before the main run, and rows can
be overridden in data/manual_verification.csv (question_id, verified, source_url, note). Output: data/questions.jsonl
"""
from __future__ import annotations

import csv
import re
import time

import urllib.parse
import urllib.request
import json

from common import load_config, normalize, p, write_jsonl

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "prompt-sensitivity-research/0.1 (student research project; contact via repo)"}


def parse_bank(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        qid, dom, tier, q, ans, aliases, title, kws = [x.strip() for x in line.rstrip("\n").split("|")]
        rows.append({"question_id": qid, "domain": dom, "tier": tier, "question": q, "answer": ans,
                     "aliases": [a.strip() for a in aliases.split(";") if a.strip()],
                     "source_title": title, "evidence_keywords": [k.strip() for k in kws.split(";") if k.strip()]})
    return rows


def fetch(title):
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts|revisions", "explaintext": 1,
        "rvprop": "ids|timestamp", "redirects": 1, "titles": title})
    req = urllib.request.Request(f"{API}?{q}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        js = json.loads(r.read().decode("utf-8"))
    page = next(iter(js["query"]["pages"].values()))
    if "missing" in page:
        return None
    rev = page["revisions"][0]
    return {"title": page["title"], "text": page.get("extract", ""), "revid": rev["revid"],
            "rev_ts": rev["timestamp"]}


_ABBREV = re.compile(r"(\b[A-Z]\.|\bU\.S\.|\bSt\.|\bDr\.|\bMr\.|\bNo\.|\bc\.)$")


def sentences(text):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    merged = []
    for s in parts:  # re-join splits made after initials/abbreviations ("J. K. Rowling", "U.S.")
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged


def has(sentence_lc, phrase):
    ph = re.escape(phrase.lower())
    return re.search(rf"(?<![\w]){ph}(?![\w])", sentence_lc) is not None


def find_evidence(text, aliases, kws):
    for s in sentences(text):
        lc = s.lower()
        # keywords: every ;-item must match; "/" separates acceptable alternatives
        if any(has(lc, a) for a in aliases) and all(any(has(lc, alt) for alt in k.split("/")) for k in kws):
            return s
    # Fallback: lead sentences often use a pronoun ("It is the largest country..."), so allow
    # the alias in the sentence immediately before the keyword sentence. Flagged as a window match.
    ss = sentences(text)
    for a_s, k_s in zip(ss, ss[1:]):
        if any(has(a_s.lower(), a) for a in aliases) and \
           all(any(has(k_s.lower(), alt) for alt in k.split("/")) for k in kws):
            return "[2-sentence window] " + a_s + " " + k_s
    return None


def main():
    cfg = load_config()
    bank = parse_bank(p("data/question_bank_authored.psv"))
    ai = {}
    apath = p("data/ai_review.csv")
    if apath.exists():
        ai = {r["question_id"]: r for r in csv.DictReader(open(apath, encoding="utf-8"))}
    manual = {}
    mpath = p("data/manual_verification.csv")
    if mpath.exists():
        for r in csv.DictReader(open(mpath, encoding="utf-8")):
            manual[r["question_id"]] = r
    out, report = [], []
    for row in bank:
        try:
            page = fetch(row["source_title"])
        except Exception as e:  # network hiccup: record, don't crash
            page, err = None, str(e)
        if page is None:
            row.update(status="fetch_failed", source_url=None, evidence_sentence=None)
        else:
            # For finding evidence only (not for grading), a person's surname also counts, since
            # articles refer to people by surname ("Thomson", "Curie", "Lie").
            ev_aliases = row["aliases"] + ([row["answer"].split()[-1]] if len(row["answer"].split()) > 1 else [])
            ev = find_evidence(page["text"], ev_aliases, row["evidence_keywords"])
            row.update(
                source_url=f"https://en.wikipedia.org/w/index.php?title={page['title'].replace(' ', '_')}&oldid={page['revid']}",
                source_revision_timestamp=page["rev_ts"],
                evidence_sentence=ev, status="auto_verified" if ev else "needs_manual")
        m = manual.get(row["question_id"])
        if m and m.get("verified", "").strip().upper() == "Y":
            row["status"] = "manual_verified"
            row["manual_note"] = m.get("note", "")
            row["source_url"] = m.get("source_url") or row["source_url"]
        row["normalized_aliases"] = sorted({normalize(a) for a in row["aliases"]})
        out.append(row)
        report.append((row["question_id"], row["status"], (row.get("evidence_sentence") or "")[:160]))
        time.sleep(0.3)

    keep = [r for r in out if r["status"] in ("auto_verified", "manual_verified")]
    write_jsonl(p("data/questions_all_with_status.jsonl"), out)
    write_jsonl(p(cfg["data"]["questions_file"]), keep)
    with open(p("data/verification_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # human_verified / human_note are filled in by the student verifier (see design.md, 3.3)
        w.writerow(["question_id", "domain", "tier", "question", "answer", "aliases", "status", "source_url",
                    "evidence_sentence", "ai_verification", "ai_note", "human_verified(Y/N)", "human_note"])
        for r in out:
            a = ai.get(r["question_id"], {})
            w.writerow([r["question_id"], r["domain"], r["tier"], r["question"], r["answer"], "; ".join(r["aliases"]),
                        r["status"], r.get("source_url"), r.get("evidence_sentence"),
                        a.get("ai_verdict", ""), a.get("ai_note", ""), "", ""])
    from collections import Counter
    print(Counter(r["status"] for r in out))
    for qid, st, ev in report:
        if st != "auto_verified":
            print(qid, st)
    print(f"{len(keep)} questions frozen -> {cfg['data']['questions_file']}")


if __name__ == "__main__":
    main()
