"""Grade raw responses and write results/results.csv with the agreed columns:

  question_id, model, prompt_style, model_answer, correct, consistent, hallucination, error_type

plus helper columns (domain, tier, gold_answer, extraction, latency_s, output_tokens, raw_response).

Automatic error_type (one per row):
  none        correct answer
  refusal     model declines / says it does not know
  format      no usable answer could be extracted (empty output, or output cut off at the
              token cap before any answer). Rows answered only via the 'fallback_long' rule
              (no "Final answer:" line) are graded normally but flagged in `extraction`, and a
              robustness check re-scores them as format failures.
  incomplete  answer is a strict, non-matching part of a gold alias (e.g. "George" for "George III")
  wrong_fact  a committed answer that matches no gold alias
"invented_detail" (false claims inside the reasoning even when the final answer is right) can
only be coded by a person; see the manual coding sheet written by this script.

hallucination = 1 when the row commits to a false answer (wrong_fact or incomplete); refusals
and format failures are NOT hallucinations (they are reported separately).
consistent = 1 when all four styles give the same answer for that (model, question).
Needs: pandas (analysis only; querying does not).
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from common import contains_phrase, extract_final_answer, load_config, normalize, p, read_jsonl

REFUSAL = re.compile(
    r"\b(i\s*(do\s*not|don'?t)\s*know|i'?m\s+not\s+(sure|certain)|i\s+am\s+not\s+(sure|certain)|"
    r"(can'?t|cannot|unable\s+to)\s+(answer|determine|provide|confirm|verify|be\s+sure)|"
    r"no\s+(reliable\s+)?information|not\s+enough\s+information)\b", re.I)


def answer_key(label: str, ans_norm: str) -> str:
    return {"correct": "<GOLD>", "refusal": "<REFUSAL>", "format": "<NOANSWER>"}.get(label, ans_norm)


def same_answer(k1: str, k2: str) -> bool:
    if k1 == k2:
        return True
    if k1.startswith("<") or k2.startswith("<") or not k1 or not k2:
        return False
    return contains_phrase(k1, k2) or contains_phrase(k2, k1)   # "canberra" ~ "canberra australia"


_ABBR = re.compile(r"(\b[A-Z]\.|\bSt\.|\bDr\.|\bMr\.|\bMt\.|\bNo\.|\bc\.)$")


def first_sentence(text: str) -> str:
    """First sentence of an extracted answer. Audit fix (v2): some models put a whole
    sentence (or several) after 'Final answer:', e.g. 'The deepest lake in the world is Lake
    Baikal in Russia, with a maximum depth of ...'. We grade the first sentence only, so extra
    trailing facts neither earn nor cost credit. Initials ('J. K.') do not end a sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = parts[0] if parts else ""
    for nxt in parts[1:]:
        if _ABBR.search(out):
            out += " " + nxt
        else:
            break
    return out


def grade_row(r: dict, q: dict) -> dict:
    ans, method = extract_final_answer(r["response"])
    na = normalize(first_sentence(ans))
    aliases = q["normalized_aliases"]
    hit = bool(na) and any(na == a or contains_phrase(na, a) for a in aliases)
    truncated = r.get("done_reason") == "length" and method != "final_line"
    max_words = 40   # v2: first sentence only, so a longer limit is safe
    if hit and len(na.split()) <= max_words:
        et = "none"
    elif REFUSAL.search(ans or r["response"]) and not hit:
        et = "refusal"
    elif method == "none" or truncated or not na:
        et = "format"
    elif any(set(na.split()) < set(a.split()) for a in aliases):
        et = "incomplete"
    else:
        et = "wrong_fact"
    return {"model_answer": ans, "extraction": method, "error_type": et,
            "correct": int(et == "none"), "hallucination": int(et in ("wrong_fact", "incomplete")),
            "_key": answer_key({"none": "correct", "refusal": "refusal", "format": "format"}.get(et, "x"), na)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    qs = {q["question_id"]: q for q in read_jsonl(p(cfg["data"]["questions_file"]))}
    models = args.models or cfg["models"]
    rows = []
    for m in models:
        f = p(cfg["paths"]["raw_dir"]) / f"{m.replace(':', '_').replace('/', '_')}{args.tag}.jsonl"
        for r in read_jsonl(f):
            q = qs[r["question_id"]]
            g = grade_row(r, q)
            rows.append({"question_id": r["question_id"], "model": r["model"], "prompt_style": r["prompt_style"],
                         **g, "domain": q["domain"], "tier": q["tier"], "gold_answer": q["answer"],
                         "latency_s": r.get("latency_s"), "output_tokens": r.get("output_tokens"),
                         "raw_response": r["response"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No raw results found. Run src/run_queries.py first.")

    n_styles = len(cfg["prompt_styles"])
    cons = {}
    for (m, qid), g in df.groupby(["model", "question_id"]):
        keys = g["_key"].tolist()
        cons[(m, qid)] = int(len(keys) == n_styles and all(same_answer(a, b) for a in keys for b in keys))
    df["consistent"] = [cons[(m, q)] for m, q in zip(df.model, df.question_id)]
    df = df.drop(columns="_key")
    cols = ["question_id", "model", "prompt_style", "model_answer", "correct", "consistent",
            "hallucination", "error_type", "domain", "tier", "gold_answer", "extraction",
            "latency_s", "output_tokens", "raw_response"]
    out = p(cfg["paths"]["results_csv"]).with_name(f"results{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values(["model", "question_id", "prompt_style"]).to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows -> {out}")
    print(df.groupby(["model", "error_type"]).size().unstack(fill_value=0))

    # Manual coding sheet: stratified sample per model, all wrong rows first, then some correct
    # rows (correct-but-invented-detail is only visible to a human, mostly in CoT).
    k = cfg["stats"]["audit_n_per_model"]
    parts = []
    for m, g in df.groupby("model"):
        wrong = g[g.correct == 0]
        right = g[(g.correct == 1)]
        n_wrong = min(len(wrong), int(k * 0.67))
        parts.append(wrong.sample(n=n_wrong, random_state=cfg["seed"]))
        # correct rows: stratified by prompt style (equal share per style)
        n_right = min(len(right), k - n_wrong)
        styles = sorted(right.prompt_style.unique())
        for i, s in enumerate(styles):
            gs = right[right.prompt_style == s]
            n_s = n_right // len(styles) + (1 if i < n_right % len(styles) else 0)
            parts.append(gs.sample(n=min(len(gs), n_s), random_state=cfg["seed"] + i))
    aud = pd.concat(parts)[["model", "question_id", "prompt_style", "gold_answer", "raw_response",
                            "model_answer", "extraction", "correct", "error_type"]].copy()
    for c in ["coder1_error_type", "coder1_invented_detail(Y/N)", "coder2_error_type",
              "coder2_invented_detail(Y/N)", "final_error_type", "notes"]:
        aud[c] = ""
    a_path = p(cfg["paths"]["audit_dir"]) / f"manual_coding_sheet{args.tag}.csv"
    a_path.parent.mkdir(parents=True, exist_ok=True)
    aud.to_csv(a_path, index=False, encoding="utf-8")
    print(f"manual coding sheet ({len(aud)} rows) -> {a_path}")


if __name__ == "__main__":
    main()
