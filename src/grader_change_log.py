"""Document the post-run grader fix: re-score pass-1 rows with the v1 rule (whole extracted answer,
<=15 words, v1 aliases from data/questions_v1_as_run.jsonl) and list every row whose label changed
under the v2 grader (first sentence only, <=40 words, 3 aliases added after the audit)."""
from __future__ import annotations
import json
import pandas as pd
from common import contains_phrase, extract_final_answer, load_config, normalize, p


def main():
    cfg = load_config()
    q1 = {}
    for l in open(p("data/questions_v1_as_run.jsonl"), encoding="utf-8"):
        d = json.loads(l); q1[d["question_id"]] = d
    df = pd.read_csv(p(cfg["paths"]["results_csv"]), dtype={"question_id": str})
    old = []
    for _, r in df.iterrows():
        ans, m = extract_final_answer(r.raw_response if isinstance(r.raw_response, str) else "")
        na = normalize(ans); al = q1[r.question_id]["normalized_aliases"]
        ok = bool(na) and any(na == a or contains_phrase(na, a) for a in al) and len(na.split()) <= (40 if m == "fallback_long" else 15)
        old.append(int(ok))
    df["correct_v1"] = old
    ch = df[df.correct_v1 != df.correct]
    reason = []
    for _, r in ch.iterrows():
        if r.question_id in ("K09", "S22"): reason.append("ambiguous item: alias added")
        elif r.question_id == "H29" and "halvdan" in str(r.model_answer).lower(): reason.append("alias added (full name)")
        else: reason.append("verbose final answer: first-sentence rule")
    ch = ch.assign(reason=reason)[["model", "question_id", "prompt_style", "gold_answer", "model_answer", "correct_v1", "correct", "reason"]]
    ch.to_csv(p(cfg["paths"]["tables_dir"]) / "grader_v1_v2_changes.csv", index=False)
    s = df.groupby("model").agg(acc_v1=("correct_v1", "mean"), acc_v2=("correct", "mean"), n_changed=("correct", lambda x: 0)).reset_index()
    s["n_changed"] = s.model.map(ch.groupby("model").size()).fillna(0).astype(int)
    s.to_csv(p(cfg["paths"]["tables_dir"]) / "grader_v1_v2_summary.csv", index=False)
    print(s.to_string(index=False)); print(ch.reason.value_counts().to_string())


if __name__ == "__main__":
    main()
