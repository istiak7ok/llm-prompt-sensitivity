"""Record the assistant's (coder 1) audit of the automatic grading.

Coder 1 is the AI assistant that drafted this study, reading each sampled row (gold answer,
extracted answer and full raw response). Rows where coder 1 agrees with the automatic label get
the automatic error_type; disagreements and judgement calls are listed in OVERRIDES / NOTES below
with a reason. A human second coder (coder 2) is still pending; kappa.py reports coder1-vs-coder2
kappa once that column is filled in. invented_detail is NOT coded by the assistant (left blank).

Input : results/audit/manual_coding_sheet.csv (written by grade.py, v2 grader)
Output: results/audit/manual_coding_sheet_coded.csv
"""
from __future__ import annotations

import pandas as pd

from common import load_config, p

# (model, question_id, prompt_style) -> coder-1 error_type, where it differs from the auto label
OVERRIDES = {
    ("qwen2.5:7b", "K21", "polite"): ("wrong_fact",
        "Cut off at the token cap with no Final answer line, but the text already commits to false "
        "claims ('theoretically unlimited', 'maximum break would be 75 points'). Coder 1 codes "
        "wrong_fact; auto code is format."),
}
NOTES = {
    "G24": "Answer 'Nur-Sultan' is the 2019-2022 name; accepted because it was a pre-registered alias. "
           "Outdated-name sensitivity reported separately.",
    "K09": "Item flagged ambiguous (64 cells vs 204 squares of all sizes); both accepted.",
    "S22": "Item flagged ambiguous (author vs publisher John Murray); both accepted.",
}


def main():
    cfg = load_config()
    d = pd.read_csv(p(cfg["paths"]["audit_dir"]) / "manual_coding_sheet.csv", dtype=str, keep_default_na=False)
    c1, notes = [], []
    for _, r in d.iterrows():
        key = (r.model, r.question_id, r.prompt_style)
        if key in OVERRIDES:
            c1.append(OVERRIDES[key][0]); notes.append(OVERRIDES[key][1])
        else:
            c1.append(r.error_type)
            n = NOTES.get(r.question_id, "")
            if r.question_id == "G24" and "nur" not in r.model_answer.lower():
                n = ""
            notes.append(n)
    d["coder1_error_type"] = c1
    d["coder1_correct"] = [ "1" if e == "none" else "0" for e in c1]
    d["notes"] = notes
    d["coder2_error_type"] = ""          # human second coder pending
    out = p(cfg["paths"]["audit_dir"]) / "manual_coding_sheet_coded.csv"
    d.to_csv(out, index=False, encoding="utf-8")
    from sklearn.metrics import cohen_kappa_score
    agree_c = (d.correct == d.coder1_correct).mean()
    agree_e = (d.error_type == d.coder1_error_type).mean()
    k_e = cohen_kappa_score(d.error_type, d.coder1_error_type)
    k_c = cohen_kappa_score(d.correct, d.coder1_correct) if d.coder1_correct.nunique() > 1 else float("nan")
    summ = pd.DataFrame([{"n_rows": len(d), "n_auto_wrong": int((d.correct == "0").sum()),
                          "agree_correct": round(agree_c, 4), "kappa_correct": round(k_c, 4),
                          "agree_error_type": round(agree_e, 4), "kappa_error_type": round(k_e, 4)}])
    by_model = d.assign(ac=(d.correct == d.coder1_correct), ae=(d.error_type == d.coder1_error_type)) \
        .groupby("model").agg(n=("ac", "size"), agree_correct=("ac", "mean"), agree_error_type=("ae", "mean")).reset_index()
    summ.to_csv(p(cfg["paths"]["tables_dir"]) / "audit_agreement.csv", index=False)
    by_model.to_csv(p(cfg["paths"]["tables_dir"]) / "audit_agreement_by_model.csv", index=False)
    print(summ.to_string(index=False)); print(by_model.to_string(index=False)); print("->", out)


if __name__ == "__main__":
    main()
