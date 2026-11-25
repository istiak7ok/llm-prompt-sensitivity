"""Point estimates for every metric in design.md section 5, computed from results/results.csv.

Acc(m,s)   = mean correct                          per model m and prompt style s
H(m,s)     = mean hallucination (committed false answer)
R(m,s)     = refusal rate,  F(m,s) = format-failure rate   (Acc + H + R + F = 1)
CR(m)      = share of questions whose 4 answers are all the same
PS(m)      = max_s Acc(m,s) - min_s Acc(m,s)        (accuracy spread across styles)
SD(m)      = population std of Acc(m,s) over the 4 styles
QS(m)      = share of questions whose correctness differs across styles (per-question sensitivity)
v_q        = 4 * cbar_q * (1 - cbar_q), cbar_q = share of styles answering q correctly (0..1)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_config, p


def load(cfg, tag=""):
    return pd.read_csv(p(cfg["paths"]["results_csv"]).with_name(f"results{tag}.csv"))


def by_style(df):
    g = df.groupby(["model", "prompt_style"])
    t = g.agg(N=("correct", "size"), accuracy=("correct", "mean"), hallucination=("hallucination", "mean"))
    for et, name in [("refusal", "refusal_rate"), ("format", "format_failure_rate")]:
        t[name] = g["error_type"].apply(lambda s: (s == et).mean())
    t["halluc_given_answered"] = t.hallucination / (t.accuracy + t.hallucination).replace(0, np.nan)
    return t.reset_index()


def per_question(df):
    g = df.groupby(["model", "question_id"])
    q = g.agg(cbar=("correct", "mean"), consistent=("consistent", "first"),
              domain=("domain", "first"), tier=("tier", "first")).reset_index()
    q["v_q"] = 4 * q.cbar * (1 - q.cbar)
    q["sensitive"] = ((q.cbar > 0) & (q.cbar < 1)).astype(int)
    return q


def by_model(df):
    bs, pq = by_style(df), per_question(df)
    rows = []
    for m in bs.model.unique():
        a = bs[bs.model == m].accuracy
        qm = pq[pq.model == m]
        d = df[df.model == m]
        rows.append({"model": m, "N_questions": len(qm), "accuracy_mean": d.correct.mean(),
                     "hallucination_mean": d.hallucination.mean(),
                     "consistency_rate": qm.consistent.mean(),
                     "consistent_and_correct": (qm.cbar == 1).mean(),
                     "sensitivity_spread_PS": a.max() - a.min(), "sensitivity_sd": a.std(ddof=0),
                     "question_sensitivity_QS": qm.sensitive.mean(), "mean_v_q": qm.v_q.mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    df = load(cfg, args.tag)
    out = p(cfg["paths"]["tables_dir"]); out.mkdir(parents=True, exist_ok=True)
    bs, bm, pq = by_style(df), by_model(df), per_question(df)
    et = df.groupby(["model", "prompt_style", "error_type"]).size().unstack(fill_value=0).reset_index()
    dom = df.groupby(["model", "domain"]).agg(accuracy=("correct", "mean"), hallucination=("hallucination", "mean")).reset_index()
    tier = df.groupby(["model", "tier"]).agg(accuracy=("correct", "mean"), hallucination=("hallucination", "mean")).reset_index()
    for name, t in [("metrics_by_style", bs), ("metrics_by_model", bm), ("per_question", pq),
                    ("error_types", et), ("by_domain", dom), ("by_tier", tier)]:
        t.to_csv(out / f"{name}{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(bs.round(3).to_string(index=False)); print(); print(bm.round(3).to_string(index=False))
    print(); print(et.to_string(index=False))


if __name__ == "__main__":
    main()
