"""Uncertainty and significance (design.md section 6).

* 95% percentile bootstrap CIs (B resamples of the question set, fixed seed) for
  accuracy / hallucination per (model, style), and for consistency rate and the accuracy
  spread PS per model (the spread is recomputed inside every resample).
* Cochran's Q per model: does correctness differ across the 4 prompt styles?
* Exact McNemar for all 6 style pairs per model, Holm-corrected within each model.
* Exact McNemar on hallucination: simple vs CoT per model (does CoT change false answers?).
Writes CSV + Markdown + LaTeX tables to results/tables/.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from common import load_config, p
from metrics import load


def pct(m, lo, hi):
    return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def mcnemar_exact(a: pd.Series, b: pd.Series):
    tab = pd.crosstab(a.astype(bool), b.astype(bool)).reindex(index=[False, True], columns=[False, True], fill_value=0).values
    return mcnemar(tab, exact=True).pvalue, int(tab[1, 0]), int(tab[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    B, alpha, styles = cfg["stats"]["bootstrap_B"], cfg["stats"]["alpha"], cfg["prompt_styles"]
    df = load(cfg, args.tag)
    out = p(cfg["paths"]["tables_dir"]); out.mkdir(parents=True, exist_ok=True)
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    ci_rows, model_rows, tests = [], [], []
    for m, d in df.groupby("model"):
        rng = np.random.default_rng(cfg["seed"])
        wc = d.pivot(index="question_id", columns="prompt_style", values="correct")[styles].dropna()
        wh = d.pivot(index="question_id", columns="prompt_style", values="hallucination")[styles].loc[wc.index]
        cons = d.groupby("question_id")["consistent"].first().loc[wc.index].values
        C, H = wc.values.astype(float), wh.values.astype(float)
        n = len(C)
        idx = rng.integers(0, n, size=(B, n))
        accB = C[idx].mean(axis=1)            # B x styles
        halB = H[idx].mean(axis=1)
        for j, s in enumerate(styles):
            ci_rows.append({"model": m, "prompt_style": s, "metric": "accuracy", "mean": C[:, j].mean(),
                            "lo": np.quantile(accB[:, j], lo_q), "hi": np.quantile(accB[:, j], hi_q)})
            ci_rows.append({"model": m, "prompt_style": s, "metric": "hallucination", "mean": H[:, j].mean(),
                            "lo": np.quantile(halB[:, j], lo_q), "hi": np.quantile(halB[:, j], hi_q)})
        consB = cons[idx].mean(axis=1)
        psB = accB.max(axis=1) - accB.min(axis=1)
        ps = C.mean(axis=0).max() - C.mean(axis=0).min()
        model_rows += [
            {"model": m, "metric": "consistency_rate", "mean": cons.mean(), "lo": np.quantile(consB, lo_q), "hi": np.quantile(consB, hi_q)},
            {"model": m, "metric": "sensitivity_spread_PS", "mean": ps, "lo": np.quantile(psB, lo_q), "hi": np.quantile(psB, hi_q)},
        ]
        if n > 1 and C.shape[1] >= 3:
            q = cochrans_q(C.astype(int))
            tests.append({"model": m, "test": "Cochran's Q (accuracy, 4 styles)", "statistic": q.statistic,
                          "df": C.shape[1] - 1, "p": q.pvalue, "p_holm": np.nan})
        pair = []
        for s1, s2 in combinations(styles, 2):
            pv, b, c = mcnemar_exact(wc[s1], wc[s2])
            pair.append({"model": m, "test": f"McNemar accuracy {s1} vs {s2}", "statistic": np.nan,
                         "df": 1, "p": pv, "only_first_correct": b, "only_second_correct": c})
        for x, pa in zip(pair, multipletests([x["p"] for x in pair], method="holm")[1]):
            x["p_holm"] = pa
        tests += pair
        if {"simple", "cot"} <= set(styles):
            pv, b, c = mcnemar_exact(wh["simple"], wh["cot"])
            tests.append({"model": m, "test": "McNemar hallucination simple vs cot", "statistic": np.nan,
                          "df": 1, "p": pv, "p_holm": np.nan, "only_first_correct": b, "only_second_correct": c})

    ci, mr, tt = pd.DataFrame(ci_rows), pd.DataFrame(model_rows), pd.DataFrame(tests)
    ci.to_csv(out / f"bootstrap_ci{args.tag}.csv", index=False)
    mr.to_csv(out / f"model_level_ci{args.tag}.csv", index=False)
    tt.to_csv(out / f"significance_tests{args.tag}.csv", index=False)

    for metric in ["accuracy", "hallucination"]:
        t = ci[ci.metric == metric].copy()
        t["cell"] = [pct(a, b, c) for a, b, c in zip(t["mean"], t.lo, t.hi)]
        piv = t.pivot(index="model", columns="prompt_style", values="cell")[styles]
        piv.to_markdown(out / f"table_{metric}{args.tag}.md")
        piv.to_latex(out / f"table_{metric}{args.tag}.tex", escape=True, label=f"tab:{metric}",
                     caption=f"{metric.capitalize()} (\\%) by model and prompt style, with 95\\% bootstrap CIs.")
    mt = mr.assign(cell=[pct(a, b, c) for a, b, c in zip(mr["mean"], mr.lo, mr.hi)]).pivot(index="model", columns="metric", values="cell")
    mt.to_markdown(out / f"table_consistency_sensitivity{args.tag}.md")
    mt.to_latex(out / f"table_consistency_sensitivity{args.tag}.tex", escape=True, label="tab:cons",
                caption="Consistency rate and accuracy spread across prompt styles (\\%), 95\\% bootstrap CIs.")
    print(ci.assign(cell=[pct(a, b, c) for a, b, c in zip(ci["mean"], ci.lo, ci.hi)])
            .pivot_table(index=["model", "metric"], columns="prompt_style", values="cell", aggfunc="first")[styles].to_string())
    print(mt.to_string()); print(tt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
