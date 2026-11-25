"""Secondary analyses added after the main metrics (all from results/results*.csv and raw files):

  1. format_compliance_by_style.csv : per model x style - share with a 'Final answer:' line,
     share of verbose answers (> 5 words in the extracted answer), median words, median output
     tokens, median latency. Cochran's Q across styles for 'verbose' per model.
  2. pooled_style_tests.csv          : accuracy and hallucination per style pooled over models
     (480 model-question units) with cluster bootstrap CIs (resampling questions), Cochran's Q.
  3. strict_consistency.csv          : surface-form consistency (normalized first sentence of the
     extracted answer identical in all 4 styles) vs label consistency, with bootstrap CIs.
  4. h2_consistency_vs_halluc.csv    : hallucination rate on consistent vs inconsistent questions.
  5. pass2_stability.csv             : pass 1 vs pass 2 on the 30-question subset: identical raw
     response, identical extracted answer, identical correctness label.
  6. robustness.csv                  : accuracy under alternative grading choices.
  7. hard_items.csv                  : every (model, question) with at least one wrong style.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q
from scipy.stats import fisher_exact

from common import load_config, normalize, p, read_jsonl
from grade import first_sentence

STYLES = ["simple", "detailed", "polite", "cot"]


def boot_ci(units: np.ndarray, B: int, rng, stat=np.mean):
    n = len(units)
    vals = [stat(units[rng.integers(0, n, n)]) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def main():
    cfg = load_config(); B = cfg["stats"]["bootstrap_B"]; T = p(cfg["paths"]["tables_dir"])
    df = pd.read_csv(p(cfg["paths"]["results_csv"]), dtype={"question_id": str}, keep_default_na=False)
    df["first"] = df.model_answer.astype(str).map(lambda s: normalize(first_sentence(s)))
    df["nwords"] = df.model_answer.astype(str).map(lambda s: len(s.split()))
    df["verbose"] = (df.nwords > 5).astype(int)
    df["has_final"] = (df.extraction == "final_line").astype(int)
    for c in ["correct", "hallucination", "consistent", "output_tokens"]:
        df[c] = pd.to_numeric(df[c])
    df["latency_s"] = pd.to_numeric(df.latency_s)

    # 1. format compliance
    f = df.groupby(["model", "prompt_style"]).agg(
        final_line_rate=("has_final", "mean"), verbose_rate=("verbose", "mean"),
        median_answer_words=("nwords", "median"), median_output_tokens=("output_tokens", "median"),
        median_latency_s=("latency_s", "median"), max_latency_s=("latency_s", "max")).reset_index()
    f.to_csv(T / "format_compliance_by_style.csv", index=False)
    qrows = []
    for m, g in df.groupby("model"):
        for var in ["verbose", "has_final"]:
            w = g.pivot(index="question_id", columns="prompt_style", values=var)[STYLES]
            if w.values.min() == w.values.max():
                qrows.append({"model": m, "variable": var, "Q": np.nan, "df": 3, "p": np.nan}); continue
            r = cochrans_q(w.values)
            qrows.append({"model": m, "variable": var, "Q": round(r.statistic, 3), "df": 3, "p": r.pvalue})
    pd.DataFrame(qrows).to_csv(T / "format_cochran_q.csv", index=False)
    print(f.to_string(index=False)); print(pd.DataFrame(qrows).to_string(index=False))

    # 2. pooled over models; cluster bootstrap over questions
    rng = np.random.default_rng(cfg["seed"])
    qids = np.array(sorted(df.question_id.unique()))
    rows = []
    for var in ["correct", "hallucination"]:
        w = df.pivot_table(index="question_id", columns="prompt_style", values=var, aggfunc="mean")[STYLES]
        for s in STYLES:
            lo, hi = boot_ci(w[s].values, B, rng)
            rows.append({"metric": var, "style": s, "pooled_rate": w[s].mean(), "ci_lo": lo, "ci_hi": hi})
        w2 = df.assign(u=df.model + "|" + df.question_id).pivot(index="u", columns="prompt_style", values=var)[STYLES]
        r = cochrans_q(w2.values)
        rows.append({"metric": var, "style": "Cochran Q (480 units)", "pooled_rate": r.statistic, "ci_lo": np.nan, "ci_hi": r.pvalue})
    pooled = pd.DataFrame(rows); pooled.to_csv(T / "pooled_style_tests.csv", index=False); print(pooled.to_string(index=False))

    # 3. strict (surface) consistency
    srows = []
    for m, g in df.groupby("model"):
        w = g.pivot(index="question_id", columns="prompt_style", values="first")[STYLES]
        strict = (w.nunique(axis=1) == 1).astype(int).values
        lab = g.groupby("question_id").consistent.first().reindex(w.index).values
        lo, hi = boot_ci(strict, B, rng); lo2, hi2 = boot_ci(lab, B, rng)
        srows.append({"model": m, "label_consistency": lab.mean(), "label_ci_lo": lo2, "label_ci_hi": hi2,
                      "strict_surface_consistency": strict.mean(), "strict_ci_lo": lo, "strict_ci_hi": hi})
    s = pd.DataFrame(srows); s.to_csv(T / "strict_consistency.csv", index=False); print(s.to_string(index=False))

    # 4. H2
    hrows = []
    for m, g in df.groupby("model"):
        cons = g.groupby("question_id").consistent.first()
        h = g.groupby("question_id").hallucination.sum()
        a, b = h[cons == 1].sum(), (cons == 1).sum() * 4 - h[cons == 1].sum()
        c, d = h[cons == 0].sum(), (cons == 0).sum() * 4 - h[cons == 0].sum()
        pv = fisher_exact([[a, b], [c, d]])[1]
        hrows.append({"model": m, "n_consistent_q": int((cons == 1).sum()), "n_inconsistent_q": int((cons == 0).sum()),
                      "halluc_rate_consistent": a / max(a + b, 1), "halluc_rate_inconsistent": c / max(c + d, 1),
                      "fisher_p_rowlevel": pv})
    h2 = pd.DataFrame(hrows); h2.to_csv(T / "h2_consistency_vs_halluc.csv", index=False); print(h2.to_string(index=False))

    # 5. pass-2 stability
    raw = p(cfg["paths"]["raw_dir"])
    p2 = pd.read_csv(p("results/results_pass2.csv"), dtype={"question_id": str}, keep_default_na=False)
    prow = []
    for mdl in cfg["models"]:
        tag = mdl.replace(":", "_").replace("/", "_")
        r1 = {(r["question_id"], r["prompt_style"]): r["response"] for r in read_jsonl(raw / f"{tag}.jsonl")}
        r2 = {(r["question_id"], r["prompt_style"]): r["response"] for r in read_jsonl(raw / f"{tag}_pass2.jsonl")}
        keys = sorted(r2)
        a1 = df[df.model == mdl].set_index(["question_id", "prompt_style"])
        a2 = p2[p2.model == mdl].set_index(["question_id", "prompt_style"])
        raw_eq = [r1[k] == r2[k] for k in keys]
        ans_eq = [normalize(a1.loc[k, "model_answer"]) == normalize(a2.loc[k, "model_answer"]) for k in keys]
        lab_eq = [int(a1.loc[k, "correct"]) == int(a2.loc[k, "correct"]) for k in keys]
        et_eq = [a1.loc[k, "error_type"] == a2.loc[k, "error_type"] for k in keys]
        diff = [k for k, e in zip(keys, raw_eq) if not e]
        prow.append({"model": mdl, "n_pairs": len(keys), "identical_raw_response": np.mean(raw_eq),
                     "identical_extracted_answer": np.mean(ans_eq), "identical_correct_label": np.mean(lab_eq),
                     "identical_error_type": np.mean(et_eq), "n_raw_differences": len(diff),
                     "differing_items": "; ".join(f"{q}/{s}" for q, s in diff)})
        for k in diff:
            print("PASS2 DIFF", mdl, k, "| p1:", repr(r1[k][-120:]), "| p2:", repr(r2[k][-120:]))
    ps = pd.DataFrame(prow); ps.to_csv(T / "pass2_stability.csv", index=False); print(ps.drop(columns="differing_items").to_string(index=False))

    # 6. robustness
    base = df.groupby("model").correct.mean().rename("main")
    rob = pd.DataFrame(base)
    rob["excl_ambiguous_K09_S22"] = df[~df.question_id.isin(["K09", "S22"])].groupby("model").correct.mean()
    rob["fallback_long_as_format"] = df.assign(c=np.where(df.extraction == "fallback_long", 0, df.correct)).groupby("model").c.mean()
    rob["NurSultan_as_wrong"] = df.assign(c=np.where(df.first.str.contains("nur sultan"), 0, df.correct)).groupby("model").c.mean()
    v1 = pd.read_csv(T / "grader_v1_v2_summary.csv").set_index("model").acc_v1
    rob["grader_v1"] = v1
    rob["n_NurSultan_rows"] = df[df.first.str.contains("nur sultan")].groupby("model").size()
    rob["n_fallback_long_rows"] = df[df.extraction == "fallback_long"].groupby("model").size()
    rob = rob.fillna(0).reset_index(); rob.to_csv(T / "robustness.csv", index=False); print(rob.to_string(index=False))

    # 7. hard items
    hard = df[df.correct == 0][["model", "question_id", "prompt_style", "gold_answer", "model_answer", "error_type", "extraction"]]
    hard.to_csv(T / "hard_items.csv", index=False)
    print(df[df.correct == 0].groupby("question_id").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()


def cluster_permutation(n_perm: int = 10000):
    """Pooled style test that respects clustering by question: under H0 (style has no effect)
    the 4 style labels are exchangeable within a question, so we apply ONE random permutation of
    the style labels per question, shared by all 4 models, and recompute the pooled Cochran's Q."""
    cfg = load_config(); T = p(cfg["paths"]["tables_dir"])
    df = pd.read_csv(p(cfg["paths"]["results_csv"]), dtype={"question_id": str}, keep_default_na=False)
    rng = np.random.default_rng(cfg["seed"])
    models = sorted(df.model.unique()); qids = sorted(df.question_id.unique())
    out = []
    for var in ["correct", "hallucination"]:
        X = np.stack([df[df.model == m].pivot(index="question_id", columns="prompt_style", values=var)
                      .reindex(qids)[STYLES].values.astype(int) for m in models])  # (M, Q, 4)

        def qstat(arr):
            Y = arr.reshape(-1, 4); k = 4
            C = Y.sum(0); R = Y.sum(1); N = Y.sum()
            den = k * N - (R ** 2).sum()
            return 0.0 if den == 0 else (k - 1) * (k * (C ** 2).sum() - N ** 2) / den
        obs = qstat(X); cnt = 0
        for _ in range(n_perm):
            perm = np.array([rng.permutation(4) for _ in qids])            # (Q, 4)
            Xp = np.take_along_axis(X, perm[None, :, :].repeat(len(models), 0), axis=2)
            cnt += qstat(Xp) >= obs - 1e-12
        out.append({"metric": var, "pooled_cochran_Q": obs, "cluster_perm_p": (cnt + 1) / (n_perm + 1), "n_perm": n_perm})
    o = pd.DataFrame(out); o.to_csv(T / "pooled_cluster_permutation.csv", index=False); print(o.to_string(index=False))


if __name__ == "__main__":
    cluster_permutation()
