"""Build the paper's tables (Markdown pipe tables) straight from results/tables/*.csv.
Output: paper/tables_generated.md-fragments as a dict saved to paper/_tables.json, used by build_paper.py."""
import json, sys
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from common import display_model

T = R / "results" / "tables"
STY = ["simple", "detailed", "polite", "cot"]; SN = {"simple": "Simple", "detailed": "Detailed", "polite": "Polite", "cot": "CoT"}
pc = lambda x: f"{100*x:.1f}"
def tok(x):
    v = float(x)
    return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def md(df, seps=None):
    cols = list(df.columns)
    seps = seps or ([":--"] + ["--:"] * (len(cols) - 1))
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(seps) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


tabs = {}
b = pd.read_csv(T / "bootstrap_ci.csv"); ms = pd.read_csv(T / "metrics_by_style.csv")
rows = []
for m in sorted(b.model.unique()):
    for s in STY:
        a = b[(b.model == m) & (b.prompt_style == s) & (b.metric == "accuracy")].iloc[0]
        h = ms[(ms.model == m) & (ms.prompt_style == s)].iloc[0]
        rows.append({
            "Model": display_model(m),
            "Style": SN[s],
            "Accuracy [95% CI]": f"{pc(a['mean'])} [{pc(a.lo)}, {pc(a.hi)}]",
            "Halluc.": pc(h.hallucination),
            "Format": pc(h.format_failure_rate),
        })
tabs["T_STYLE"] = md(pd.DataFrame(rows), [":--", ":--", "--:", "--:", "--:"])

o = pd.read_csv(T / "model_overall_ci.csv"); ml = pd.read_csv(T / "model_level_ci.csv"); sc = pd.read_csv(T / "strict_consistency.csv")
mm = pd.read_csv(T / "metrics_by_model.csv"); sig = pd.read_csv(T / "significance_tests.csv")
rows = []
for m in sorted(o.model.unique()):
    a = o[(o.model == m) & (o.metric == "accuracy")].iloc[0]; h = o[(o.model == m) & (o.metric == "hallucination")].iloc[0]
    cr = ml[(ml.model == m) & (ml.metric == "consistency_rate")].iloc[0]; ps = ml[(ml.model == m) & (ml.metric == "sensitivity_spread_PS")].iloc[0]
    s = sc[sc.model == m].iloc[0]; q = sig[(sig.model == m) & (sig.test.str.startswith("Cochran"))].iloc[0]
    rows.append({"Model": display_model(m), "Accuracy": f"{pc(a['mean'])} [{pc(a.lo)}, {pc(a.hi)}]", "Halluc.": f"{pc(h['mean'])} [{pc(h.lo)}, {pc(h.hi)}]",
                 "CR (label)": f"{pc(cr['mean'])} [{pc(cr.lo)}, {pc(cr.hi)}]",
                 "Surface cons.": f"{pc(s.strict_surface_consistency)} [{pc(s.strict_ci_lo)}, {pc(s.strict_ci_hi)}]",
                 "PS (pp)": f"{pc(ps['mean'])} [{pc(ps.lo)}, {pc(ps.hi)}]",
                 "QS": pc(mm[mm.model == m].question_sensitivity_QS.iloc[0]),
                 "Cochran Q (p)": f"{q.statistic:.2f} ({q.p:.3f})"})
tabs["T_MODEL"] = md(pd.DataFrame(rows))

f = pd.read_csv(T / "format_compliance_by_style.csv"); fq = pd.read_csv(T / "format_cochran_q.csv")
rows = []
for m in sorted(f.model.unique()):
    r = {"Model": display_model(m)}
    for s in STY:
        x = f[(f.model == m) & (f.prompt_style == s)].iloc[0]
        r[SN[s]] = f"{pc(x.verbose_rate)} / {tok(x.median_output_tokens)}"
    q = fq[(fq.model == m) & (fq.variable == "verbose")].iloc[0]
    r["Cochran Q (p)"] = f"{q.Q:.1f} ({q.p:.2g})"
    rows.append(r)
tabs["T_FORMAT"] = md(pd.DataFrame(rows))

ps = pd.read_csv(T / "pass2_stability.csv")
tabs["T_PASS2"] = md(pd.DataFrame([{"Model": display_model(r.model), "Pairs": r.n_pairs, "Identical raw text": pc(r.identical_raw_response),
                                    "Identical extracted answer": pc(r.identical_extracted_answer),
                                    "Identical correctness label": pc(r.identical_correct_label)} for r in ps.itertuples()]))

g = pd.read_csv(T / "grader_v1_v2_summary.csv"); rb = pd.read_csv(T / "robustness.csv")
rows = []
for r in rb.itertuples():
    rows.append({"Model": display_model(r.model), "Strict v1 grader": pc(r.grader_v1), "Final v2 grader": pc(r.main),
                 "Excl. K09, S22": pc(r.excl_ambiguous_K09_S22), "No-'Final answer' rows wrong": pc(r.fallback_long_as_format),
                 "'Nur-Sultan' wrong": pc(r.NurSultan_as_wrong)})
tabs["T_ROBUST"] = md(pd.DataFrame(rows))

pp = pd.read_csv(T / "pooled_style_tests.csv"); cp = pd.read_csv(T / "pooled_cluster_permutation.csv")
rows = []
for met, lab in [("correct", "Accuracy"), ("hallucination", "Hallucination")]:
    r = {"Metric (all models)": lab}
    for s in STY:
        x = pp[(pp.metric == met) & (pp["style"] == s)].iloc[0]
        r[SN[s]] = f"{pc(x.pooled_rate)} [{pc(x.ci_lo)}, {pc(x.ci_hi)}]"
    q = pp[(pp.metric == met) & (pp["style"].str.startswith("Cochran"))].iloc[0]
    c = cp[cp.metric == met].iloc[0]
    r["Q; naive p; cluster-perm. p"] = f"{q.pooled_rate:.2f}; {q.ci_hi:.3f}; {c.cluster_perm_p:.3f}"
    rows.append(r)
tabs["T_POOLED"] = md(pd.DataFrame(rows))

def abbreviate_answer(text: str) -> str:
    """Short display form for Table 7. The full extracted answer remains in results/tables/hard_items.csv and the raw JSONL."""
    import re
    s = str(text).replace("|", "/")
    # Llama-3.1-8B K21 CoT collapsed into the word "Red" until the token limit.
    if re.match(r"^(Red,\s*){3,}", s):
        return "Red, Red, ... (repeated until the token limit)"
    if len(s) <= 72:
        return s
    cut = s[:72]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + " ..."

e = pd.read_csv(T / "hard_items.csv")
e = e.assign(model_answer=e.model_answer.map(abbreviate_answer))
e = e.assign(model=e.model.map(display_model), prompt_style=e.prompt_style.map(lambda s: SN.get(s, s)))
e = e[["model", "question_id", "prompt_style", "gold_answer", "model_answer", "error_type"]].rename(columns={
    "model": "Model", "question_id": "Item", "prompt_style": "Style", "gold_answer": "Gold",
    "model_answer": "Extracted answer (abbreviated)", "error_type": "Auto code"})
tabs["T_ERRORS"] = md(e, [":" + "-" * 11, ":" + "-" * 4, ":" + "-" * 7, ":" + "-" * 9, ":" + "-" * 40, ":" + "-" * 9])
json.dump(tabs, open(R / "paper" / "_tables.json", "w"), indent=1)
for k, v in tabs.items(): print(k); print(v); print()
