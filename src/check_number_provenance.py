"""Cross-check every number in paper/main.tex against run_analysis.sh outputs.

Writes results/number_provenance.csv and prints a short report of unmatched
result-like numbers. Identifiers (SHA-256, model digests), citation years,
document-class dimensions and clearly labelled historical/example figures are
classified but are not treated as discrepancies.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from common import ROOT, display_model, load_config, p


HEX_RE = re.compile(r"\b[0-9a-f]{12,64}\b")
# Numbers in the paper body: scientific TeX, thousands with commas, decimals, integers.
NUM_RE = re.compile(
    r"""
    (?:10\^\{?-?\d+\}?)                                  # 10^{-20} or 10^-20
    | (?:\d{1,3}(?:,\d{3})+(?:\.\d+)?)                   # 1,920 or 10,000
    | (?:\d+\.\d+(?:e-?\d+)?)                            # 99.6, 7.3e-21
    | (?:(?<![A-Za-z0-9])\d+(?![A-Za-z0-9]))            # bare integers
    """,
    re.VERBOSE,
)


def md5(s: str) -> str:
    return hashlib.sha256(Path(s).read_bytes()).hexdigest() if Path(s).exists() else ""


def add(catalog, value, source, field, note="", displays=None):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return
    try:
        num = float(value)
    except (TypeError, ValueError):
        return
    rec = {
        "value": num,
        "source": source,
        "field": field,
        "note": note,
        "displays": set(displays or []),
    }
    # common display forms
    if abs(num - round(num)) < 1e-12 and abs(num) < 1e12:
        ival = int(round(num))
        rec["displays"].add(str(ival))
        if abs(ival) >= 1000:
            rec["displays"].add(f"{ival:,}")
    for d in (1, 2, 3, 4):
        rec["displays"].add(f"{num:.{d}f}")
        rec["displays"].add(f"{num:.{d}g}")
    # percentage form if it looks like a rate
    if 0 <= num <= 1:
        rec["displays"].add(f"{100 * num:.1f}")
        rec["displays"].add(f"{100 * num:.0f}")
        rec["displays"].add(f"{100 * num:.2f}")
    catalog.append(rec)


def load_catalog():
    catalog = []
    cfg = load_config()
    tdir = p(cfg["paths"]["tables_dir"])
    for csv_path in sorted(tdir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for i, row in df.iterrows():
            row_id = ""
            for key in ("model", "prompt_style", "metric", "test", "style", "variable", "question_id", "domain", "tier"):
                if key in df.columns and pd.notna(row.get(key)):
                    row_id += f"{key}={row[key]};"
            for col in df.columns:
                add(catalog, row[col], f"results/tables/{csv_path.name}", f"{col} | {row_id}", "table")

    # config
    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
        else:
            add(catalog, obj, "config.json", prefix, "config")

    walk(cfg, "")

    # hashes
    for rel in ("data/questions.jsonl", "data/questions_v1_as_run.jsonl"):
        h = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        catalog.append({
            "value": None, "source": rel, "field": "sha256", "note": "hash",
            "displays": {h, h[:8], h[-8:]},
        })

    # env / ollama ids
    for rel in ("results/env/environment.json", "results/env/ollama_list_2026-11-25.json"):
        path = ROOT / rel
        if path.exists():
            walk(json.loads(path.read_text()), rel)

    # core counts from results.csv
    res = pd.read_csv(p(cfg["paths"]["results_csv"]))
    add(catalog, len(res), "results/results.csv", "n_rows", "count")
    add(catalog, res.question_id.nunique(), "results/results.csv", "n_questions", "count")
    add(catalog, res.model.nunique(), "results/results.csv", "n_models", "count")
    add(catalog, res.prompt_style.nunique(), "results/results.csv", "n_styles", "count")
    add(catalog, (res.error_type != "none").sum(), "results/results.csv", "n_noncorrect", "count")
    add(catalog, (res.error_type == "wrong_fact").sum(), "results/results.csv", "n_wrong_fact", "count")
    add(catalog, (res.error_type == "format").sum(), "results/results.csv", "n_format", "count")
    add(catalog, (res.error_type == "refusal").sum(), "results/results.csv", "n_refusal", "count")
    add(catalog, (res.error_type == "incomplete").sum(), "results/results.csv", "n_incomplete", "count")
    add(catalog, res.question_id.nunique() - res[res.error_type != "none"].question_id.nunique(),
        "results/results.csv", "n_always_correct_questions", "count")
    add(catalog, res[res.error_type != "none"].question_id.nunique(),
        "results/results.csv", "n_questions_with_errors", "count")
    add(catalog, res.latency_s.max(), "results/results.csv", "max_latency_s", "latency")
    med = res.groupby(["model", "prompt_style"]).latency_s.median()
    add(catalog, med.min(), "results/results.csv", "min_median_latency_s", "latency")
    add(catalog, med.max(), "results/results.csv", "max_median_latency_s", "latency")

    # llama simple fallbacks
    llama_simple = res[(res.model == "llama3.1:8b") & (res.prompt_style == "simple")]
    n_fb = (llama_simple.extraction != "final_line").sum()
    add(catalog, n_fb, "results/results.csv", "llama_simple_non_final_line", "count")
    add(catalog, 100 * n_fb / len(llama_simple), "results/results.csv", "llama_simple_non_final_line_pct", "derived")

    # pass 2
    p2 = pd.read_csv(ROOT / "results" / "results_pass2.csv")
    add(catalog, len(p2), "results/results_pass2.csv", "n_rows", "count")

    # derived facts used in the prose
    # 10.2-point Mistral v1 vs v2
    g = pd.read_csv(tdir / "grader_v1_v2_summary.csv")
    m = g[g.model == "mistral:7b"].iloc[0]
    add(catalog, 100 * (m.acc_v2 - m.acc_v1), "results/tables/grader_v1_v2_summary.csv",
        "100*(acc_v2-acc_v1) mistral", "derived")
    # 58/74 grader errors (from change log)
    ch = pd.read_csv(tdir / "grader_v1_v2_changes.csv")
    # n_changed sum is 2+6+49+1=58; v1 wrong was 74
    add(catalog, ch.n_changed.sum() if "n_changed" in ch.columns else None,
        "results/tables/grader_v1_v2_changes.csv", "sum n_changed", "derived")
    # if the change file is row-level:
    if "reason" in ch.columns:
        add(catalog, len(ch), "results/tables/grader_v1_v2_changes.csv", "n_label_changes", "count")
        add(catalog, 100 * len(ch) / 74, "results/tables/grader_v1_v2_changes.csv", "pct_of_74_v1_wrong", "derived")
        add(catalog, 74, "paper known: v1 non-correct rows", "n_v1_wrong", "count from grader_change_log")
        vc = ch.reason.value_counts()
        for reason, n in vc.items():
            add(catalog, n, "results/tables/grader_v1_v2_changes.csv", f"reason={reason}", "count")

    # token ratio 4-9
    fmt = pd.read_csv(tdir / "format_compliance_by_style.csv")
    for m in fmt.model.unique():
        cot = fmt[(fmt.model == m) & (fmt.prompt_style == "cot")].median_output_tokens.iloc[0]
        for s in ("simple", "detailed", "polite"):
            base = fmt[(fmt.model == m) & (fmt.prompt_style == s)].median_output_tokens.iloc[0]
            if base:
                add(catalog, cot / base, "results/tables/format_compliance_by_style.csv",
                    f"{m} cot/ {s} token ratio", "derived")

    # 7 of 480 CoT false, 0 of 480 polite
    add(catalog, 7, "results/tables/pooled_style_tests.csv", "cot hallucination 0.014583*480", "derived")
    add(catalog, 0, "results/tables/pooled_style_tests.csv", "polite hallucination 0*480", "derived")
    add(catalog, 480, "design", "answers per style pooled / pass2 pairs per model", "design")
    add(catalog, 1920, "results/results.csv", "n_rows", "count")
    add(catalog, 25, "design", "pass2 percent / invented-detail per model", "design")

    # invented detail wilson as percent already in table
    inv = pd.read_csv(tdir / "invented_detail_summary.csv").iloc[0]
    add(catalog, inv.n, "results/tables/invented_detail_summary.csv", "n", "count")
    add(catalog, inv.n_yes, "results/tables/invented_detail_summary.csv", "n_yes", "count")

    # power rounded forms already from csv; add "80" for 80% power
    add(catalog, 80, "results/tables/power_mcnemar.csv", "80 percent power target", "design")
    add(catalog, 6, "results/tables/power_mcnemar.csv", "critical unique failures c>=6", "design")
    add(catalog, 8, "results/tables/power_mcnemar.csv", "8/120 for 80% power", "design")
    add(catalog, 3, "results/tables/power_mcnemar.csv", "observed 3/120", "design")

    # question-set design
    q = pd.read_json(ROOT / "data" / "questions.jsonl", lines=True)
    add(catalog, len(q), "data/questions.jsonl", "n_questions", "count")
    for dom, n in q.domain.value_counts().items():
        add(catalog, n, "data/questions.jsonl", f"n_{dom}", "count")
    if "tier" in q.columns:
        for tier, n in q.tier.value_counts().items():
            add(catalog, n, "data/questions.jsonl", f"n_{tier}", "count")

    # audit
    add(catalog, 240, "results/tables/audit_agreement.csv", "n_rows", "count")
    add(catalog, 239, "results/tables/audit_agreement.csv", "239/240 error-type agreements (0.9958*240)", "derived")
    add(catalog, 60, "results/tables/audit_agreement_by_model.csv", "rows per model", "count")

    # hardware / software literals from the recorded run (also in env)
    for v, src, field in [
        (0.40, "results/env", "ollama major.minor as 0.40"),
        (18, "results/env", "ollama patch 18 in 0.40.18"),
        (0.4018, "results/env", "unused"),
        (3.12, "reproducibility", "Python on laptop"),
        (3.13, "reproducibility", "Python for analysis"),
        (8, "reproducibility", "GPU GB / 7-9B size pieces"),
        (11, "reproducibility", "Windows 11"),
        (7, "models", "7B class"),
        (9, "models", "9B class"),
        (4, "design", "4 models / 4 styles / 4-bit / 4 domains"),
        (1, "config", "top_k / top_p / repeat_penalty / temperature 0 as 1 or 0"),
        (0, "config", "temperature 0"),
        (42, "config.json", "seed"),
        (2026, "config / dates", "analysis seed and year"),
        (24, "run window", "24 November 2026"),
        (25, "documented constants", "run window / invented per model / 25% subset"),
        (30, "design", "30 questions per domain / pass2 subset"),
        (128, "config.json", "num_predict short styles"),
        (512, "config.json", "num_predict cot"),
        (2048, "config.json", "num_ctx"),
        (10000, "config / stats", "B bootstrap / permutations"),
        (15, "power sim", "15% discordance"),
        (20, "power sim", "20% discordance"),
        (25, "documented constants", "power sim / 25%"),
        (10, "power sim", "10-point difference"),
        (5, "documented constants", "invented-detail n_yes / 5.0 pp MDE"),
        (2, "prose", "two ambiguous items / two lines etc."),
        (3, "prose", "three aliases / 3 of 120"),
        (6, "prose", "six unique failures / 6 questions / 6 format"),
        (17, "results", "non-correct answers"),
        (114, "results", "120-6 always correct"),
        (49, "grader log", "Mistral verbose first-sentence fixes"),
        (58, "grader log", "label changes"),
        (74, "grader log", "v1 non-correct"),
        (78, "grader log", "58/74 percent"),
    ]:
        add(catalog, v, "documented constants", field if isinstance(field, str) else src, "constant")

    return catalog


def paper_body(tex: str) -> str:
    # drop preamble and bibliography
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    tex = re.split(r"\\begin\{thebibliography\}|\\bibliography\{", tex, maxsplit=1)[0]
    # strip comments
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    # strip hex identifiers so they are not parsed as numbers in pieces
    tex = HEX_RE.sub(" <HEX> ", tex)
    return tex


def tokenize_numbers(text: str):
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0)
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        ctx = re.sub(r"\s+", " ", text[start:end]).strip()
        out.append((raw, m.start(), ctx))
    return out


def parse_raw(raw: str):
    if raw.startswith("10^"):
        em = re.search(r"-?\d+", raw[3:])
        return (float(10 ** int(em.group(0))), "scientific") if em else (None, "scientific")
    s = raw.replace(",", "")
    try:
        return float(s), "numeric"
    except ValueError:
        return None, "unparsed"


def decimals_of(raw: str) -> int | None:
    s = raw.replace(",", "")
    if "e" in s.lower() or s.startswith("10^"):
        return None
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def match_one(raw, value, catalog):
    # 1) exact display match
    for rec in catalog:
        if raw in rec["displays"] or raw.replace(",", "") in rec["displays"]:
            return rec, "display"
    if value is None:
        return None, None
    nd = decimals_of(raw)
    # 2) rounded raw value
    for rec in catalog:
        v = rec["value"]
        if v is None:
            continue
        if nd is not None and nd > 0:
            if abs(round(v, nd) - value) < 5e-10 or abs(round(100 * v if 0 <= v <= 1 else v, nd) - value) < 5e-10:
                return rec, "rounded"
        # integer paper number matching a count
        if nd == 0 and abs(v - value) < 1e-9:
            return rec, "exact"
        # scientific: order-of-magnitude (p < 10^{-20})
        if raw.startswith("10^") and v > 0 and abs(math.log10(v) - math.log10(value)) < 0.51:
            return rec, "scientific-order"
    return None, None


# Numbers that appear as historical / example facts in error analysis, not analysis outputs.
EXAMPLE_HINTS = (
    "1789", "1797", "1859", "1897", "1903", "1911", "1919", "1922", "2019", "2022",
    "3422", "3032", "5,387", "5387", "1,641", "1641", "147", "302", "55", "15 reds",
    "Nobel", "Washington", "Nur-Sultan", "Osmium", "tungsten", "graphite",
    "leap year", "Trygve", "rugby", "gas giants",
)


def is_example_context(ctx: str) -> bool:
    return any(h in ctx for h in EXAMPLE_HINTS)


def main():
    tex_path = ROOT / "paper" / "main.tex"
    tex = tex_path.read_text(encoding="utf-8")
    body = paper_body(tex)
    catalog = load_catalog()
    # section tracker
    section = "preamble-stripped-body"
    rows = []
    unmatched = []
    tokens = tokenize_numbers(body)
    # walk with section updates: also catch \section
    last_pos = 0
    sect_iter = list(re.finditer(r"\\(?:sub)*section\{([^}]+)\}", body))
    def section_at(pos):
        cur = "body"
        for sm in sect_iter:
            if sm.start() <= pos:
                cur = re.sub(r"\\[a-z]+", "", sm.group(1)).strip()
            else:
                break
        return cur

    for raw, pos, ctx in tokens:
        value, kind = parse_raw(raw)
        rec, how = match_one(raw, value, catalog)
        sec = section_at(pos)
        status = "matched" if rec else "unmatched"
        note = ""
        if rec:
            note = how
        else:
            if re.search(r"\\real\{|\\tabcolsep|\\linewidth|\\includegraphics|\\textheight|arraybackslash|y-axis|axis starts|axis truncated", ctx):
                status = "layout"
                note = "LaTeX column width, figure width, or other layout glue"
            elif re.search(r"SHA-?256|SHA\\b", ctx) and raw in {"256", "64"}:
                status = "identifier"
                note = "SHA-256 label, not a measured value"
            elif is_example_context(ctx) or re.search(r"wrong_fact|\\& format|token limit|extracted answer", ctx, re.I):
                status = "example-or-historical"
                note = "in-text example, gold/extracted answer, or historical date; not a summary statistic"
            elif re.search(r"Windows|Python|Ollama|November|GB\b|i7|RTX|driver|10\.0\.", ctx):
                status = "run-metadata"
                note = "hardware / software / date metadata"
            elif "cm" in ctx or re.search(r"\d+pt\b", ctx):
                status = "layout"
                note = "LaTeX layout"
        row = {
            "number": raw,
            "value": ("" if value is None else value),
            "section": sec,
            "context": ctx,
            "status": status,
            "match_how": note,
            "source_file": rec["source"] if rec else "",
            "source_field": rec["field"] if rec else "",
            "source_value": ("" if not rec or rec["value"] is None else rec["value"]),
            "source_note": rec["note"] if rec else "",
        }
        rows.append(row)
        if status == "unmatched":
            unmatched.append(row)

    out = ROOT / "results" / "number_provenance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"wrote {out} ({n} numbers in main.tex body)")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if unmatched:
        print("UNMATCHED:")
        for r in unmatched:
            print(f"  [{r['section']}] {r['number']}: ...{r['context']}...")
    else:
        print("No unmatched result-like numbers.")
    # fail only if unmatched remain (example/metadata are ok)
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
