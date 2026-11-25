#!/usr/bin/env bash
# Re-run the whole analysis from results/raw/*.jsonl (needs: pip install -r requirements.txt).
# Only the four main models are graded; the pilot (qwen2.5_7b_pilot_qwen.jsonl) and smoke files are ignored.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/results/tables" "$ROOT/results/figures" "$ROOT/results/audit" "$ROOT/paper/figures"
cd "$ROOT/src"
python grade.py                 # pass 1 -> results/results.csv + audit sample
python grade.py --tag _pass2    # pass 2 -> results/results_pass2.csv
python audit_assistant.py       # coder-1 audit codes + agreement tables
python grader_change_log.py     # v1 (strict) vs v2 grader
python metrics.py && python metrics.py --tag _pass2
python stats.py                 # bootstrap CIs, Cochran's Q, McNemar/Holm
python model_overall_ci.py
python extra_analysis.py        # verbosity, pooled tests, surface consistency, H2, pass-2 stability, robustness
python power_and_invented.py    # invented-detail summary + McNemar power / ceiling MDE
python plots.py && python plots_extra.py
python kappa.py || true         # coder1 vs coder2 if a second coding sheet is filled; otherwise a no-op
mkdir -p "$ROOT/paper/figures"
for f in fig1_accuracy_by_style fig2_hallucination_by_style \
         fig6_verbosity_by_style fig7_label_vs_surface_consistency fig8_pass2_stability; do
  cp -f "$ROOT/results/figures/${f}.png" "$ROOT/results/figures/${f}.pdf" "$ROOT/paper/figures/"
done
cd "$ROOT/paper" && python build_tables.py >/dev/null && python build_paper.py
python "$ROOT/src/check_number_provenance.py"
