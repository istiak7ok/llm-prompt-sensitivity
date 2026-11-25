# Invented-detail sample

A stratified random sample of 100 automatically correct pass-1 answers (25 per model, `random_state=2026`) was drawn from `results/results.csv` into `invented_detail_sample.csv` and coded in `invented_detail_sample_coded.csv`.

**Definition.** A row is coded **Y** only if the response asserts a *false* factual claim in the reasoning or in extra text, even though the final answer is correct. Extra true facts are coded **N**. Uncertain or merely vague wording is coded **N**.

**Coder.** The codes were assigned by the AI assistant (Grok Bot) used in this project. There was no human verification of these codes. That is a limitation of the study, not a step still scheduled.

**Reproduction.** `src/power_and_invented.py` (invoked by `run_analysis.sh`) recomputes the rate and Wilson 95% confidence interval from the coded sheet. It does not re-assign the Y/N codes.
