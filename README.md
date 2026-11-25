This folder is the push-ready public snapshot of the project (code, questions, raw outputs from both passes, graded results, and the analysis script). No GitHub or Zenodo URL has been created yet.

# Prompt sensitivity, consistency and hallucination in open-weight LLMs

How much does the *wording* of a plain factual question change what a local open-weight LLM answers, and how often is that answer confidently wrong?
This repository asks 120 original, source-verified short-answer questions to four 7-9B models (Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Qwen2.5-7B) in four prompt styles (simple, detailed, polite, chain-of-thought) through Ollama. It then measures accuracy, consistency, hallucination, prompt sensitivity and error types.

- Background and gap: `lit_review.md`
- Full method, formulas, statistics, venues and timeline: `design.md`
- **Status (25 Nov 2026, final submission version):** the experiment has been run (pass 1: 1,920 calls; pass 2: 480 calls, 25% subset) on the author's laptop, graded, audited and analysed. Paper: `paper/main.pdf` (source `paper/main.tex`, `paper/paper.md`); submission to-do list: `paper/submission_checklist.md`. Gold answers and the audit were verified by script and by the AI assistant only; there was no human verification. That is a limitation of the study, not a step still scheduled.
- **Headline:** a strict first grader turned Mistral-7B's verbose answers into a 10.2-point accuracy deficit (88.8% rather than 99.0%); style changed answer *form* far more than correctness; false answers were rare (0.2-1.0%) but patterned; temperature-0 outputs were not fully repeatable, though no correctness label changed. Accuracy was near ceiling (98.3-99.6%), so the study is under-powered for small correctness differences. See `paper/paper.md`.
- **Excluded from all results:** `results/raw/qwen2.5_7b_pilot_qwen.jsonl` (pilot) and any `smoke-*` files.

## Repository layout
```
research-paper/
├── README.md, LICENSE, requirements.txt, config.json   # config.json = every setting in one place
├── lit_review.md, design.md, run_analysis.sh
├── paper/        # paper.md, main.tex, references.bib, main.pdf, submission_checklist.md, figures/
├── data/
│   ├── question_bank_authored.psv     # the 120 questions as written (human-readable)
│   ├── questions.jsonl                # verified, frozen set used by the experiment
│   ├── questions_all_with_status.jsonl
│   └── verification_report.csv        # evidence sentence + pinned Wikipedia URL per item; human_verified column is empty (no human check)
├── src/
│   ├── common.py            # stdlib helpers: config, Ollama HTTP client, normalization, answer extraction
│   ├── verify_questions.py  # Wikipedia evidence check + freeze (stdlib)
│   ├── record_env.py        # saves Ollama version, model digests, GPU, data hash (stdlib)
│   ├── run_queries.py       # queries the models, resumable JSONL logging with latency (stdlib)
│   ├── grade.py             # -> results/results.csv + manual coding sheet      (pandas)
│   ├── metrics.py           # accuracy, hallucination, consistency, sensitivity, error types
│   ├── stats.py             # bootstrap CIs, Cochran's Q, McNemar + Holm; Markdown/LaTeX tables
│   ├── plots.py, plots_extra.py   # 8 figures (PNG 300 dpi + PDF)
│   ├── audit_assistant.py, grader_change_log.py, model_overall_ci.py, extra_analysis.py
│   └── kappa.py             # inter-annotator agreement for the manual coding
└── results/
    ├── raw/        # one JSONL per model: prompt, response, latency, tokens, timestamps
    ├── results.csv # question_id, model, prompt_style, model_answer, correct, consistent, hallucination, error_type, ...
    ├── tables/  figures/  audit/  env/
```

## Running on Windows (laptop with an NVIDIA GPU)

### 1. One-time setup
1. **Ollama:** install it from https://ollama.com/download (Windows). It runs in the background and serves `http://localhost:11434`.
2. **Models** (PowerShell, about 20 GB to download in total):
   ```powershell
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   ollama pull gemma2:9b
   ollama pull qwen2.5:7b
   ollama list            # note the IDs/digests; record_env.py saves them too
   ```
3. **Python:** install Python 3.12 or newer from https://www.python.org/downloads/windows/ and tick **"Add python.exe to PATH"**. Check it with `python --version` in a new PowerShell window.
4. **Get this folder** onto the laptop (git clone or copy), then:
   ```powershell
   cd C:\path\to\research-paper
   ```

### 2. Pilot run (about 2-5 minutes)
`run_queries.py` uses only the Python standard library, so nothing has to be installed for it.
```powershell
python src\run_queries.py --check                  # Ollama reachable? all 4 models pulled?
python src\record_env.py                           # saves results\env\environment.json
python src\run_queries.py --limit 5 --tag _pilot   # 5 questions x 4 styles x 4 models = 80 calls
```
It prints seconds per call and an ETA. In another window, `ollama ps` should show each model as `100% GPU`. If it shows CPU, close other GPU-heavy programs.

### 3. Full run
```powershell
python src\run_queries.py                    # pass 1: 1,920 calls; this is the main data
python src\run_queries.py --tag _pass2       # pass 2: same calls again, measures run-to-run stability
```
- **Resumable:** stop anytime with Ctrl+C (or if the laptop sleeps or restarts) and run the **same command** again. Rows already in `results\raw\<model>.jsonl` are skipped.
- Keep the laptop plugged in, and set Windows power mode to "Best performance" during runs.
- Single model: `python src\run_queries.py --models gemma2:9b`

### 4. Analysis
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
bash run_analysis.sh                  # whole analysis + paper tables (Git Bash / WSL / Linux / macOS)
```
`run_analysis.sh` runs, in order: `grade.py` (pass 1 and `--tag _pass2`), `audit_assistant.py` (assistant audit codes and agreement), `grader_change_log.py` (strict v1 vs final v2 grader), `metrics.py`, `stats.py`, `model_overall_ci.py`, `extra_analysis.py` (verbosity, pooled and cluster-permutation tests, surface consistency, H2, pass-2 stability, robustness), `power_and_invented.py` (invented-detail summary and McNemar power / ceiling MDE), `plots.py`, `plots_extra.py`, `kappa.py` (a no-op unless a second coding sheet is filled; this submission has no human second coder), then copies figures and rebuilds the paper tables (`paper/build_tables.py`, `paper/build_paper.py`). Compile the PDF with `cd paper && tectonic -X compile main.tex` (or pdfLaTeX + BibTeX).

## Decoding and reproducibility
Greedy decoding through `/api/chat`: `temperature 0, top_k 1, top_p 1, repeat_penalty 1, seed 42, num_ctx 2048`, with `num_predict` set to 128 (simple/detailed/polite) and 512 (CoT). All settings live in `config.json`.
Even at temperature 0, outputs aren't guaranteed to be bit-identical across runs. Our smoke test reproduced 17 of 20 responses exactly, and all 3 differences were long CoT replies. That's why pass 2 exists.
Analysis seed: 2026 (bootstrap and audit sample). The SHA-256 of the question file used for the run (`data/questions_v1_as_run.jsonl`) is `88b7882986da0a09af9e3e227f0b45896449167ad0647fb6bb1dfa0ea38b1647` and is recorded in `results/env/environment.json`. Ollama `ollama list` short IDs recorded on 25 November 2026 (all Q4_K_M): `llama3.1:8b` 46e0c10c039e, `mistral:7b` 6577803aa9a0, `gemma2:9b` ff02c3702f32, `qwen2.5:7b` 845dbda0ea48. Full 64-character digests from the pre-run snapshot: `llama3.1:8b` 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e, `qwen2.5:7b` 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e (`mistral:7b` and `gemma2:9b` were pulled after that snapshot; see `results/env/ollama_list_2026-11-25.json`).

## Re-verifying the question set (any OS, needs internet)
```powershell
python src\verify_questions.py
```
This downloads each source article, pins its revision, and writes `data\verification_report.csv`. Every row was reviewed by the AI assistant (`ai_verification`, `ai_note`). For this submission version, gold answers were verified by script and by the AI assistant only; there was no human verification (the `human_verified` column is empty). That is a limitation of the study, not a step still scheduled. See `design.md` §3.3 and `data/README.md`.

## Licence
Code: MIT. Question set: CC BY-SA 4.0.

## AI-assistance disclosure (keep this, and adapt it for the paper)
Author: Istiak Hossain Saikot (Independent researcher, Bangladesh). An AI assistant designed the experiment with the author, wrote the Python code, found and verified the citations, reviewed all 120 gold answers, served as the only coder in the grading audit and in a 100-row invented-detail sample, and drafted the manuscript. The author ran the experiments on the author's laptop and is responsible for the final text. Gold answers and the audit were verified by script and by the AI assistant only; there was no human verification of the answer key, the audit codes, or the invented-detail codes. That is a limitation of this study. Targets: arXiv (cs.CL) and/or JSR; see `design.md` §12 and `paper/submission_checklist.md`.
