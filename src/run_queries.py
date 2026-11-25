"""Query every (model, question, prompt style) cell through Ollama's HTTP API and append one
JSON line per call to results/raw/<model>.jsonl.

* Standard library only: runs on a fresh Windows Python 3.10+ (no pip install needed).
* Resumable: rows already logged (same question_id + prompt_style) are skipped, so you can
  stop with Ctrl+C and simply run the same command again.
* Greedy decoding (temperature 0, top_k 1, seed 42, num_ctx 2048) from config.json.

Examples
  python src/run_queries.py --check                 # is Ollama up, which models are pulled?
  python src/run_queries.py --limit 5               # 5-question pilot on all 4 models
  python src/run_queries.py                         # full run (120 q x 4 styles x 4 models)
  python src/run_queries.py --models qwen2.5:7b     # one model only
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone

from common import append_jsonl, http_json, load_config, ollama_chat, p, read_jsonl


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def build_prompt(cfg: dict, style: str, question: str) -> str:
    return cfg["prompts"][style].format(question=question)


def check(cfg) -> dict:
    host = cfg["ollama"]["host"].rstrip("/")
    try:
        ver = http_json(host + "/api/version", timeout=10)["version"]
        tags = [m["name"] for m in http_json(host + "/api/tags", timeout=10).get("models", [])]
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {host}: {e}\nStart it (Ollama app or `ollama serve`) and retry.")
    print(f"Ollama {ver} at {host}")
    for m in cfg["models"]:
        ok = any(t == m or t == m + ":latest" for t in tags)
        print(f"  {'OK     ' if ok else 'MISSING'} {m}" + ("" if ok else f"   -> run: ollama pull {m}"))
    return {"version": ver, "tags": tags}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="subset of model tags (default: all in config.json)")
    ap.add_argument("--styles", nargs="*", help="subset of prompt styles")
    ap.add_argument("--limit", type=int, help="first N questions only (pilot)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _pilot")
    ap.add_argument("--subset-frac", type=float, help="random fraction of questions (fixed seed 2026), e.g. 0.25 for the pass-2 stability subset")
    ap.add_argument("--check", action="store_true", help="only check Ollama and pulled models")
    args = ap.parse_args()

    cfg = load_config()
    info = check(cfg)
    if args.check:
        return
    questions = read_jsonl(p(cfg["data"]["questions_file"]))
    if args.limit:
        questions = questions[: args.limit]
    if args.subset_frac:
        import random
        k = round(len(questions) * args.subset_frac)
        keep = set(q["question_id"] for q in random.Random(2026).sample(questions, k))
        questions = [q for q in questions if q["question_id"] in keep]
        print(f"subset: {len(questions)} questions (seed 2026)")
    models = args.models or cfg["models"]
    styles = args.styles or cfg["prompt_styles"]

    for model in models:
        out = p(cfg["paths"]["raw_dir"]) / f"{safe_name(model)}{args.tag}.jsonl"
        done = {(r["question_id"], r["prompt_style"]) for r in read_jsonl(out) if "response" in r}
        todo = [(q, s) for q in questions for s in styles if (q["question_id"], s) not in done]
        print(f"\n[{model}] {len(done)} already logged, {len(todo)} to go -> {out}")
        t_start, n = time.perf_counter(), 0
        for q, style in todo:
            prompt = build_prompt(cfg, style, q["question"])
            res = ollama_chat(cfg, model, prompt, cfg["num_predict"][style])
            append_jsonl(out, {
                "question_id": q["question_id"], "model": model, "prompt_style": style,
                "prompt": prompt, "response": res["response"],
                "latency_s": res["latency_s"], "load_s": res["load_s"], "gen_s": res["gen_s"],
                "prompt_tokens": res["prompt_tokens"], "output_tokens": res["output_tokens"],
                "done_reason": res["done_reason"], "num_predict": cfg["num_predict"][style],
                "decoding": cfg["decoding"], "ollama_version": info["version"],
                "host": platform.node(), "os": platform.platform(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            n += 1
            if n % 10 == 0 or n == len(todo):
                el = time.perf_counter() - t_start
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)}  {el/n:.2f} s/call  ETA {eta/60:.1f} min", flush=True)
        if n:
            print(f"[{model}] done: {n} calls, mean {(time.perf_counter()-t_start)/n:.2f} s/call")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run the same command to resume.")
