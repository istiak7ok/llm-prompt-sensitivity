"""Shared helpers. Standard library only, so run_queries.py works on a fresh Windows Python."""
from __future__ import annotations

import json
import re
import string
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Display names used in the paper, tables and figures. Tags stay in data files.
MODEL_DISPLAY = {
    "llama3.1:8b": "Llama-3.1-8B",
    "mistral:7b": "Mistral-7B",
    "gemma2:9b": "Gemma-2-9B",
    "qwen2.5:7b": "Qwen2.5-7B",
}


def display_model(tag: str) -> str:
    return MODEL_DISPLAY.get(str(tag), str(tag))



def load_config(path: str | Path = ROOT / "config.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def p(rel: str) -> Path:
    return ROOT / rel


# ---------------------------------------------------------------- JSONL
def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written last line after a crash: ignore, it will be re-queried
    return rows


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- Ollama (urllib, no extra packages)
def http_json(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_chat(cfg: dict, model: str, prompt: str, num_predict: int, retries: int = 3) -> dict:
    """One single-turn /api/chat call with greedy decoding. Returns text + latency counters."""
    opts = dict(cfg["decoding"], num_predict=num_predict)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "options": opts, "keep_alive": cfg["ollama"]["keep_alive"]}
    url = cfg["ollama"]["host"].rstrip("/") + "/api/chat"
    last_err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            js = http_json(url, payload, timeout=cfg["ollama"]["timeout_s"])
            return {
                "response": js["message"]["content"],
                "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": js.get("prompt_eval_count"),
                "output_tokens": js.get("eval_count"),
                "load_s": round((js.get("load_duration") or 0) / 1e9, 3),
                "gen_s": round((js.get("eval_duration") or 0) / 1e9, 3),
                "done_reason": js.get("done_reason"),
            }
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {retries} tries: {last_err}")


# ---------------------------------------------------------------- normalisation + extraction
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation + "‘’´`“”–—")


def normalize(s: str) -> str:
    """SQuAD/TriviaQA-style: lowercase, punctuation -> space, drop articles, collapse spaces.
    Thousands separators are removed first so '5,700' == '5700'."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(s).lower()).replace("_", " ")
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


_FINAL = re.compile(r"final[ \t]*answer[ \t]*(?:is)?[ \t]*[:：\-]?[ \t]*(.*)", re.I)


def extract_final_answer(raw: str) -> tuple[str, str]:
    """Return (answer, method). method = 'final_line' | 'fallback_short' | 'fallback_long' | 'none'.
    All four prompt styles ask for a 'Final answer: ...' line, so extraction is identical
    across styles; the fallback only accepts a short last/first line."""
    text = str(raw).strip()
    hits = [m.group(1).strip().strip("*").strip().rstrip(".").strip().strip("*\"'").strip()
            for m in _FINAL.finditer(text)]
    hits = [h for h in hits if h]
    if hits:
        return hits[-1], "final_line"
    lines = [ln.strip().strip("*").strip() for ln in text.splitlines() if ln.strip()]
    if lines:  # no "Final answer:" line -> use the LAST line (where conclusions usually are)
        last = lines[-1].rstrip(".")
        # e.g. a CoT reply ending "Therefore, ... 253 bones." becomes 'fallback_long'
        return last, ("fallback_short" if len(last.split()) <= 8 else "fallback_long")
    return "", "none"


def contains_phrase(haystack_norm: str, needle_norm: str) -> bool:
    return bool(needle_norm) and f" {needle_norm} " in f" {haystack_norm} "
