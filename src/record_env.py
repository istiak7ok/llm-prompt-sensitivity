"""Save the exact environment of a run to results/env/environment.json (standard library only):
Ollama version, model digests (pins the exact weights behind each tag), Python/OS, GPU name if
nvidia-smi is available, and the SHA-256 of the frozen question file."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

from common import http_json, load_config, p


def main():
    cfg = load_config()
    host = cfg["ollama"]["host"].rstrip("/")
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": sys.version, "os": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor()}
    try:
        env["ollama_version"] = http_json(host + "/api/version", timeout=10)["version"]
        tags = http_json(host + "/api/tags", timeout=10).get("models", [])
        env["models"] = {t["name"]: {"digest": t.get("digest"), "size": t.get("size"),
                                     "details": t.get("details")} for t in tags}
    except Exception as e:
        env["ollama_error"] = str(e)
    try:
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        env["gpu"] = None
    qf = p(cfg["data"]["questions_file"])
    env["questions_sha256"] = hashlib.sha256(qf.read_bytes()).hexdigest()
    env["config"] = cfg
    out = p("results/env/environment.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps({k: env[k] for k in ("ollama_version", "gpu", "questions_sha256") if k in env}, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
