"""Per-model overall accuracy / hallucination (all 4 styles, 480 rows) with 95% percentile bootstrap
CIs resampling questions (B and seed from config)."""
import numpy as np, pandas as pd
from common import load_config, p

cfg = load_config(); rng = np.random.default_rng(cfg["seed"]); B = cfg["stats"]["bootstrap_B"]
df = pd.read_csv(p(cfg["paths"]["results_csv"]), dtype={"question_id": str}, keep_default_na=False)
rows = []
for m, g in df.groupby("model"):
    for var in ["correct", "hallucination"]:
        u = g.groupby("question_id")[var].mean().values; n = len(u)
        bs = [u[rng.integers(0, n, n)].mean() for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"model": m, "metric": "accuracy" if var == "correct" else "hallucination",
                     "mean": u.mean(), "lo": lo, "hi": hi, "n_rows": len(g), "n_events": int(g[var].sum())})
o = pd.DataFrame(rows); o.to_csv(p(cfg["paths"]["tables_dir"]) / "model_overall_ci.csv", index=False); print(o.to_string(index=False))
