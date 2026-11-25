"""Recompute (1) invented-detail rate and Wilson CI from the coded sample and
(2) McNemar power / ceiling minimum detectable effect.

The Y/N invented-detail codes themselves are judgements in
results/audit/invented_detail_sample_coded.csv (coder = AI assistant; no human
verification). This script only summarises that sheet and re-runs the power
calculations reported in the paper.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
import numpy as np

from common import load_config, p, display_model


def mcnemar_power(delta: float, disc: float, N=120, B=10_000, seed=2026, alpha=0.05) -> float:
    """Power of an exact two-sided McNemar test.

    Each question is independently discordant with probability `disc`. Given a
    discordant pair, the probability it favours the better style is
    0.5 + delta / (2 * disc), so the expected accuracy difference is `delta`.
    """
    rng = np.random.default_rng(seed)
    n_disc = rng.binomial(N, disc, size=B)
    p_b = 0.5 + delta / (2 * disc)
    b = rng.binomial(n_disc, p_b)
    c = n_disc - b
    n = b + c
    k = np.minimum(b, c)
    pval = np.ones(B)
    mask = n > 0
    # standard doubling rule for an exact two-sided binomial test
    pval[mask] = np.minimum(1.0, 2 * binom.cdf(k[mask], n[mask], 0.5))
    return float((pval < alpha).mean())


def ceiling_power(k: int, N=120, critical=6) -> float:
    """Power when discordance is one-sided (better style never uniquely fails).

    The exact McNemar test (alpha=0.05, two-sided) rejects iff the worse style
    has at least `critical` unique failures (c >= 6 when b = 0).
    """
    if k <= 0:
        return 0.0
    return float(1 - binom.cdf(critical - 1, N, k / N))


def main():
    cfg = load_config()
    tdir = p(cfg["paths"]["tables_dir"])
    tdir.mkdir(parents=True, exist_ok=True)

    coded = pd.read_csv(
        p(cfg["paths"]["audit_dir"]) / "invented_detail_sample_coded.csv",
        dtype=str,
        keep_default_na=False,
    )
    x = int((coded.invented_detail == "Y").sum())
    n = len(coded)
    lo, hi = proportion_confint(x, n, alpha=0.05, method="wilson")
    by_model = (
        coded.groupby("model", sort=False)
        .invented_detail.apply(lambda v: int((v == "Y").sum()))
        .to_dict()
    )
    rows = [{
        "n": n,
        "n_yes": x,
        "rate": x / n,
        "ci_lo": lo,
        "ci_hi": hi,
        "coder": "AI assistant (Grok Bot)",
        "human_verified": "no",
        "seed": 2026,
        "design": "stratified random sample, 25 automatically correct pass-1 rows per model",
        "by_model": "; ".join(
            f"{display_model(m)} {by_model.get(m, 0)}/25" for m in coded.model.drop_duplicates()
        ),
    }]
    pd.DataFrame(rows).to_csv(tdir / "invented_detail_summary.csv", index=False)
    print(f"invented detail: {x}/{n} = {x/n:.3f}; Wilson 95% CI [{lo:.3f}, {hi:.3f}]")

    power_rows = []
    for disc, delta in [(0.15, 0.10), (0.20, 0.10), (0.25, 0.10), (0.20, 0.08)]:
        pw = mcnemar_power(delta, disc)
        power_rows.append({
            "setting": "pre-run simulation (exact McNemar, N=120, alpha=0.05 two-sided, B=10000, seed 2026)",
            "delta_pp": round(100 * delta, 2),
            "discordance": disc,
            "power": pw,
        })
        print(f"power  {100*delta:.0f} pp at {100*disc:.0f}% discordance: {pw:.3f}")

    for k in range(0, 13):
        pw = ceiling_power(k)
        power_rows.append({
            "setting": "ceiling, one-sided discordance (analytic; reject iff c>=6 when b=0)",
            "delta_pp": round(100 * k / 120, 2),
            "discordance": None,
            "power": pw,
        })
    pd.DataFrame(power_rows).to_csv(tdir / "power_mcnemar.csv", index=False)
    for k in (3, 6, 8):
        print(f"ceiling MDE  {k}/120 = {100*k/120:.2f} pp  power={ceiling_power(k):.3f}")


if __name__ == "__main__":
    main()
