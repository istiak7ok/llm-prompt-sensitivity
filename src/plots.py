"""Figures (matplotlib only; 300-dpi PNG + PDF) from the tables written by stats.py/metrics.py.
  fig1_accuracy_by_style      grouped bars: accuracy per model x prompt style, 95% CI
  fig2_hallucination_by_style grouped bars: hallucination rate per model x prompt style, 95% CI
  fig3_consistency_spread     consistency rate and accuracy spread per model, 95% CI
  fig4_accuracy_change        line plot: accuracy change relative to the simple prompt
  fig5_error_types            stacked bars: error-type shares per model
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config, p, display_model

LABEL = {"simple": "Simple", "detailed": "Detailed", "polite": "Polite", "cot": "Chain-of-thought"}
ERR_COLORS = {"wrong_fact": "#C44E52", "incomplete": "#DD8452", "refusal": "#8C8C8C", "format": "#937860"}


def new_fig(w=7.2, h=3.8):
    return plt.subplots(figsize=(w, h), layout="constrained")


def outside_legend(fig, ax, ncol=3, fontsize=8):
    """Place the legend above the axes, outside the data area (figure legend)."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.6,
        columnspacing=1.15,
        handletextpad=0.5,
        borderaxespad=0.25,
    )


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def grouped(ci, metric, styles, models, ylabel, out, name):
    d = ci[ci.metric == metric]
    fig, ax = new_fig(7.4, 3.9)
    w = 0.8 / len(models)
    x = np.arange(len(styles))
    for i, m in enumerate(models):
        r = d[d.model == m].set_index("prompt_style").loc[styles]
        ax.bar(
            x - 0.4 + w * (i + 0.5),
            100 * r["mean"],
            w,
            label=display_model(m),
            yerr=[100 * (r["mean"] - r.lo), 100 * (r.hi - r["mean"])],
            capsize=2,
        )
    ax.set_xticks(x, [LABEL.get(s, s) for s in styles])
    ax.set_ylabel(ylabel)
    lo_all, hi_all = 100 * d.lo.min(), 100 * d.hi.max()
    if lo_all > 50:  # near-ceiling accuracy: zoom in and say so
        ax.set_ylim(5 * np.floor((lo_all - 1) / 5), 100.5)
        ax.set_ylabel(ylabel + "\n(axis truncated)")
    else:
        ax.set_ylim(0, max(1.0, 1.15 * hi_all))
    outside_legend(fig, ax, ncol=min(4, len(models)))
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    styles = cfg["prompt_styles"]
    tdir, out = p(cfg["paths"]["tables_dir"]), p(cfg["paths"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    ci = pd.read_csv(tdir / f"bootstrap_ci{args.tag}.csv")
    mr = pd.read_csv(tdir / f"model_level_ci{args.tag}.csv")
    et = pd.read_csv(tdir / f"error_types{args.tag}.csv")
    models = list(dict.fromkeys(ci.model))

    grouped(ci, "accuracy", styles, models, "Accuracy (%)", out, f"fig1_accuracy_by_style{args.tag}")
    grouped(ci, "hallucination", styles, models, "Hallucination rate (%)", out, f"fig2_hallucination_by_style{args.tag}")

    fig, ax = new_fig(7.2, 3.8)
    for j, (met, lab) in enumerate(
        [("consistency_rate", "Consistency rate"), ("sensitivity_spread_PS", "Accuracy spread (PS)")]
    ):
        r = mr[mr.metric == met].set_index("model").loc[models]
        ax.bar(
            np.arange(len(models)) + (j - 0.5) * 0.38,
            100 * r["mean"],
            0.38,
            label=lab,
            yerr=[100 * (r["mean"] - r.lo), 100 * (r.hi - r["mean"])],
            capsize=2,
        )
    ax.set_xticks(range(len(models)), [display_model(m) for m in models])
    ax.set_ylabel("%")
    outside_legend(fig, ax, ncol=2)
    save(fig, out, f"fig3_consistency_spread{args.tag}")

    fig, ax = new_fig(7.2, 3.8)
    acc = ci[ci.metric == "accuracy"]
    for m in models:
        r = acc[acc.model == m].set_index("prompt_style").loc[styles]["mean"] * 100
        ax.plot([LABEL.get(s, s) for s in styles], r - r[styles[0]], marker="o", label=display_model(m))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"Accuracy change vs {LABEL[styles[0]]} (pp)")
    outside_legend(fig, ax, ncol=4)
    save(fig, out, f"fig4_accuracy_change{args.tag}")

    agg = et.drop(columns="prompt_style").groupby("model").sum()
    agg = agg.div(agg.sum(axis=1), axis=0) * 100
    agg = agg.rename(index=display_model)
    fig, ax = new_fig(7.2, 3.8)
    bottom = np.zeros(len(agg))
    n_err = 0
    for c, col in ERR_COLORS.items():
        if c in agg:
            ax.bar(agg.index, agg[c], bottom=bottom, color=col, label=c.replace("_", " "))
            bottom += agg[c].values
            n_err += 1
    ax.set_ylabel("% of all answers")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    outside_legend(fig, ax, ncol=max(1, n_err))
    save(fig, out, f"fig5_error_types{args.tag}")
    print("figures ->", out)


if __name__ == "__main__":
    main()
