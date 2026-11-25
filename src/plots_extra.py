"""Extra figures: fig6 verbose-answer rate by style, fig7 label vs surface consistency,
fig8 pass-2 stability. PNG (300 dpi) + PDF.

Legends sit above the axes (outside the data area) so they never overlap bars.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_config, p, display_model

cfg = load_config()
T = p(cfg["paths"]["tables_dir"])
F = p(cfg["paths"]["figures_dir"])
STY = ["simple", "detailed", "polite", "cot"]
STY_LAB = {"simple": "Simple", "detailed": "Detailed", "polite": "Polite", "cot": "CoT"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def new_fig(w=7.4, h=4.0):
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


def save(fig, name):
    fig.savefig(F / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


f = pd.read_csv(T / "format_compliance_by_style.csv")
models = sorted(f.model.unique())
fig, ax = new_fig()
w = 0.2
x = np.arange(len(models))
for i, s in enumerate(STY):
    v = [f[(f.model == m) & (f.prompt_style == s)].verbose_rate.iloc[0] * 100 for m in models]
    ax.bar(x + (i - 1.5) * w, v, w, label=STY_LAB.get(s, s))
ax.set_xticks(x, [display_model(m) for m in models])
ax.set_ylabel("Final answers longer than 5 words (%)")
ax.set_ylim(0, 105)
outside_legend(fig, ax, ncol=4)
save(fig, "fig6_verbosity_by_style")

s = pd.read_csv(T / "strict_consistency.csv")
fig, ax = new_fig()
x = np.arange(len(s))
w = 0.35
for j, (col, lab) in enumerate(
    [("label_consistency", "same correctness label (CR)"), ("strict_surface_consistency", "identical answer string")]
):
    pre = "label" if col.startswith("label") else "strict"
    m = s[col] * 100
    lo = (s[col] - s[f"{pre}_ci_lo"]) * 100
    hi = (s[f"{pre}_ci_hi"] - s[col]) * 100
    ax.bar(x + (j - 0.5) * w, m, w, yerr=[lo, hi], capsize=3, label=lab)
ax.set_xticks(x, [display_model(m) for m in s.model])
ax.set_ylabel("Questions consistent across 4 styles (%)")
ax.set_ylim(0, 110)
outside_legend(fig, ax, ncol=2)
save(fig, "fig7_label_vs_surface_consistency")

ps = pd.read_csv(T / "pass2_stability.csv")
fig, ax = new_fig(7.6, 4.0)
x = np.arange(len(ps))
w = 0.27
for j, (c, lab) in enumerate(
    [
        ("identical_raw_response", "identical raw response"),
        ("identical_extracted_answer", "identical extracted answer"),
        ("identical_correct_label", "identical correctness label"),
    ]
):
    ax.bar(x + (j - 1) * w, ps[c] * 100, w, label=lab)
ax.set_xticks(x, [display_model(m) for m in ps.model])
ax.set_ylim(75, 101)
ax.set_ylabel("Pass 1 vs pass 2 (%), n=120 each")
outside_legend(fig, ax, ncol=3, fontsize=8)
save(fig, "fig8_pass2_stability")
print("ok")
