"""Scatter plot: human mean score vs LLM judge score, one panel per language pair.

Reads reports/error_analysis_{split}_{version}.csv (made by script 13).

Usage: python scripts\\15_plot_scatter.py test v1
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

SPLIT = sys.argv[1]
VERSION = sys.argv[2]

REPORT_DIR = Path("reports")
df = pd.read_csv(REPORT_DIR / f"error_analysis_{SPLIT}_{VERSION}.csv")

pairs = sorted(df["pair"].unique())
fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), sharex=True, sharey=True)

for ax, pair in zip(axes, pairs):
    group = df[df["pair"] == pair]
    rho, _ = spearmanr(group["human_mean"], group["llm_score"])
    ax.scatter(group["human_mean"], group["llm_score"], alpha=0.5, s=18)
    ax.plot([0, 100], [0, 100], color="grey", linestyle="--", linewidth=1)  # perfect agreement
    ax.set_title(f"{pair} (Spearman = {rho:.2f}, n = {len(group)})")
    ax.set_xlabel("Human mean score")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

axes[0].set_ylabel("LLM judge score")
fig.suptitle(f"Human vs LLM judge ({SPLIT}, {VERSION}); dashed line = identical scores")
fig.tight_layout()

out = REPORT_DIR / f"scatter_human_vs_judge_{SPLIT}_{VERSION}.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")