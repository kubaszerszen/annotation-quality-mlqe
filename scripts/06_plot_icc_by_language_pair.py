"""Plot ICC per language pair (train data) and ICC vs spread of sentence quality.

Input:  reports/icc_train_all_pairs.csv (created by 05_icc_train_all_pairs.py)
Output: reports/icc_by_language_pair.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

table = pd.read_csv(REPORTS_DIR / "icc_train_all_pairs.csv")

# nicer labels: "roen" -> "ro-en"; sorted so the chart reads from low to high ICC
table["label"] = table["pair"].str[:2] + "-" + table["pair"].str[2:]
table = table.sort_values("icc").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left chart: ICC with 95% confidence interval for each language pair
ax = axes[0]
errors = np.vstack([table["icc"] - table["ci_low"], table["ci_high"] - table["icc"]])
ax.errorbar(table["icc"], range(len(table)), xerr=errors, fmt="o", capsize=4)
ax.set_yticks(range(len(table)))
ax.set_yticklabels(table["label"])
ax.set_xlim(0, 1)
ax.set_xlabel("ICC (3 raters per sentence)")
ax.set_title("Inter-annotator agreement by language pair")

# right chart: does agreement follow how much sentence quality varies?
ax = axes[1]
ax.scatter(table["std_of_means"], table["icc"])
for _, row in table.iterrows():
    ax.annotate(row["label"], (row["std_of_means"], row["icc"]),
                textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("Spread of sentence quality (std of sentence means)")
ax.set_ylabel("ICC")
ax.set_title("Agreement vs spread of quality")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "icc_by_language_pair.png", dpi=150)
print("saved", REPORTS_DIR / "icc_by_language_pair.png")
