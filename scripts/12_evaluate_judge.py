import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "data" / "judge"

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
VERSION = sys.argv[2] if len(sys.argv) > 2 else "v1"

scores = pd.read_csv(JUDGE_DIR / f"judge_scores_{SPLIT}_{VERSION}.csv")
human = pd.read_csv(JUDGE_DIR / f"human_reference_{SPLIT}.csv")
df = scores.merge(human.drop(columns=["pair", "split"]), on="sample_id")


def bootstrap_spearman(x, y, reps=1000, seed=42):
    """95% confidence interval for the Spearman correlation."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(spearmanr(x[idx], y[idx])[0])
    return np.percentile(values, [2.5, 97.5])


rows = []
for pair, group in df.groupby("pair"):
    x = group["llm_score"].to_numpy(dtype=float)      # LLM scores
    y = group["human_mean"].to_numpy(dtype=float)     # mean of the 6 human scores
    low, high = bootstrap_spearman(x, y)
    rows.append({
        "pair": pair,
        "n": len(group),
        "spearman": spearmanr(x, y)[0],
        "ci_low": low,
        "ci_high": high,
        "pearson": np.corrcoef(x, y)[0, 1],
        "mean_diff": (x - y).mean(),                  # LLM - human
        "human_ceiling": np.corrcoef(group["group_a_mean"], group["group_b_mean"])[0, 1],
        "vs_group_a": np.corrcoef(x, group["group_a_mean"])[0, 1],
        "vs_group_b": np.corrcoef(x, group["group_b_mean"])[0, 1],
    })

print(f"{SPLIT} / {VERSION}")
print(pd.DataFrame(rows).round(3).to_string(index=False))