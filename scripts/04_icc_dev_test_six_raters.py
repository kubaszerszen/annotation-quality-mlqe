import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Project\data\dev_test")


def load_ratings(path, k):
    """Read one TSV file and keep only sentences with exactly k ratings."""
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
    scores = df["scores"].apply(ast.literal_eval)
    scores = scores[scores.apply(len) == k]
    return np.array(scores.tolist(), dtype=float)


def compute_icc(ratings):
    """ICC(1,1) for a (sentences x raters) array."""
    n, k = ratings.shape
    sentence_means = ratings.mean(axis=1)
    grand_mean = ratings.mean()
    ms_between = k * ((sentence_means - grand_mean) ** 2).sum() / (n - 1)
    ms_within = ((ratings - sentence_means[:, None]) ** 2).sum() / (n * (k - 1))
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


def bootstrap_ci(ratings, reps=1000, seed=42):
    """95% confidence interval for the ICC, by resampling sentences."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sample_idx = rng.integers(0, len(ratings), size=len(ratings))
        values.append(compute_icc(ratings[sample_idx]))
    return np.percentile(values, [2.5, 97.5])


patterns = [("dev", "*/dev.*.df.short.tsv"), ("test", "*/test20.*.df.short.tsv")]

results = []
for split, pattern in patterns:
    for path in sorted(DATA_DIR.glob(pattern)):
        pair = path.name.split(".")[1]  # e.g. "ende" from dev.ende.df.short.tsv
        ratings = load_ratings(path, k=6)
        if len(ratings) == 0:  # skip files without 6-rater sentences
            continue
        low, high = bootstrap_ci(ratings)
        results.append({
            "pair": pair,
            "split": split,
            "sentences": len(ratings),
            "icc": compute_icc(ratings),
            "ci_low": low,
            "ci_high": high,
            "mean_score": ratings.mean(),
            "std_of_means": ratings.mean(axis=1).std(),
        })

table = pd.DataFrame(results).round(3).sort_values(["std_of_means"], ascending=False)
print(table.to_string(index=False))