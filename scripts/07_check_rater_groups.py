import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dev_test"


def load_ratings(path, k):
    """Read one TSV file and keep only sentences with exactly k ratings."""
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
    scores = df["scores"].apply(ast.literal_eval)
    scores = scores[scores.apply(len) == k]
    return np.array(scores.tolist(), dtype=float)


patterns = [("dev", "*/dev.*.df.short.tsv"), ("test", "*/test20.*.df.short.tsv")]

for split, pattern in patterns:
    for path in sorted(DATA_DIR.glob(pattern)):
        pair = path.name.split(".")[1]
        if pair not in ("ende", "roen", "sien"):
            continue
        ratings = load_ratings(path, k=6)
        mean_a = ratings[:, :3].mean(axis=1)   # mean of the first group (positions 1-3)
        mean_b = ratings[:, 3:].mean(axis=1)   # mean of the second group (positions 4-6)
        ceiling = np.corrcoef(mean_a, mean_b)[0, 1]
        corr = np.corrcoef(ratings.T)  # 6 x 6 correlations between rater positions

        within = [corr[i, j] for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]]
        across = [corr[i, j] for i in range(3) for j in range(3, 6)]
        print(pair, split,
              "within-trio:", round(np.mean(within), 3),
              "across-trio:", round(np.mean(across), 3))
        print(pair, split, "ceiling:", round(ceiling, 3))