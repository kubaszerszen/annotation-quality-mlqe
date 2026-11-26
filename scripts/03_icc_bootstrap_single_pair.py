import ast

import numpy as np
import pandas as pd

def compute_icc(ratings):
    n, k = ratings.shape
    sentence_means = ratings.mean(axis=1)
    grand_mean = ratings.mean()
    ms_between = k * ((sentence_means - grand_mean) ** 2).sum() / (n - 1)
    ms_within = ((ratings - sentence_means[:, None]) ** 2).sum() / (n * (k - 1))
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


#path = r"C:\Project\data\ro-en-train\train.roen.df.short.tsv"
path = r"C:\Project\data\en-de-train\train.ende.df.short.tsv"  #path to en-de


df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
df["scores"] = df["scores"].apply(ast.literal_eval)

# keep only sentences with exactly 3 ratings, so every row has the same shape
df = df[df["scores"].apply(len) == 3]
ratings = np.array(df["scores"].tolist(), dtype=float)  # shape: (sentences, 3)

n, k = ratings.shape
sentence_means = ratings.mean(axis=1)
grand_mean = ratings.mean()

# variation between sentences
ms_between = k * ((sentence_means - grand_mean) ** 2).sum() / (n - 1)

# variation inside sentences (disagreement between raters)
ms_within = ((ratings - sentence_means[:, None]) ** 2).sum() / (n * (k - 1))

icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
print("ICC:", round(icc, 3))


rng = np.random.default_rng(seed=42)  # fixed seed so results are reproducible
boot_values = []
for _ in range(1000):
    # draw n sentences with replacement
    sample_idx = rng.integers(0, len(ratings), size=len(ratings))
    boot_values.append(compute_icc(ratings[sample_idx]))

low, high = np.percentile(boot_values, [2.5, 97.5])
print("ICC:", round(compute_icc(ratings), 3))
print("95% confidence interval:", round(low, 3), "-", round(high, 3))