import ast

import numpy as np
import pandas as pd

#path = r"C:\Project\data\ro-en-train\train.roen.df.short.tsv"
path = r"C:\Project\data\en-de-train\train.ende.df.short.tsv"  #path to en-de
df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
df["scores"] = df["scores"].apply(ast.literal_eval)

# average rating of each sentence
df["sentence_mean"] = df["scores"].apply(np.mean)

# disagreement inside each sentence: standard deviation of its 3 ratings
df["within_std"] = df["scores"].apply(lambda s: np.std(s, ddof=1))

print("Between sentences (std of sentence means):", round(df["sentence_mean"].std(), 1))
print("Inside sentences (average std of ratings):", round(df["within_std"].mean(), 1))

print(df["sentence_mean"].describe().round(1)) # how good translation are overall