import pandas as pd

import ast  # turns the text "[23, 23, 23]" into a real list

path = r"C:\Project\data\ro-en-train\train.roen.df.short.tsv"

df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)

# the scores column is stored as text, so convert it into lists of numbers
df["scores"] = df["scores"].apply(ast.literal_eval)

# spread of the ratings per sentence: highest score minus lowest score
df["spread"] = df["scores"].apply(lambda s: max(s) - min(s))

print(len(df), "sentences")
print(df["spread"].describe())


df[["original", "translation", "scores", "mean"]].head(200).to_csv(
    r"C:\Project\preview.csv", index=False, encoding="utf-8-sig"
)