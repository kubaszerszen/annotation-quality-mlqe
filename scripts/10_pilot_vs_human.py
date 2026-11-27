from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "data" / "judge"

pilot = pd.read_csv(JUDGE_DIR / "pilot_scores.csv")
human = pd.read_csv(JUDGE_DIR / "human_reference_dev.csv")

df = pilot.merge(human[["sample_id", "human_mean"]], on="sample_id")
df["difference"] = df["llm_score"] - df["human_mean"]

print("sentences compared:", len(df))
print("Pearson correlation: ", round(df["llm_score"].corr(df["human_mean"]), 3))
print("Spearman correlation:", round(df["llm_score"].corr(df["human_mean"], method="spearman"), 3))
print("average difference (LLM - human):", round(df["difference"].mean(), 1))

print(df.sort_values("difference", key=abs, ascending=False).round(1).to_string(index=False))