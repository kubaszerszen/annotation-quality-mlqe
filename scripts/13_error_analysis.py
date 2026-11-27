"""Error analysis of the LLM judge.

1. For each language pair, list the sentences where the judge and humans
   disagree most (compared by rank, so the judge's harshness does not matter).
2. Compare the spread of ratings of human group A vs group B.

Usage: python scripts\\13_error_analysis.py test v1
"""
import sys
from pathlib import Path

import pandas as pd

SPLIT = sys.argv[1]
VERSION = sys.argv[2]
TOP_N = 10

JUDGE_DIR = Path("data") / "judge"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

inputs = pd.read_csv(JUDGE_DIR / f"judge_input_{SPLIT}.csv")
human = pd.read_csv(JUDGE_DIR / f"human_reference_{SPLIT}.csv")
scores = pd.read_csv(JUDGE_DIR / f"judge_scores_{SPLIT}_{VERSION}.csv")

# Find the column with the judge's score (name may differ between scripts)
numeric_cols = [c for c in scores.select_dtypes("number").columns if c != "sample_id"]
score_col = next((c for c in ("llm_score", "score") if c in scores.columns), numeric_cols[0])
print(f"Using '{score_col}' as the judge score column")

df = (
    inputs[["sample_id", "pair", "source", "translation"]]
    .merge(human[["sample_id", "human_mean", "group_a_mean", "group_b_mean"]], on="sample_id")
    .merge(scores[["sample_id", score_col]].rename(columns={score_col: "llm_score"}), on="sample_id")
)

# Rank within each pair (0 = worst, 1 = best), so only the ordering matters
df["human_rank"] = df.groupby("pair")["human_mean"].rank(pct=True)
df["llm_rank"] = df.groupby("pair")["llm_score"].rank(pct=True)
df["rank_diff"] = df["llm_rank"] - df["human_rank"]  # > 0: judge kinder than humans

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

for pair, group in df.groupby("pair"):
    print(f"\n===== {pair}: judge much HARSHER than humans =====")
    print(group.nsmallest(TOP_N // 2, "rank_diff")[
        ["human_mean", "llm_score", "source", "translation"]].to_string(index=False))
    print(f"\n===== {pair}: judge much KINDER than humans =====")
    print(group.nlargest(TOP_N // 2, "rank_diff")[
        ["human_mean", "llm_score", "source", "translation"]].to_string(index=False))

print("\n===== Spread (std) of group A vs group B means =====")
print(df.groupby("pair")[["group_a_mean", "group_b_mean"]].agg(["mean", "std"]).round(2))

out = REPORT_DIR / f"error_analysis_{SPLIT}_{VERSION}.csv"
df.sort_values(["pair", "rank_diff"]).to_csv(out, index=False)
print(f"\nSaved all rows to {out}")