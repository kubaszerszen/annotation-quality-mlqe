import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "data" / "judge"

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"       # dev or test
VERSION = sys.argv[2] if len(sys.argv) > 2 else "v1"      # prompt version label
BATCH_SIZE = 20
MAX_RETRIES = 2

if os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY is set - unset it before running.")

INSTRUCTIONS = (
    "You are evaluating machine translation quality. The input contains numbered items; "
    "each item has a source sentence and its machine translation. "
    "Rate the quality of each translation from 0 (meaningless) to 100 (perfect), "
    "judging both meaning and fluency. "
    "Answer with ONLY a JSON list of integers, one per item, in the same order. No other text."
)

OUT_PATH = JUDGE_DIR / f"judge_scores_{SPLIT}_{VERSION}.csv"


def score_batch(batch):
    """Send one batch of sentences to Claude and return a list of scores."""
    lines = []
    for number, row in enumerate(batch.itertuples(), start=1):
        lines.append(f"{number}. SOURCE: {row.source}\n   TRANSLATION: {row.translation}")
    stdin_text = "\n".join(lines)

    with tempfile.TemporaryDirectory() as empty_dir:
        proc = subprocess.run(
            ["claude", "-p", INSTRUCTIONS, "--output-format", "json",
             "--model", "sonnet", "--max-turns", "1"],
            input=stdin_text, capture_output=True, text=True, encoding="utf-8",
            cwd=empty_dir, timeout=300,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr + proc.stdout)

    answer = json.loads(proc.stdout)["result"]
    start, end = answer.index("["), answer.rindex("]") + 1      # ignore any text around the list
    scores = json.loads(answer[start:end])
    if len(scores) != len(batch):
        raise ValueError(f"expected {len(batch)} scores, got {len(scores)}")
    return scores


data = pd.read_csv(JUDGE_DIR / f"judge_input_{SPLIT}.csv")

# resume: skip sentences that are already scored
done = set()
if OUT_PATH.exists():
    done = set(pd.read_csv(OUT_PATH)["sample_id"])
todo = data[~data["sample_id"].isin(done)]
print(f"{SPLIT}/{VERSION}: {len(done)} already scored, {len(todo)} to do")

for pair, group in todo.groupby("pair"):                      # batches never mix language pairs
    for start in range(0, len(group), BATCH_SIZE):
        batch = group.iloc[start:start + BATCH_SIZE]
        for attempt in range(MAX_RETRIES + 1):
            try:
                scores = score_batch(batch)
                break
            except Exception as error:
                print(f"  attempt {attempt + 1} failed: {str(error)[:200]}")
        else:
            raise SystemExit("batch failed repeatedly - stopping (progress is saved)")

        result = batch[["sample_id", "pair"]].copy()
        result["llm_score"] = scores
        result.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)
        print(f"  {pair}: scored {start + len(batch)}/{len(group)}")

print("done ->", OUT_PATH)