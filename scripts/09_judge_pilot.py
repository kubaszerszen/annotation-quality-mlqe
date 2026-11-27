import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = ROOT / "data" / "judge"

# safety check: an API key would switch billing from your plan to the API
if os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY is set - unset it before running.")

INSTRUCTIONS = (
    "You are evaluating machine translation quality. The input contains numbered items; "
    "each item has a source sentence and its machine translation. "
    "Rate the quality of each translation from 0 (meaningless) to 100 (perfect), "
    "judging both meaning and fluency. "
    "Answer with ONLY a JSON list of integers, one per item, in the same order. No other text."
)

df = pd.read_csv(JUDGE_DIR / "judge_input_dev.csv").sample(n=20, random_state=1)

# the items go to the model through stdin
lines = []
for number, row in enumerate(df.itertuples(), start=1):
    lines.append(f"{number}. SOURCE: {row.source}\n   TRANSLATION: {row.translation}")
stdin_text = "\n".join(lines)

# run in an empty temporary folder, so Claude Code does not load any project context
with tempfile.TemporaryDirectory() as empty_dir:
    proc = subprocess.run(
        ["claude", "-p", INSTRUCTIONS, "--output-format", "json",
         "--model", "sonnet", "--max-turns", "1"],
        input=stdin_text, capture_output=True, text=True, encoding="utf-8",
        cwd=empty_dir,
    )

if proc.returncode != 0:
    raise SystemExit("claude failed:\n" + proc.stderr + proc.stdout)

payload = json.loads(proc.stdout)
print("keys in the JSON response:", list(payload.keys()))
print("raw answer:", payload["result"])

scores = json.loads(payload["result"].strip().strip("`").removeprefix("json").strip())
assert len(scores) == len(df), f"expected {len(df)} scores, got {len(scores)}"

df["llm_score"] = scores
df[["sample_id", "pair", "llm_score"]].to_csv(JUDGE_DIR / "pilot_scores.csv", index=False)
print(df[["sample_id", "llm_score"]].to_string(index=False))