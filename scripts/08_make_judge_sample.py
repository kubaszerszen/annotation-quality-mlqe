import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dev_test"
OUT_DIR = ROOT / "data" / "judge"

PAIRS = ("ende", "roen", "sien")
SAMPLE_SIZE = {"dev": 100, "test": 300}
SEED = 42  # fixed seed so the same sentences are drawn every time

patterns = {"dev": "*/dev.*.df.short.tsv", "test": "*/test20.*.df.short.tsv"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

for split, pattern in patterns.items():
    inputs, references = [], []
    for path in sorted(DATA_DIR.glob(pattern)):
        pair = path.name.split(".")[1]
        if pair not in PAIRS:
            continue

        df = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
        df["scores"] = df["scores"].apply(ast.literal_eval)
        df = df[df["scores"].apply(len) == 6]                      # keep 6-score sentences
        df = df.sample(n=SAMPLE_SIZE[split], random_state=SEED)    # random sample

        df["sample_id"] = pair + "-" + split + "-" + df["index"]
        df["pair"] = pair
        df["split"] = split
        df["human_mean"] = df["scores"].apply(lambda s: sum(s) / 6)
        df["group_a_mean"] = df["scores"].apply(lambda s: sum(s[:3]) / 3)
        df["group_b_mean"] = df["scores"].apply(lambda s: sum(s[3:]) / 3)

        # what the judge model will see: NO human scores
        inputs.append(
            df.rename(columns={"original": "source"})[
                ["sample_id", "pair", "split", "source", "translation"]
            ]
        )
        # human scores, kept in a separate file
        references.append(
            df[["sample_id", "pair", "split", "human_mean", "group_a_mean", "group_b_mean"]]
        )

    pd.concat(inputs).to_csv(OUT_DIR / f"judge_input_{split}.csv", index=False, encoding="utf-8")
    pd.concat(references).to_csv(OUT_DIR / f"human_reference_{split}.csv", index=False, encoding="utf-8")
    print(split, "saved", sum(len(x) for x in inputs), "sentences")