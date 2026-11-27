"""Write the error analysis as a readable Markdown report.

Reads reports/error_analysis_{split}_{version}.csv (made by script 13).

Usage: python scripts\\14_error_report_md.py test v1
"""
import sys
from pathlib import Path

import pandas as pd

SPLIT = sys.argv[1]
VERSION = sys.argv[2]
TOP_N = 10  # cases per direction per language pair

REPORT_DIR = Path("reports")
df = pd.read_csv(REPORT_DIR / f"error_analysis_{SPLIT}_{VERSION}.csv")


def write_case(lines, row):
    lines.append(f"**Humans: {row.human_mean:.0f}  |  LLM judge: {row.llm_score:.0f}**\n")
    lines.append(f"- Source: {row.source}")
    lines.append(f"- Translation: {row.translation}\n")


lines = [f"# Judge vs humans: biggest disagreements ({SPLIT}, {VERSION})\n"]
lines.append("Cases are chosen by rank difference, so the judge's overall "
             "harshness does not matter.\n")

for pair, group in df.groupby("pair"):
    lines.append(f"## {pair}\n")

    lines.append("### Judge much harsher than humans\n")
    for row in group.nsmallest(TOP_N, "rank_diff").itertuples():
        write_case(lines, row)

    lines.append("### Judge much kinder than humans\n")
    for row in group.nlargest(TOP_N, "rank_diff").itertuples():
        write_case(lines, row)

out = REPORT_DIR / f"error_report_{SPLIT}_{VERSION}.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {out}")