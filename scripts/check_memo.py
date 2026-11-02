"""Check docs/memo.md against the limits of AC-H-04."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MEMO = Path(__file__).parents[1] / "docs" / "memo.md"
SECTIONS = (
    "Executive summary",
    "Problem",
    "Method",
    "Results",
    "Implications for a PM",
    "Limitations",
)
MAX_WORDS = 2000
MAX_SUMMARY_WORDS = 150
MAX_FIGURES = 4
TABLES = 1


def problems(text: str) -> list[str]:
    """Return every limit the memo breaks."""
    found: list[str] = []
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    if headings != list(SECTIONS):
        found.append(f"sections are {headings}")
    words = len(text.split())
    if words > MAX_WORDS:
        found.append(f"{words} words, limit {MAX_WORDS}")
    summary = text.split("## Executive summary", 1)[-1].split("\n## ", 1)[0]
    summary_words = len(summary.split())
    if summary_words > MAX_SUMMARY_WORDS:
        found.append(f"summary has {summary_words} words, limit {MAX_SUMMARY_WORDS}")
    figures = len(re.findall(r"^!\[", text, flags=re.MULTILINE))
    if figures > MAX_FIGURES:
        found.append(f"{figures} figures, limit {MAX_FIGURES}")
    tables = len(re.findall(r"^\|---", text, flags=re.MULTILINE))
    if tables != TABLES:
        found.append(f"{tables} tables, expected {TABLES}")
    return found


def main() -> None:
    """Print the counts and exit non-zero on a broken limit."""
    text = MEMO.read_text(encoding="utf-8")
    found = problems(text)
    print(f"{len(text.split())} words")
    for problem in found:
        print(problem)
    if found:
        sys.exit(1)


if __name__ == "__main__":
    main()
