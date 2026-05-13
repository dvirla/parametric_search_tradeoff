"""
Remove rows whose paraphrased_question is a yes/no question from
atomic_fact_confidence.csv using a liberal heuristic:
  - question heuristic: paraphrased_question starts with an auxiliary/copula verb, OR
  - answer signal: ≥3 of 5 samples start with "Yes" or "No"

Rows that match are dropped so that evaluate_atomic_fact_confidence.py
will regenerate them (its resumability logic skips rows already in the file).
"""

import csv
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_confidence.csv')

SAMPLE_COLS = ['sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']

YN_STARTERS = (
    'did ', 'do ', 'does ', 'is ', 'are ', 'was ', 'were ',
    'has ', 'have ', 'can ', 'could ', 'would ', 'will ', 'should ',
    'is it', 'does it',
)
YN_ANSWER_PAT = re.compile(r'^\s*(yes|no)[.,!?;\s]', re.IGNORECASE)


def is_yn_row(row: dict) -> bool:
    q = (row.get('paraphrased_question') or '').strip().lower()
    if q.startswith(YN_STARTERS):
        return True
    yn_votes = sum(1 for col in SAMPLE_COLS if YN_ANSWER_PAT.match(row.get(col) or ''))
    return yn_votes >= 3


def main() -> None:
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    keep, removed = [], []
    for row in rows:
        (removed if is_yn_row(row) else keep).append(row)

    print(f"Total rows : {len(rows):,}")
    print(f"Removed    : {len(removed):,}  ({len(removed)/len(rows):.1%})  — yes/no by liberal heuristic")
    print(f"Kept       : {len(keep):,}  ({len(keep)/len(rows):.1%})")

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keep)

    print(f"\nWrote {len(keep):,} rows back to {CSV_PATH}")
    print("Run evaluate_atomic_fact_confidence.py to regenerate the removed rows.")


if __name__ == '__main__':
    main()
