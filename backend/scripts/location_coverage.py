#!/usr/bin/env python
"""Read-only location coverage report for the property database.

Usage (PowerShell)::
    python backend/scripts/location_coverage.py
    python backend/scripts/location_coverage.py --db path\\to\\realtyvoice.db

Opens the SQLite file in read-only mode (never writes) and reports:

* every distinct location label with its row count and share;
* city-level labels vs locality labels (labels containing a comma);
* how concentrated the data is (top-N share);
* labels that one search term would also pull in (substring collisions);
* whether every label is found by its own name and by its spoken form
  (hyphens replaced by spaces) using the production matcher.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.services.location_normalizer import resolve_locations, squash  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=str(ROOT / "data" / "realtyvoice.db"))
    args = parser.parse_args()

    db_path = pathlib.Path(args.db)
    if not db_path.is_file():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT location, COUNT(*) FROM properties "
            "WHERE location IS NOT NULL AND TRIM(location) <> '' "
            "GROUP BY location ORDER BY COUNT(*) DESC, location"
        ).fetchall()
        null_rows = conn.execute(
            "SELECT COUNT(*) FROM properties "
            "WHERE location IS NULL OR TRIM(location) = ''"
        ).fetchone()[0]
    finally:
        conn.close()

    labels = [r[0] for r in rows]
    total = sum(r[1] for r in rows)
    print(f"Distinct labels: {len(labels)}   rows with a location: {total}   "
          f"rows without: {null_rows}\n")

    print(f"{'label':32}{'rows':>9}{'share':>8}  kind")
    for label, n in rows:
        kind = "locality" if "," in label else "city"
        print(f"{label:32}{n:9d}{n / total * 100:7.2f}%  {kind}")

    print("\nConcentration:")
    acc = 0
    for i, (_label, n) in enumerate(rows, 1):
        acc += n
        if i in (1, 3, 5, 8, 10, 20, 30):
            print(f"  top {i:2d} labels = {acc / total * 100:5.1f}% of rows")

    print("\nSubstring collisions (searching the shorter name must NOT return the longer):")
    found = False
    for a, b in itertools.permutations(labels, 2):
        if squash(a) in squash(b) and squash(a) != squash(b):
            print(f"  {a!r} is contained in {b!r}")
            found = True
    if not found:
        print("  none")

    print("\nSelf-match check with the production matcher:")
    problems = 0
    for label in labels:
        spoken = label.replace("-", " ")
        for query in (label, spoken, spoken.upper(), spoken.replace(" ", "")):
            if label not in resolve_locations(query, labels):
                print(f"  NOT FOUND: query {query!r} does not return {label!r}")
                problems += 1
    print("  all labels found by their own name and spoken forms"
          if not problems else f"  {problems} problem(s) above")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
