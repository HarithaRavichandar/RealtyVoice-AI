#!/usr/bin/env python
"""Command‑line interface for importing property data.

The implementation delegates all heavy‑lifting to the existing
`backend.services.property_importer.PropertyImporter` class.  The CLI
exposes the required arguments, respects dry‑run default behaviour and
prints a concise summary after each source is processed.
"""

import argparse
import sys
from pathlib import Path

# Project configuration – reuse the paths defined in backend.config
from backend.config import DATA_DIR
from backend.services.property_importer import PropertyImporter


def _print_summary(source_name: str, stats: dict) -> None:
    """Print a human‑readable import summary for a given source.

    The *stats* dictionary is the one populated by ``PropertyImporter``
    after a successful import.  Only the fields required by the task
    description are displayed.
    """
    print(f"\n=== {source_name.upper()} IMPORT SUMMARY ===")
    print(f"Mode                : {stats.get('mode', 'dry-run')}" )
    print(f"Total records       : {stats.get('total_records', 0)}")
    print(f"Valid records       : {stats.get('valid_records', 0)}")
    print(f"Rejected records    : {stats.get('rejected_records', 0)}")
    print(f"Duplicate IDs       : {stats.get('duplicate_ids', 0)}")
    print(f"Price parsed        : {stats.get('price_parsed', 0)}")
    print(f"Price unparsed      : {stats.get('price_unparsed', 0)}")
    print(f"Batches processed   : {stats.get('batches', 0)}")
    reasons = stats.get('rejection_reasons', [])
    if reasons:
        print("Rejection reasons   :")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("Rejection reasons   : None")
    print("" )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import property listings from the public CSV and/or the synthetic JSON dataset.",
        prog="import_properties",
    )
    parser.add_argument(
        "--source",
        choices=["public", "synthetic", "all"],
        default="public",
        help="Select which data source to import (default: public).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write records to the SQLite database (default is dry‑run).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of records processed per batch (default: 5000).",
    )

    args = parser.parse_args(argv)
    if args.batch_size < 1:
        print("Error: --batch-size must be a positive integer (>=1)", file=sys.stderr)
        return 2

    # Resolve source file paths relative to the project data directory
    public_csv = DATA_DIR / "raw" / "house_prices.csv"
    synthetic_json = DATA_DIR / "synthetic" / "properties.json"

    # Validate existence of requested source files before proceeding
    if args.source in ("public", "all") and not public_csv.is_file():
        print(f"Error: public CSV not found at {public_csv}", file=sys.stderr)
        return 1
    if args.source in ("synthetic", "all") and not synthetic_json.is_file():
        print(f"Error: synthetic JSON not found at {synthetic_json}", file=sys.stderr)
        return 1

    # Initialise the importer – dry_run is the inverse of --apply
    importer = PropertyImporter(
        batch_size=args.batch_size,
        dry_run=not args.apply,
    )

    exit_code = 0
    try:
        if args.source in ("public", "all"):
            stats = importer.import_public_csv(public_csv)
            _print_summary("public", stats)
            if stats.get("total_records", 0) != stats.get("valid_records", 0) + stats.get("rejected_records", 0):
                print("Invariant violation: total != valid + rejected", file=sys.stderr)
                exit_code = 1
        if args.source in ("synthetic", "all"):
            stats = importer.import_synthetic_json(synthetic_json)
            _print_summary("synthetic", stats)
            if stats.get("total_records", 0) != stats.get("valid_records", 0) + stats.get("rejected_records", 0):
                print("Invariant violation: total != valid + rejected", file=sys.stderr)
                exit_code = 1
    except Exception as exc:  # pragma: no cover – unexpected failures
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

