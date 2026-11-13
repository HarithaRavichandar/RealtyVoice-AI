"""Audit script for the raw house_prices CSV.

- Parses the `Amount(in rupees)` column handling Indian units (Lac/Lakh, Cr/Crore).
- Reports parsing success/failure statistics.
- Provides price distribution metrics and top‑priced listings.
- Analyses the separate `Price (in rupees)` column as a distinct numeric field.
- Summarises categorical attributes (Status, Transaction, Furnishing, Ownership).
- Extracts BHK configuration from the `Title` column.
- Lists locations with the most listings.

The script prints a JSON report to stdout and writes the same JSON to
`data/processed/raw_data_audit.json`.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def resolve_csv_path() -> Path:
    """Return absolute path to the raw CSV file.
    The script lives in <project>/backend/scripts/, so two parents up is the
    project root.
    """
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "house_prices.csv"


def resolve_output_path() -> Path:
    """Path for the JSON audit report (data/processed/)."""
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "raw_data_audit.json"


def parse_indian_price(value: str):
    """Convert strings like "42 Lac", "1.40 Cr" to a numeric INR value.
    Returns None for empty / unparsable values.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Normalise spaces and commas
    s = s.replace(",", "").replace(" ", "")
    # Patterns
    lac_match = re.fullmatch(r"([0-9]*\.?[0-9]+)(Lac|Lakh)", s, re.IGNORECASE)
    cr_match = re.fullmatch(r"([0-9]*\.?[0-9]+)(Cr|Crore)", s, re.IGNORECASE)
    num_match = re.fullmatch(r"[0-9]*\.?[0-9]+", s)
    try:
        if lac_match:
            return float(lac_match.group(1)) * 1e5
        if cr_match:
            return float(cr_match.group(1)) * 1e7
        if num_match:
            return float(s)
    except Exception:
        return None
    return None


def parse_price_per_sqft(value: str):
    """Parse the `Price (in rupees)` column which appears to be a numeric
    per‑square‑foot value. Returns a float or None.
    """
    if pd.isna(value):
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    csv_path = resolve_csv_path()
    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load CSV – read everything as strings to avoid dtype inference issues
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # -------------------------------------------------------------------
    # 1. Analyze Amount(in rupees)
    # -------------------------------------------------------------------
    amount_raw = df["Amount(in rupees)"].copy()
    amount_parsed = amount_raw.apply(parse_indian_price)
    df["_parsed_amount"] = amount_parsed

    total_rows = len(df)
    null_count = amount_raw.isna().sum() + (amount_raw == "").sum()
    success_mask = amount_parsed.notna()
    success_count = success_mask.sum()
    failure_count = total_rows - success_count - null_count
    success_pct = (success_count / total_rows) * 100 if total_rows else 0

    distinct_formats = sorted(set(amount_raw.dropna().unique()))[:20]
    # Representative failures (first 5 distinct values that failed to parse)
    failed_vals = amount_raw[~success_mask & amount_raw.notna()].dropna().unique()
    failed_sample = list(failed_vals[:5])

    # Distribution metrics (on successful parses only)
    parsed_series = amount_parsed.dropna().astype(float)
    price_stats = {}
    if not parsed_series.empty:
        price_stats = {
            "min": parsed_series.min(),
            "max": parsed_series.max(),
            "mean": parsed_series.mean(),
            "median": parsed_series.median(),
            "p25": parsed_series.quantile(0.25),
            "p75": parsed_series.quantile(0.75),
            "p90": parsed_series.quantile(0.90),
            "p95": parsed_series.quantile(0.95),
            "p99": parsed_series.quantile(0.99),
        }

    # Top 20 priced records (include Index, location, Title, original, parsed)
    top20 = (
        df.loc[parsed_series.index]
        .sort_values(by="_parsed_amount", ascending=False)
        .head(20)
        .loc[:, ["Index", "location", "Title", "Amount(in rupees)", "_parsed_amount"]]
        .to_dict(orient="records")
    )

    # -------------------------------------------------------------------
    # 2. Analyze Price (in rupees) – treat as separate numeric field
    # -------------------------------------------------------------------
    price_sqft_raw = df.get("Price (in rupees)")
    price_sqft_parsed = None
    sqft_success_pct = None
    sqft_stats = {}
    if price_sqft_raw is not None:
        price_sqft_parsed = price_sqft_raw.apply(parse_price_per_sqft)
        sqft_success = price_sqft_parsed.notna().sum()
        sqft_failure = total_rows - sqft_success
        sqft_success_pct = (sqft_success / total_rows) * 100 if total_rows else 0
        sqft_series = price_sqft_parsed.dropna().astype(float)
        if not sqft_series.empty:
            sqft_stats = {
                "min": sqft_series.min(),
                "max": sqft_series.max(),
                "mean": sqft_series.mean(),
                "median": sqft_series.median(),
                "p25": sqft_series.quantile(0.25),
                "p75": sqft_series.quantile(0.75),
                "p90": sqft_series.quantile(0.90),
                "p95": sqft_series.quantile(0.95),
                "p99": sqft_series.quantile(0.99),
            }

    # -------------------------------------------------------------------
    # 3. Categorical attribute summaries
    # -------------------------------------------------------------------
    categorical_summary = {}
    for col in ["Status", "Transaction", "Furnishing", "Ownership"]:
        if col in df.columns:
            counts = (
                df[col]
                .fillna("<missing>")
                .value_counts()
                .head(20)
                .to_dict()
            )
            categorical_summary[col] = counts

    # BHK configuration extraction from Title
    bhk_counts = {}
    if "Title" in df.columns:
        for title in df["Title"].dropna():
            m = re.search(r"(\d+)\s*BHK", title, re.IGNORECASE)
            if m:
                bhk = int(m.group(1))
                bhk_counts[bhk] = bhk_counts.get(bhk, 0) + 1
    # Property types – very loosely inferred from Title (e.g., "Flat", "Apartment", "Villa")
    property_type_counts = {}
    if "Title" in df.columns:
        for title in df["Title"].dropna():
            lowered = title.lower()
            for pt in ["flat", "apartment", "villa", "house", "bungalow", "row house"]:
                if pt in lowered:
                    property_type_counts[pt] = property_type_counts.get(pt, 0) + 1
                    break

    # Locations with highest listing counts
    location_counts = {}
    if "location" in df.columns:
        location_counts = (
            df["location"].fillna("<missing>").value_counts().head(20).to_dict()
        )

    # -------------------------------------------------------------------
    # Assemble report
    # -------------------------------------------------------------------
    report = {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "amount_parsing": {
            "total": total_rows,
            "null_or_empty": int(null_count),
            "parsed_success": int(success_count),
            "parsed_failure": int(failure_count),
            "success_percent": round(success_pct, 2),
            "distinct_raw_values_sample": distinct_formats,
            "failed_samples": failed_sample,
        },
        "amount_distribution": price_stats,
        "top_20_price_records": top20,
        "price_per_sqft_parsing": {
            "total": total_rows,
            "parsed_success": int(sqft_success) if price_sqft_raw is not None else None,
            "parsed_failure": int(sqft_failure) if price_sqft_raw is not None else None,
            "success_percent": round(sqft_success_pct, 2) if sqft_success_pct is not None else None,
        },
        "price_per_sqft_distribution": sqft_stats,
        "categorical_counts": categorical_summary,
        "bhk_distribution": bhk_counts,
        "property_type_distribution": property_type_counts,
        "top_locations_by_listing_count": location_counts,
    }

    # Write report to file
    out_path = resolve_output_path()
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Print report (pretty) to stdout – limited size for readability
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()

