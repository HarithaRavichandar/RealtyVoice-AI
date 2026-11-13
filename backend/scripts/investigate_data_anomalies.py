"""Investigate data anomalies in the raw house_prices CSV.

This script performs a focused second‑audit:
1. Total‑price (`Amount(in rupees)`) parsing failures – list distinct failed values,
   counts, and a few example rows.
2. `Price (in rupees)` (price‑per‑sqft) anomalies – missing/invalid/zero/negative counts,
   distribution metrics, and top values.
3. Total‑price outliers – 30 highest parsed prices and counts above key thresholds.
4. Record identity – uniqueness of `Index` and duplicate combinations of location,
   title, and original price.

The script **does not modify** any source data. It writes a JSON report to:
`data/processed/data_anomaly_report.json` and prints a concise summary.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Helper for safe JSON serialization
def default_converter(obj):
    """Convert numpy / pandas types to native Python types for JSON.
    Handles:
      - numpy integer, floating, bool
      - numpy ndarray
      - pandas Timestamp
      - pandas NA / NaT values (converted to None)
    Falls back to str for any other object.
    """
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # pandas / numpy missing values
    if pd.isna(obj):
        return None
    # Fallback – let json raise TypeError for unexpected types
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Helpers – identical to those used in audit_price_data.py
# ---------------------------------------------------------------------------

def resolve_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "house_prices.csv"


def resolve_output_path() -> Path:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "data_anomaly_report.json"


def parse_indian_price(value: str):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "").replace(" ", "")
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
def main():
    csv_path = resolve_csv_path()
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load entire CSV as strings to avoid dtype inference issues
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    total_rows = len(df)

    # ---------------------------------------------------
    # 1. Total‑price parsing failures
    # ---------------------------------------------------
    amount_raw = df["Amount(in rupees)"].copy()
    amount_parsed = amount_raw.apply(parse_indian_price)
    df["_parsed_amount"] = amount_parsed

    # Failure mask: original not null/empty and parse result is None
    failure_mask = amount_raw.notna() & (amount_raw.str.strip() != "") & amount_parsed.isna()
    failures = df[failure_mask]
    failure_counts = failures["Amount(in rupees)"].value_counts().to_dict()
    distinct_failure_examples = []
    for _, row in failures.head(5).iterrows():
        distinct_failure_examples.append({
            "Index": row.get("Index"),
            "location": row.get("location"),
            "Title": row.get("Title"),
            "Amount(in rupees)": row.get("Amount(in rupees)"),
        })

    # ---------------------------------------------------
    # 2. Price‑per‑sqft anomalies
    # ---------------------------------------------------
    price_raw = df.get("Price (in rupees)")
    price_parsed = None
    missing_cnt = 0
    invalid_cnt = 0
    zero_cnt = 0
    negative_cnt = 0
    price_stats = {}
    top20_price_sqft = []
    if price_raw is not None:
        price_parsed = price_raw.apply(parse_price_per_sqft)
        df["_price_psf"] = price_parsed
        missing_cnt = price_raw.isna().sum() + (price_raw.str.strip() == "").sum()
        # Invalid = original not missing but parsed is None
        invalid_mask = price_raw.notna() & (price_raw.str.strip() != "") & price_parsed.isna()
        invalid_cnt = invalid_mask.sum()
        # Valid numeric values
        valid_series = price_parsed.dropna().astype(float)
        zero_cnt = (valid_series == 0).sum()
        negative_cnt = (valid_series < 0).sum()
        if not valid_series.empty:
            price_stats = {
                "min": valid_series.min(),
                "max": valid_series.max(),
                "median": valid_series.median(),
                "p95": valid_series.quantile(0.95),
                "p99": valid_series.quantile(0.99),
            }
            top20_price_sqft = (
                df.loc[valid_series.index]
                .sort_values(by="_price_psf", ascending=False)
                .head(20)
                .loc[:, ["Index", "location", "Title", "Price (in rupees)"]]
                .to_dict(orient="records")
            )

    # ---------------------------------------------------
    # 3. Total‑price outliers
    # ---------------------------------------------------
    valid_amount_series = amount_parsed.dropna().astype(float)
    top30_total = (
        df.loc[valid_amount_series.index]
        .sort_values(by="_parsed_amount", ascending=False)
        .head(30)
        .loc[:, ["Index", "location", "Title", "Amount(in rupees)", "_parsed_amount"]]
        .to_dict(orient="records")
    )
    # Threshold counts (₹1 cr = 1e7, ₹5 cr = 5e7, etc.)
    thresholds = {
        "above_1_crore": (valid_amount_series >= 1e7).sum(),
        "above_5_crore": (valid_amount_series >= 5e7).sum(),
        "above_10_crore": (valid_amount_series >= 1e8).sum(),
        "above_25_crore": (valid_amount_series >= 2.5e8).sum(),
        "above_50_crore": (valid_amount_series >= 5e8).sum(),
    }

    # ---------------------------------------------------
    # 4. Record identity checks
    # ---------------------------------------------------
    index_unique = df["Index"].is_unique
    duplicate_index_count = df["Index"].duplicated().sum()
    # Duplicate combos of location + Title + original amount
    combo_cols = ["location", "Title", "Amount(in rupees)"]
    duplicate_combo_count = df.duplicated(subset=combo_cols).sum()

    # ---------------------------------------------------
    # Assemble report
    # ---------------------------------------------------
    report = {
        "row_count": total_rows,
        "total_price_parsing_failures": {
            "total_failures": int(failure_mask.sum()),
            "distinct_failed_values": failure_counts,
            "representative_examples": distinct_failure_examples,
        },
        "price_per_sqft_anomalies": {
            "missing": int(missing_cnt),
            "invalid": int(invalid_cnt),
            "zero": int(zero_cnt),
            "negative": int(negative_cnt),
            "distribution": price_stats,
            "top_20_values": top20_price_sqft,
        },
        "total_price_outliers": {
            "top_30_highest": top30_total,
            "threshold_counts": thresholds,
        },
        "record_identity": {
            "index_unique": bool(index_unique),
            "duplicate_index_count": int(duplicate_index_count),
            "duplicate_location_title_amount_count": int(duplicate_combo_count),
        },
    }

    out_path = resolve_output_path()
    # Use a safe JSON encoder that converts numpy types to native Python types
    out_path.write_text(json.dumps(report, indent=2, default=default_converter), encoding="utf-8")

    # Concise terminal summary (json with only high‑level sections)
    summary = {
        "rows": total_rows,
        "price_parse_failures": int(failure_mask.sum()),
        "price_psf_missing": int(missing_cnt),
        "price_psf_invalid": int(invalid_cnt),
        "price_psf_zero": int(zero_cnt),
        "price_psf_negative": int(negative_cnt),
        "top_price_outlier_count": len(top30_total),
        "duplicate_index": int(duplicate_index_count),
    }
    print(json.dumps(summary, indent=2, default=default_converter))

if __name__ == "__main__":
    main()

