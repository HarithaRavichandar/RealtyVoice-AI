# -*- coding: utf-8 -*-
"""Automated tests for the PropertyImporter.

The tests use an isolated temporary SQLite database and verify the
behaviour of the importer without touching the production database or
source files.
"""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.property import Property
from backend.services.property_importer import PropertyImporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def temp_engine() -> Any:
    """Create an in‑memory SQLite engine for the duration of a test."""
    engine = create_engine("sqlite:///:memory:", future=True)
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def temp_session_factory(temp_engine) -> Callable[[], Any]:
    """SQLAlchemy session factory bound to the temporary engine.

    The tables are created from the ORM metadata before the first session is
    returned.
    """
    from backend.database import Base

    Base.metadata.create_all(temp_engine)
    return sessionmaker(bind=temp_engine, autoflush=False, autocommit=False, future=True)

@pytest.fixture(scope="function")
def sample_public_row() -> dict:
    """A minimal CSV row that matches the expected header exactly."""
    return {
        "Index": "1",
        "Title": "2 BHK Flat in Mumbai",
        "Description": "Spacious flat",
        "Amount(in rupees)": "42 Lac",
        "Price (in rupees)": "",
        "location": "Mumbai",
        "Carpet Area": "",
        "Status": "",
        "Floor": "",
        "Transaction": "",
        "Furnishing": "",
        "facing": "",
        "overlooking": "",
        "Society": "Sunrise Apartments",
        "Bathroom": "",
        "Balcony": "",
        "Car Parking": "",
        "Ownership": "",
        "Super Area": "",
        "Dimensions": "",
        "Plot Area": "",
    }

@pytest.fixture(scope="function")
def sample_synthetic_record() -> dict:
    """A minimal synthetic JSON record."""
    return {
        "property_id": "syn-1",
        "location": "Delhi",
        "property_type": "apartment",
        "configuration": "3 BHK",
        "price_min": 1.4,
        "price_max": 1.6,
        "listing_price": 1.5,
        "possession_date": "2024-01-01T00:00:00",
        "amenities": ["Gym", "Pool"],
        "nearby_locations": ["Metro"],
        "availability_status": "available",
        "description": "A nice apartment",
        "project_name": "Green Heights",
    }

# ---------------------------------------------------------------------------
# Helper utilities for creating temporary source files
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    """Write *rows* to *path* using the header expected by the importer."""
    header = [
        "Index",
        "Title",
        "Description",
        "Amount(in rupees)",
        "Price (in rupees)",
        "location",
        "Carpet Area",
        "Status",
        "Floor",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Society",
        "Bathroom",
        "Balcony",
        "Car Parking",
        "Ownership",
        "Super Area",
        "Dimensions",
        "Plot Area",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row.get(col, "") for col in header) + "\n")

def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------
# A. Transformation tests
# ---------------------------------------------------------------------------

def test_parse_price_variants():
    parse = PropertyImporter._parse_price
    assert parse("42 Lac") == (Decimal(4200000), True)
    assert parse("1.40 Cr") == (Decimal(14000000), True)
    assert parse("16.5 Lac") == (Decimal(1650000), True)
    assert parse("₹42 Lac") == (Decimal(4200000), True)
    assert parse("Call for Price") == (None, False)

def test_extract_bhk_and_type_classification():
    extractor = PropertyImporter._extract_bhk
    classifier = PropertyImporter._classify_property_type

    # Existing assertions (retain)
    assert extractor("2 BHK Row House in Chennai") == "2 BHK"
    assert extractor("3-BHK Apartment in Bangalore") == "3 BHK"
    assert extractor("Studio") is None

    assert classifier("2 BHK Row House in Chennai") == "row house"
    assert classifier("3 BHK Apartment in Bangalore") == "apartment"
    assert classifier("Residential Flat for Sale") == "flat"

    # New edge‑case formats (including various dash characters)
    assert extractor("4 BHK Villa in Hyderabad") == "4 BHK"
    assert extractor("4-BHK Villa in Hyderabad") == "4 BHK"
    assert extractor("4\u2011BHK Villa in Hyderabad") == "4 BHK"  # non‑breaking hyphen
    assert extractor("4\u2013BHK Villa in Hyderabad") == "4 BHK"  # en dash
    assert extractor("4\u2014BHK Villa in Hyderabad") == "4 BHK"  # em dash
    assert extractor("4BHK Villa in Hyderabad") == "4 BHK"

def test_process_public_row_mapping(sample_public_row):
    importer = PropertyImporter(session_factory=lambda: None, dry_run=True)
    prop, reason = importer._process_public_row(sample_public_row, set())
    assert reason is None
    assert isinstance(prop, Property)
    # Price handling
    assert prop.listing_price == Decimal(4200000)
    assert prop.price_min is None and prop.price_max is None
    # Historical fields must stay NULL
    assert prop.availability_status is None
    assert prop.possession_date is None
    # Preserve raw metadata as dict (null for empty strings)
    for key, value in sample_public_row.items():
        expected = value if value != "" else None
        assert prop.raw_source_metadata.get(key) == expected
    # Project name from Society when present
    assert prop.project_name == "Sunrise Apartments"
    # Description fallback
    assert prop.description == "Spacious flat"

def test_process_synthetic_record_mapping(sample_synthetic_record):
    importer = PropertyImporter(session_factory=lambda: None, dry_run=True)
    prop, reason = importer._process_synthetic_record(sample_synthetic_record, set())
    assert reason is None
    assert isinstance(prop, Property)
    # Price fields are Decimal values
    assert prop.price_min == Decimal(str(sample_synthetic_record["price_min"]))
    assert prop.price_max == Decimal(str(sample_synthetic_record["price_max"]))
    # Historical fields are kept as provided
    assert prop.availability_status == sample_synthetic_record["availability_status"]
    # Raw metadata preservation
    for k, v in sample_synthetic_record.items():
        expected = v if v != "" else None
        assert prop.raw_source_metadata.get(k) == expected

# ---------------------------------------------------------------------------
# B. Dry‑run isolation
# ---------------------------------------------------------------------------

def test_dry_run_no_db_access(tmp_path, temp_session_factory):
    # Create tiny source files
    csv_path = tmp_path / "house_prices.csv"
    write_csv(csv_path, [
        {
            "Index": "1",
            "Title": "1 BHK Flat",
            "Description": "",
            "Amount(in rupees)": "10 Lac",
            "Price (in rupees)": "",
            "location": "Kolkata",
            "Carpet Area": "",
            "Status": "",
            "Floor": "",
            "Transaction": "",
            "Furnishing": "",
            "facing": "",
            "overlooking": "",
            "Society": "",
            "Bathroom": "",
            "Balcony": "",
            "Car Parking": "",
            "Ownership": "",
            "Super Area": "",
            "Dimensions": "",
            "Plot Area": "",
        }
    ])
    json_path = tmp_path / "properties.json"
    write_json(json_path, [
        {
            "property_id": "syn-1",
            "location": "Delhi",
            "property_type": "apartment",
            "configuration": "2 BHK",
            "price_min": 0.5,
            "price_max": 0.7,
            "listing_price": 0.6,
        }
    ])

    # Session factory that raises if called – should never be used in dry‑run
    def exploding_factory():
        raise RuntimeError("DB session should not be created in dry‑run")

    importer = PropertyImporter(session_factory=exploding_factory, dry_run=True)
    # Both sources must complete without touching the DB
    stats_pub = importer.import_public_csv(csv_path)
    stats_syn = importer.import_synthetic_json(json_path)
    # Invariant check – the importer already asserts this, but we double‑check
    assert stats_pub["total_records"] == stats_pub["valid_records"] + stats_pub["rejected_records"]
    assert stats_syn["total_records"] == stats_syn["valid_records"] + stats_syn["rejected_records"]

# ---------------------------------------------------------------------------
# C. Apply‑mode tests with a temporary SQLite DB
# ---------------------------------------------------------------------------

def test_apply_import_insert_and_idempotency(tmp_path, temp_session_factory):
    csv_path = tmp_path / "house_prices.csv"
    write_csv(csv_path, [
        {
            "Index": "10",
            "Title": "3 BHK Apartment",
            "Description": "Nice view",
            "Amount(in rupees)": "1.2 Cr",
            "Price (in rupees)": "",
            "location": "Bangalore",
            "Carpet Area": "",
            "Status": "",
            "Floor": "",
            "Transaction": "",
            "Furnishing": "",
            "facing": "",
            "overlooking": "",
            "Society": "",
            "Bathroom": "",
            "Balcony": "",
            "Car Parking": "",
            "Ownership": "",
            "Super Area": "",
            "Dimensions": "",
            "Plot Area": "",
        }
    ])
    importer = PropertyImporter(session_factory=temp_session_factory, dry_run=False)
    stats = importer.import_public_csv(csv_path)
    assert stats["valid_records"] == 1
    # Verify the record exists in the DB
    session = temp_session_factory()
    prop = session.get(Property, 1)
    assert prop is not None
    assert prop.source_record_id == "10"
    assert prop.listing_price == Decimal(12000000)
    # Re‑run the same import – should not create a duplicate
    stats2 = importer.import_public_csv(csv_path)
    assert stats2["duplicate_ids"] == 1
    # The number of rows in the table stays at 1
    count = session.query(Property).count()
    assert count == 1

def test_apply_import_update_preserves_created_at(tmp_path, temp_session_factory):
    json_path = tmp_path / "properties.json"
    record = {
        "property_id": "syn-2",
        "location": "Chennai",
        "property_type": "flat",
        "configuration": "2 BHK",
        "price_min": 0.8,
        "price_max": 1.0,
        "listing_price": 0.9,
        "description": "Original",
    }
    write_json(json_path, [record])
    importer = PropertyImporter(session_factory=temp_session_factory, dry_run=False)
    stats = importer.import_synthetic_json(json_path)
    assert stats["valid_records"] == 1
    # First query to capture created_at
    session = temp_session_factory()
    prop = session.query(Property).filter_by(source_record_id="syn-2").one()
    original_created = prop.created_at
    session.close()
    # Modify the fixture – change description
    record["description"] = "Updated"
    write_json(json_path, [record])
    # Run import again – should update not duplicate
    stats2 = importer.import_synthetic_json(json_path)
    assert stats2["duplicate_ids"] == 1
    # Use a fresh session to see updated values
    session2 = temp_session_factory()
    prop2 = session2.query(Property).filter_by(source_record_id="syn-2").one()
    assert prop2.description == "Updated"
    # created_at stays the same, updated_at changes
    assert prop2.created_at == original_created
    assert prop2.updated_at > prop2.created_at
    session2.close()

# ---------------------------------------------------------------------------
# D. Transaction rollback
# ---------------------------------------------------------------------------

def test_transaction_rollback_on_error(tmp_path, temp_session_factory, monkeypatch):
    csv_path = tmp_path / "house_prices.csv"
    write_csv(csv_path, [
        {
            "Index": "20",
            "Title": "1 BHK Flat",
            "Description": "",
            "Amount(in rupees)": "5 Lac",
            "Price (in rupees)": "",
            "location": "Pune",
            "Carpet Area": "",
            "Status": "",
            "Floor": "",
            "Transaction": "",
            "Furnishing": "",
            "facing": "",
            "overlooking": "",
            "Society": "",
            "Bathroom": "",
            "Balcony": "",
            "Car Parking": "",
            "Ownership": "",
            "Super Area": "",
            "Dimensions": "",
            "Plot Area": "",
        },
        {
            "Index": "21",
            "Title": "Invalid Price",
            "Description": "",
            "Amount(in rupees)": "not a number",
            "Price (in rupees)": "",
            "location": "Pune",
            "Carpet Area": "",
            "Status": "",
            "Floor": "",
            "Transaction": "",
            "Furnishing": "",
            "facing": "",
            "overlooking": "",
            "Society": "",
            "Bathroom": "",
            "Balcony": "",
            "Car Parking": "",
            "Ownership": "",
            "Super Area": "",
            "Dimensions": "",
            "Plot Area": "",
        },
    ])

    # Monkey‑patch the batch upsert to raise after processing the first batch
    def exploding_upsert(self, objs, session):  # noqa: N802 – method signature matches
        raise RuntimeError("Simulated failure inside batch upsert")

    monkeypatch.setattr(PropertyImporter, "_upsert_batch", exploding_upsert)
    importer = PropertyImporter(session_factory=temp_session_factory, dry_run=False, batch_size=1)
    with pytest.raises(RuntimeError, match="Simulated failure"):
        importer.import_public_csv(csv_path)
    # After the failure the table must be empty – the transaction was rolled back
    session = temp_session_factory()
    count = session.query(Property).count()
    assert count == 0
    session.close()

