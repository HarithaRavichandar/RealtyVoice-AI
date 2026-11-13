# -*- coding: utf-8 -*-
"""
Integration tests for the PropertyImporter (Phase 2E).

The tests run against a **temporary SQLite database** created for each test
run.  No production database or files are touched.

Key checks performed:
* Correct transformation of public CSV rows.
* Correct handling of synthetic JSON records.
* Idempotent upserts – duplicate detection, timestamp preservation.
* Foreign‑key enforcement on the temporary SQLite engine.
"""
from decimal import Decimal
import csv
import json
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from backend.models.property import Property
from backend.services.property_importer import PropertyImporter
from backend.database import Base

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def temp_db_path(tmp_path: Path) -> Path:
    """Path to a temporary SQLite file used for the test run."""
    return tmp_path / "temp_realtyvoice.db"

@pytest.fixture(scope="function")
def engine(temp_db_path: Path):
    """Create an SQLite engine with foreign‑key enforcement enabled."""
    eng = create_engine(
        f"sqlite:///{temp_db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    # Enable foreign‑key constraints for every new connection.
    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Verify the pragma works.
    with eng.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1

    # Create tables from ORM metadata.
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()

@pytest.fixture(scope="function")
def temp_session_factory(engine):
    """Factory returning new Session objects bound to the temporary engine."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# ---------------------------------------------------------------------------
# Helper functions for fixture files
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file using the exact public‑CSV header.

    Uses ``csv.writer`` to guarantee proper quoting/escaping.
    """
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
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, "") for col in header])


def write_json(path: Path, data: list[dict]) -> None:
    """Write a list of dictionaries to a JSON file (UTF‑8)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_importer_integration(tmp_path: Path, temp_session_factory, engine):
    """End‑to‑end verification of the importer against a clean temporary DB.

    The test covers:
    * public CSV import mapping,
    * synthetic JSON import mapping,
    * idempotent upserts with timestamp checks,
    * and foreign‑key enforcement.
    """

    # -------------------------------------------------------------------
    # 1️⃣  Prepare CSV fixture (5 rows covering required cases)
    # -------------------------------------------------------------------
    csv_path = tmp_path / "house_prices.csv"
    csv_rows = [
        # Normal flat, price in Lakhs, with Society name.
        {
            "Index": "1",
            "Title": "2 BHK Flat in Mumbai",
            "Description": "Spacious flat with garden",
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
        },
        # Apartment, price in Crores.
        {
            "Index": "2",
            "Title": "3 BHK Apartment in Bangalore",
            "Description": "",
            "Amount(in rupees)": "1.40 Cr",
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
        },
        # Call for price – unparsable amount.
        {
            "Index": "3",
            "Title": "1 BHK Row House",
            "Description": "",
            "Amount(in rupees)": "Call for Price",
            "Price (in rupees)": "",
            "location": "Chennai",
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
        # Missing optional fields.
        {
            "Index": "4",
            "Title": "Studio Flat",
            "Description": "",
            "Amount(in rupees)": "16.5 Lac",
            "Price (in rupees)": "",
            "location": "Delhi",
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
        # Different BHK configuration.
        {
            "Index": "5",
            "Title": "4‑BHK Villa in Hyderabad",
            "Description": "",
            "Amount(in rupees)": "2.5 Cr",
            "Price (in rupees)": "",
            "location": "Hyderabad",
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
    ]
    write_csv(csv_path, csv_rows)

    # -------------------------------------------------------------------
    # 2️⃣  Prepare synthetic JSON fixture (2 records)
    # -------------------------------------------------------------------
    json_path = tmp_path / "properties.json"
    synthetic_records = [
        {
            "property_id": "syn-1",
            "project_name": "Green Heights",
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
        },
        {
            "property_id": "syn-2",
            "project_name": "Lakeview",
            "location": "Mumbai",
            "property_type": "villa",
            "configuration": "4 BHK",
            "price_min": 2.5,
            "price_max": 3.0,
            "listing_price": 2.75,
            "possession_date": "2025-06-01T00:00:00",
            "amenities": ["Gym", "Clubhouse"],
            "nearby_locations": ["Railway Station"],
            "availability_status": "under construction",
            "description": "Luxury villa",
        },
    ]
    write_json(json_path, synthetic_records)

    # -------------------------------------------------------------------
    # 3️⃣  Instantiate the importer with the temporary session factory
    # -------------------------------------------------------------------
    importer = PropertyImporter(session_factory=temp_session_factory, dry_run=False)

    # -------------------------------------------------------------------
    # 4️⃣  Run public CSV import (apply mode)
    # -------------------------------------------------------------------
    csv_stats = importer.import_public_csv(csv_path)
    assert csv_stats["total_records"] == 5
    assert csv_stats["valid_records"] == 5
    assert csv_stats["rejected_records"] == 0
    assert csv_stats["duplicate_ids"] == 0
    # Four rows have parsable prices, one is "Call for Price"
    assert csv_stats["price_parsed"] == 4
    assert csv_stats["price_unparsed"] == 1

    # Verify a couple of rows directly from the DB.
    session = temp_session_factory()
    try:
        p1 = session.query(Property).filter_by(source_record_id="1").one()
        assert p1.source_name == "kaggle_house_price_india"
        assert p1.source_url == "https://www.kaggle.com/datasets/juhibhojani/house-price"
        # 42 Lac = 4_200_000
        assert p1.listing_price == 4_200_000
        assert p1.price_min is None and p1.price_max is None
        assert p1.availability_status is None
        assert p1.possession_date is None
        assert p1.project_name == "Sunrise Apartments"
        assert p1.raw_source_metadata["Amount(in rupees)"] == "42 Lac"

        p3 = session.query(Property).filter_by(source_record_id="3").one()
        # Unparsable price should leave listing_price NULL.
        assert p3.listing_price is None

        p5 = session.query(Property).filter_by(source_record_id="5").one()
        assert p5.configuration == "4 BHK"
    finally:
        session.close()

    # -------------------------------------------------------------------
    # 5️⃣  Run synthetic JSON import (apply mode)
    # -------------------------------------------------------------------
    json_stats = importer.import_synthetic_json(json_path)
    assert json_stats["total_records"] == 2
    assert json_stats["valid_records"] == 2
    assert json_stats["duplicate_ids"] == 0
    assert json_stats["price_parsed"] == 4  # min & max for each record

    session = temp_session_factory()
    try:
        syn1 = session.query(Property).filter_by(source_record_id="syn-1").one()
        assert syn1.source_name == "synthetic"
        assert syn1.price_min == Decimal("1.4")
        assert syn1.price_max == Decimal("1.6")
        assert syn1.possession_date == datetime.fromisoformat("2024-01-01T00:00:00")
        assert syn1.availability_status == "available"
        assert syn1.raw_source_metadata["amenities"] == ["Gym", "Pool"]

        # Total rows should be 7 (5 CSV + 2 JSON)
        total_rows = session.query(Property).count()
        assert total_rows == 7
    finally:
        session.close()

    # -------------------------------------------------------------------
    # 6️⃣  Idempotency – re‑run both imports
    # -------------------------------------------------------------------
    csv_stats2 = importer.import_public_csv(csv_path)
    json_stats2 = importer.import_synthetic_json(json_path)
    assert csv_stats2["duplicate_ids"] == 5
    assert json_stats2["duplicate_ids"] == 2

    session = temp_session_factory()
    try:
        assert session.query(Property).count() == 7
    finally:
        session.close()

    # -------------------------------------------------------------------
    # 7️⃣  Verify updated timestamps after modifying a synthetic record
    # -------------------------------------------------------------------
    # Capture original timestamps.
    session = temp_session_factory()
    try:
        original = session.query(Property).filter_by(source_record_id="syn-1").one()
        original_created = original.created_at
        original_updated = original.updated_at
    finally:
        session.close()

    # Modify fixture description.
    synthetic_records[0]["description"] = "Updated description"
    write_json(json_path, synthetic_records)

    # Re‑import synthetic JSON.
    json_stats3 = importer.import_synthetic_json(json_path)
    assert json_stats3["duplicate_ids"] == 2

    # Verify changes with a fresh session.
    session = temp_session_factory()
    try:
        updated = session.query(Property).filter_by(source_record_id="syn-1").one()
        assert updated.description == "Updated description"
        assert updated.created_at == original_created
        assert updated.updated_at > original_updated
    finally:
        session.close()

    # -------------------------------------------------------------------
    # 8️⃣  Reporting (printed for the test output)
    # -------------------------------------------------------------------
    print("\n--- Integration Test Summary ---")
    print(f"Temporary DB URL: {engine.url}")
    print(f"CSV – imported: {csv_stats['valid_records']}, rejected: {csv_stats['rejected_records']}")
    print(f"JSON – imported: {json_stats['valid_records']}, rejected: {json_stats['rejected_records']}")
    print(f"Total rows after first run: {total_rows}")
    print(f"Duplicate IDs on re‑run – CSV: {csv_stats2['duplicate_ids']}, JSON: {json_stats2['duplicate_ids']}")
    print("--- End of Summary ---\n")

