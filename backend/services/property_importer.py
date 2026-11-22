"""Property importer implementation for RealtyVoice AI.

This module provides a `PropertyImporter` class that can import data from:
* The public Kaggle CSV dataset (historical property listings).
* The synthetic JSON dataset bundled with the project.

The implementation follows the strict specifications given in the latest
user request.  It deliberately avoids any database interaction when the
instance is in *dry‑run* mode, making it safe for inspection and unit‑test
execution.
"""

import csv
import json
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from ..models.property import Property
from ..database import SessionLocal


class PropertyImporter:
    """Unified importer for public CSV and synthetic JSON property data.

    By default the importer runs in *dry‑run* mode – no database queries or
    writes are performed.  When ``dry_run=False`` a single transaction is used
    for the entire source import.
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(self, session_factory: Any = SessionLocal, batch_size: int = 5_000, dry_run: bool = True) -> None:
        self.session_factory = session_factory
        self.batch_size = max(1, batch_size)
        self.dry_run = dry_run
        # Initialise stats with all keys used across the importer to avoid KeyError
        self.stats: Dict[str, Any] = {
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "duplicate_ids": 0,
            "price_parsed": 0,
            "price_unparsed": 0,
            "missing_configuration": 0,
            "missing_property_type": 0,
            "missing_location": 0,
            "rejection_reasons": [],
            "batches": 0,
            "mode": "dry-run" if dry_run else "apply",
        }

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_price(raw: str) -> Tuple[Decimal | None, bool]:
        """Parse Indian monetary strings into a ``Decimal``.

        Returns (value, True) for valid inputs, otherwise (None, False)."""
        if not raw or not isinstance(raw, str):
            return None, False
        # Strip whitespace and commas
        s = raw.strip().replace(",", "")
        # Remove rupee symbol and INR token
        s = s.replace("₹", "")
        s = re.sub(r"(?i)\bINR\b", "", s).strip()
        if not s:
            return None, False
        # Full‑match number with optional suffix
        m = re.fullmatch(r"(?i)^(?P<num>\d+(?:\.\d+)?)(?:\s*(?P<suffix>lac|lakh|cr|crore))?$", s)
        if not m:
            return None, False
        num = Decimal(m.group("num"))
        suffix = m.group("suffix")
        multiplier = Decimal(1)
        if suffix:
            if re.fullmatch(r"(?i)^(lac|lakh)$", suffix):
                multiplier = Decimal(100_000)
            elif re.fullmatch(r"(?i)^(cr|crore)$", suffix):
                multiplier = Decimal(10_000_000)
        try:
            return num * multiplier, True
        except InvalidOperation:
            return None, False

    @staticmethod
    def _extract_bhk(title: str) -> str | None:
        """Extract BHK configuration from a title string.

        Looks for patterns like ``2 BHK`` or ``3-BHK`` using word boundaries.
        Returns ``"2 BHK"`` style strings or ``None``.
        """
        if not title:
            return None
        m = re.search(r"\b(\d+)[-\s\u2010-\u2015]?BHK\b", title, re.IGNORECASE)
        if m:
            return f"{m.group(1)} BHK"
        return None

    @staticmethod
    def _classify_property_type(title: str) -> str | None:
        """Classify property type from a title using safe keyword matching.

        ``row house`` is prioritized over generic ``house``. Matching uses
        proper ``\b`` word‑boundary anchors.
        """
        if not title:
            return None
        lowered = title.lower()
        mapping = [
            ("row house", "row house"),
            ("apartment", "apartment"),
            ("flat", "flat"),
            ("villa", "villa"),
            ("house", "house"),
            ("plot", "plot"),
        ]
        for keyword, typ in mapping:
            if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                return typ
        return None

    # ---------------------------------------------------------------------
    # Core upsert logic (used only when not dry‑run)
    # ---------------------------------------------------------------------
    def _upsert_batch(self, objs: List[Property], session: Session) -> None:
        """Upsert a batch of ``Property`` objects.

        Existing rows are fetched via provenance keys.  Updates are performed only
        when any business field differs.
        """
        if not objs:
            return
        keys = [(obj.source_name, obj.source_record_id) for obj in objs]
        existing = {
            (row.source_name, row.source_record_id): row
            for row in session.execute(
                select(Property).where(tuple_(Property.source_name, Property.source_record_id).in_(keys))
            ).scalars()
        }
        to_add: List[Property] = []
        for obj in objs:
            key = (obj.source_name, obj.source_record_id)
            existing_row = existing.get(key)
            if existing_row is None:
                to_add.append(obj)
                continue
            # Existing row – count as duplicate for stats
            self.stats["duplicate_ids"] = self.stats.get("duplicate_ids", 0) + 1
            business_attrs = [
                "project_name",
                "location",
                "property_type",
                "configuration",
                "price_min",
                "price_max",
                "listing_price",
                "possession_date",
                "amenities",
                "nearby_locations",
                "availability_status",
                "description",
                "raw_source_metadata",
                "source_url",
                "source_location",
                "source_update_date",
            ]
            changed = False
            for attr in business_attrs:
                if getattr(existing_row, attr) != getattr(obj, attr):
                    setattr(existing_row, attr, getattr(obj, attr))
                    changed = True
            if changed:
                existing_row.updated_at = datetime.now(timezone.utc)
                existing_row.import_timestamp = datetime.now(timezone.utc)
        if to_add:
            session.add_all(to_add)
        # commit handled by caller

    # ---------------------------------------------------------------------
    # Stats helper
    # ---------------------------------------------------------------------
    def _reset_stats(self, source: str) -> None:
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "duplicate_ids": 0,
            "price_parsed": 0,
            "price_unparsed": 0,
            "missing_configuration": 0,
            "missing_property_type": 0,
            "missing_location": 0,
            "rejection_reasons": [],
            "batches": 0,
            "mode": "dry-run" if self.dry_run else "apply",
            "source": source,
        }

    # ---------------------------------------------------------------------
    # Row processing helpers
    # ---------------------------------------------------------------------
    def _process_public_row(self, row: Dict[str, str], seen_ids: Set[str]) -> Tuple[Property | None, str | None]:
        source_id = row.get("Index")
        if source_id is None or source_id == "":
            return None, "missing Index"
        # Accept numeric zero as valid identifier
        source_id_str = str(int(source_id)) if str(source_id).isdigit() else str(source_id)
        if source_id_str in seen_ids:
            self.stats["duplicate_ids"] += 1
            return None, "duplicate Index"
        seen_ids.add(source_id_str)

        location = row.get("location")
        if not location:
            self.stats["missing_location"] += 1
            return None, "missing location"

        title = row.get("Title", "")
        property_type = self._classify_property_type(title)
        if not property_type:
            self.stats["missing_property_type"] += 1
            return None, "unclassifiable property type"

        configuration = self._extract_bhk(title)
        if not configuration:
            self.stats["missing_configuration"] += 1
        description = row.get("Description") or title

        amount_raw = row.get("Amount(in rupees)")
        listing_price, parsed = self._parse_price(amount_raw or "")
        if parsed:
            self.stats["price_parsed"] += 1
        else:
            self.stats["price_unparsed"] += 1

        project_name = row.get("Society")
        if project_name:
            project_name = project_name.strip() or None
        else:
            project_name = None

        prop = Property(
            project_name=project_name,
            location=location,
            property_type=property_type,
            configuration=configuration,
            price_min=None,
            price_max=None,
            listing_price=listing_price,
            possession_date=None,
            amenities=None,
            nearby_locations=None,
            availability_status=None,
            description=description,
            source_name="kaggle_house_price_india",
            source_record_id=source_id_str,
            source_url="https://www.kaggle.com/datasets/juhibhojani/house-price",
            source_location=location,
            source_update_date=None,
            raw_source_metadata={k: (v if v != "" else None) for k, v in row.items()},
        )
        return prop, None

    def _process_synthetic_record(self, rec: Any, seen_ids: Set[str]) -> Tuple[Property | None, str | None]:
        if not isinstance(rec, dict):
            return None, "record not a dict"
        source_id = rec.get("property_id")
        if source_id is None or source_id == "":
            return None, "missing property_id"
        source_id_str = str(source_id)
        # Track IDs to avoid processing same record within a single import run
        if source_id_str in seen_ids:
            # Duplicate within the same file – count as duplicate but still return object for upsert handling
            self.stats["duplicate_ids"] += 1
        seen_ids.add(source_id_str)

        location = rec.get("location")
        if not location:
            self.stats["missing_location"] += 1
            return None, "missing location"
        property_type = rec.get("property_type")
        if not property_type:
            self.stats["missing_property_type"] += 1
            return None, "missing property_type"

        # Price fields – ensure Decimal and validate
        price_min_raw = rec.get("price_min")
        price_max_raw = rec.get("price_max")
        try:
            price_min = Decimal(str(price_min_raw)) if price_min_raw is not None else None
            price_max = Decimal(str(price_max_raw)) if price_max_raw is not None else None
            if price_min is not None:
                self.stats["price_parsed"] += 1
            if price_max is not None:
                self.stats["price_parsed"] += 1
        except Exception:
            return None, "invalid price numeric values"

        possession_raw = rec.get("possession_date")
        possession_date = None
        if possession_raw:
            try:
                possession_date = datetime.fromisoformat(possession_raw)
            except Exception:
                return None, "invalid possession_date format"

        prop = Property(
            project_name=rec.get("project_name"),
            location=location,
            property_type=property_type,
            configuration=rec.get("configuration"),
            price_min=price_min,
            price_max=price_max,
            listing_price=rec.get("listing_price"),
            possession_date=possession_date,
            amenities=rec.get("amenities"),
            nearby_locations=rec.get("nearby_locations"),
            availability_status=rec.get("availability_status"),
            description=rec.get("description"),
            source_name="synthetic",
            source_record_id=source_id_str,
            source_url=None,
            source_location=location,
            source_update_date=None,
            raw_source_metadata={k: (v if v != "" else None) for k, v in rec.items()},
        )
        return prop, None


    # ---------------------------------------------------------------------
    # Public import entry points
    # ---------------------------------------------------------------------
    def import_public_csv(self, csv_path: Path) -> Dict[str, Any]:
        """Import the public Kaggle CSV dataset.

        In dry‑run mode no DB interaction occurs.  In apply mode a single
        transaction is used for the whole source.
        """
        self._reset_stats("kaggle_house_price_india")
        seen_ids: Set[str] = set()
        batch: List[Property] = []

        # Header validation
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
            expected = [
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
            missing = [c for c in expected if c not in header]
            if missing:
                raise ValueError(f"CSV header missing required columns: {missing}")
            f.seek(0)
            reader = csv.DictReader(f)
            if not self.dry_run:
                session = self.session_factory()
                session.begin()
            try:
                for row in reader:
                    self.stats["total_records"] += 1
                    prop, reason = self._process_public_row(row, seen_ids)
                    if prop is None:
                        self.stats["rejected_records"] += 1
                        if reason:
                            self.stats["rejection_reasons"].append(reason)
                        continue
                    self.stats["valid_records"] += 1
                    if not self.dry_run:
                        batch.append(prop)
                        if len(batch) >= self.batch_size:
                            self._upsert_batch(batch, session)
                            batch.clear()
                            self.stats["batches"] += 1
                # final batch
                if not self.dry_run and batch:
                    self._upsert_batch(batch, session)
                    self.stats["batches"] += 1
                if not self.dry_run:
                    session.commit()
            except Exception:
                if not self.dry_run:
                    session.rollback()
                raise
            finally:
                if not self.dry_run:
                    session.close()
        # invariant check
        if self.stats["total_records"] != self.stats["valid_records"] + self.stats["rejected_records"]:
            raise AssertionError("Statistics invariant violated: total != valid + rejected")
        return self.stats

    def import_synthetic_json(self, json_path: Path) -> Dict[str, Any]:
        """Import the synthetic JSON property dataset.

        Dry‑run performs only validation; apply mode uses a single transaction.
        """
        self._reset_stats("synthetic")
        seen_ids: Set[str] = set()
        batch: List[Property] = []
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Synthetic JSON root must be a list of records")
            if not self.dry_run:
                session = self.session_factory()
                session.begin()
            try:
                for rec in data:
                    self.stats["total_records"] += 1
                    prop, reason = self._process_synthetic_record(rec, seen_ids)
                    if prop is None:
                        self.stats["rejected_records"] += 1
                        if reason:
                            self.stats["rejection_reasons"].append(reason)
                        continue
                    self.stats["valid_records"] += 1
                    if not self.dry_run:
                        batch.append(prop)
                        if len(batch) >= self.batch_size:
                            self._upsert_batch(batch, session)
                            batch.clear()
                            self.stats["batches"] += 1
                if not self.dry_run and batch:
                    self._upsert_batch(batch, session)
                    self.stats["batches"] += 1
                if not self.dry_run:
                    session.commit()
            except Exception:
                if not self.dry_run:
                    session.rollback()
                raise
            finally:
                if not self.dry_run:
                    session.close()
        if self.stats["total_records"] != self.stats["valid_records"] + self.stats["rejected_records"]:
            raise AssertionError("Statistics invariant violated: total != valid + rejected")
        return self.stats
