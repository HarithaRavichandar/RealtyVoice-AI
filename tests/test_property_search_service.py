# -*- coding: utf-8 -*-
"""Unit tests for PropertySearchService.
"""

from __future__ import annotations
import itertools
import datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.property import Property
from backend.services.property_search import PropertySearchService
from backend.database import Base
_record_ids = itertools.count(1)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def temp_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def temp_session_factory(temp_engine):
    # Create tables
    Base.metadata.create_all(temp_engine)
    return sessionmaker(bind=temp_engine, autoflush=False, autocommit=False, future=True)

def _add_property(session_factory, **kwargs):
    session = session_factory()
    try:
        prop = Property(**kwargs)
        session.add(prop)
        session.commit()
        return prop
    finally:
        session.close()

# ---------------------------------------------------------------------------
# Helper to build a minimal property dict
# ---------------------------------------------------------------------------
def make_base(**overrides):
    base = {
        "project_name": None,
        "location": "Mumbai",
        "property_type": "flat",
        "configuration": "2 BHK",
        "price_min": None,
        "price_max": None,
        "listing_price": None,
        "possession_date": None,
        "amenities": None,
        "nearby_locations": None,
        "availability_status": None,
        "description": None,
        "source_name": "kaggle_house_price_india",
        "source_record_id": f"pub-{next(_record_ids)}",
    }
    base.update(overrides)
    return base

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_location_search(temp_session_factory):
    # two records, different locations
    _add_property(temp_session_factory, **make_base(location="Mumbai"))
    _add_property(temp_session_factory, **make_base(location="Delhi"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(location="mumba")
    assert total == 1
    assert results[0].location == "Mumbai"

def test_property_type_filter(temp_session_factory):
    _add_property(temp_session_factory, **make_base(property_type="apartment"))
    _add_property(temp_session_factory, **make_base(property_type="flat"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(property_type="apartment")
    assert total == 1
    assert results[0].property_type == "apartment"

def test_configuration_filter(temp_session_factory):
    _add_property(temp_session_factory, **make_base(configuration="4 BHK"))
    _add_property(temp_session_factory, **make_base(configuration="3 BHK"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(configuration="4‑BHK")  # non‑breaking hyphen variant
    assert total == 1
    assert results[0].configuration == "4 BHK"

def test_max_price_filter(temp_session_factory):
    _add_property(temp_session_factory, **make_base(listing_price=Decimal("5000000")))
    _add_property(temp_session_factory, **make_base(listing_price=Decimal("10000000")))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(max_price=Decimal("6000000"))
    assert total == 1
    assert results[0].listing_price == Decimal("5000000")

def test_min_price_filter(temp_session_factory):
    _add_property(temp_session_factory, **make_base(price_min=Decimal("2000000"), price_max=Decimal("3000000")))
    _add_property(temp_session_factory, **make_base(price_min=Decimal("4000000")))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(min_price=Decimal("3500000"))
    # Only second record has low bound >= 3.5M, first high bound is 3M < min
    assert total == 1
    assert results[0].price_min == Decimal("4000000")

def test_budget_interval_overlap(temp_session_factory):
    # Record price interval 2M‑4M
    _add_property(
        temp_session_factory,
        **make_base(price_min=Decimal("2000000"), price_max=Decimal("4000000")),
    )
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, _ = svc.search(min_price=Decimal("3000000"), max_price=Decimal("5000000"))
    assert total == 1
    # Non‑overlapping interval should return 0
    total2, _ = svc.search(min_price=Decimal("5000000"), max_price=Decimal("6000000"))
    assert total2 == 0

def test_unknown_price_not_filtered_when_no_budget(temp_session_factory):
    _add_property(temp_session_factory, **make_base())  # all prices null
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search()
    assert total == 1
    assert results[0].price_min is None

def test_possession_before_filter(temp_session_factory):
    date1 = datetime.datetime(2025, 1, 1)
    date2 = datetime.datetime(2026, 1, 1)
    _add_property(temp_session_factory, **make_base(possession_date=date1))
    _add_property(temp_session_factory, **make_base(possession_date=date2))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(possession_before=datetime.date(2025, 12, 31))
    assert total == 1
    assert results[0].possession_date == date1

def test_availability_filter(temp_session_factory):
    _add_property(temp_session_factory, **make_base(availability_status="ready"))
    _add_property(temp_session_factory, **make_base())
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(availability="ready")
    assert total == 1
    assert results[0].availability_status == "ready"

def test_sorting_null_prices_last(temp_session_factory):
    _add_property(temp_session_factory, **make_base(listing_price=Decimal("3000000")))
    _add_property(temp_session_factory, **make_base())  # null price
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(sort="price_asc")
    assert total == 2
    assert results[0].listing_price == Decimal("3000000")
    assert results[1].listing_price is None

def test_exact_configuration_match(temp_session_factory):
    # 4 BHK should match only itself, not 14 BHK
    _add_property(temp_session_factory, **make_base(configuration="4 BHK"))
    _add_property(temp_session_factory, **make_base(configuration="14 BHK"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(configuration="4 BHK")
    assert total == 1
    assert results[0].configuration == "4 BHK"

def test_configuration_variant_normalisation(temp_session_factory):
    # non‑breaking hyphen variant should match normalised configuration
    _add_property(temp_session_factory, **make_base(configuration="4 BHK"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search(configuration="4\u2011BHK")  # U+2011 non‑breaking hyphen
    assert total == 1
    assert results[0].configuration == "4 BHK"

def test_provenance_flags(temp_session_factory):
    # synthetic source
    _add_property(temp_session_factory, **make_base(source_name="synthetic"))
    # public source
    _add_property(temp_session_factory, **make_base(source_name="kaggle_house_price_india"))
    # unknown source
    _add_property(temp_session_factory, **make_base(source_name="other"))
    svc = PropertySearchService(session_factory=temp_session_factory)
    total, results = svc.search()
    assert total == 3
    # Verify flags via direct ORM access (service does not expose flags, but we can check fields)
    synthetic = next(p for p in results if p.source_name == "synthetic")
    public = next(p for p in results if p.source_name == "kaggle_house_price_india")
    unknown = next(p for p in results if p.source_name == "other")
    # No explicit flags in ORM, just ensure source_name is stored correctly
    assert synthetic.source_name == "synthetic"
    assert public.source_name == "kaggle_house_price_india"
    assert unknown.source_name == "other"

