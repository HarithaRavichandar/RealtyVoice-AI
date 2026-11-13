
# -*- coding: utf-8 -*-
"""FastAPI integration tests for the property-search API.

These tests use an in-memory SQLite database created fresh for each
test function. The PropertySearchService dependency is overridden so
that the API operates on the temporary database rather than production.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Generator

import pytest
from backend.models.property import Property
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base
from backend.routes.property_search import get_property_search_service
from backend.services.property_search import PropertySearchService


# ---------------------------------------------------------------------------
# Temporary database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def temp_engine() -> Generator:
    """Create a fresh in-memory SQLite engine for each test."""

    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def temp_session_factory(temp_engine) -> Generator:
    """Return a session factory bound to the temporary engine."""

    factory = sessionmaker(
        bind=temp_engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )

    yield factory


# ---------------------------------------------------------------------------
# FastAPI dependency override
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def override_dependencies(temp_session_factory):
    """Make the property search API use the temporary database.

    Existing dependency overrides are preserved and restored after
    each test so unrelated application dependencies are unaffected.
    """

    previous_overrides = app.dependency_overrides.copy()

    def _override_service():
        return PropertySearchService(
            session_factory=temp_session_factory
        )

    app.dependency_overrides[get_property_search_service] = _override_service

    try:
        yield
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


# ---------------------------------------------------------------------------
# Helper to insert a property record
# ---------------------------------------------------------------------------

def _add_property(factory, **kwargs):
    """Insert a property using the supplied temporary session factory."""

    session = factory()

    try:
        prop = Property(**kwargs)
        session.add(prop)
        session.commit()

        # Capture the primary key before closing the session.
        property_id = prop.property_id

        return property_id

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_search_by_location_and_type(temp_session_factory):
    """Search properties by location and property type."""

    # Public record (historical).
    _add_property(
        temp_session_factory,
        location="Mumbai",
        property_type="flat",
        configuration="2 BHK",
        price_min=Decimal("4000000"),
        price_max=Decimal("5000000"),
        source_name="kaggle_house_price_india",
        source_record_id="pub-1",
    )

    # Synthetic demo record.
    _add_property(
        temp_session_factory,
        location="Delhi",
        property_type="apartment",
        configuration="3 BHK",
        listing_price=Decimal("6000000"),
        source_name="synthetic",
        source_record_id="syn-1",
    )

    client = TestClient(app)

    response = client.get(
        "/properties/search",
        params={
            "location": "mumbai",
            "property_type": "flat",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1

    item = data["items"][0]

    assert item["location"] == "Mumbai"
    assert item["property_type"] == "flat"
    assert item["price_note"] == "historical"
    assert item["is_synthetic"] is False

    # Raw source metadata must not be exposed.
    assert "raw_source_metadata" not in item


def test_configuration_and_budget_filters_and_sorting(temp_session_factory):
    """Test configuration matching, budget filtering and sorting."""

    # Public record with an 8-9 million price interval.
    _add_property(
        temp_session_factory,
        location="Bengaluru",
        property_type="villa",
        configuration="4 BHK",
        price_min=Decimal("8000000"),
        price_max=Decimal("9000000"),
        source_name="kaggle_house_price_india",
        source_record_id="pub-2",
    )

    # Synthetic record with a lower price.
    _add_property(
        temp_session_factory,
        location="Bengaluru",
        property_type="villa",
        configuration="3 BHK",
        listing_price=Decimal("3000000"),
        source_name="synthetic",
        source_record_id="syn-2",
    )

    client = TestClient(app)

    # Test a non-breaking hyphen variant and maximum budget.
    response = client.get(
        "/properties/search",
        params={
            "configuration": "4\u2011BHK",
            "max_price": "8500000",
            "sort": "price_desc",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1

    item = data["items"][0]

    assert item["configuration"] == "4 BHK"

    assert (
        item["price_max"] == str(Decimal("9000000"))
        or item["price_max"] == 9000000
    )


def test_pagination_and_provenance(temp_session_factory):
    """Test pagination, sorting and source provenance."""

    # Insert three records with known prices.
    for i, price in enumerate(
        [
            Decimal("1000000"),
            Decimal("2000000"),
            Decimal("3000000"),
        ]
    ):
        _add_property(
            temp_session_factory,
            location="Chennai",
            property_type="flat",
            configuration="2 BHK",
            listing_price=price,
            source_name=(
                "synthetic"
                if i % 2 == 0
                else "kaggle_house_price_india"
            ),
            source_record_id=f"rec-{i}",
        )

    client = TestClient(app)

    response = client.get(
        "/properties/search",
        params={
            "limit": 2,
            "offset": 1,
            "sort": "price_asc",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 3
    assert len(data["items"]) == 2

    # The second and third cheapest records should be returned.
    assert data["items"][0]["listing_price"] == 2000000
    assert data["items"][1]["listing_price"] == 3000000

    # Provenance verification.
    assert data["items"][0]["is_synthetic"] is False
    assert data["items"][0]["price_note"] == "historical"

    assert data["items"][1]["is_synthetic"] is True
    assert data["items"][1]["price_note"] == "demo data"


def test_get_property_detail_success_and_404(temp_session_factory):
    """Test property detail retrieval and missing-property response."""

    property_id = _add_property(
        temp_session_factory,
        location="Kochi",
        property_type="flat",
        configuration="1 BHK",
        price_min=Decimal("1500000"),
        source_name="kaggle_house_price_india",
        source_record_id="pub-99",
    )

    client = TestClient(app)

    # Existing property should return 200.
    response = client.get(f"/properties/{property_id}")

    assert response.status_code == 200

    item = response.json()

    assert item["property_id"] == property_id
    assert item["price_note"] == "historical"
    assert item["is_synthetic"] is False

    # Non-existent property should return 404.
    response_404 = client.get("/properties/999999")

    assert response_404.status_code == 404


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "params",
    [
        {"min_price": -100},
        {"max_price": -5},
        {"min_price": 100, "max_price": 50},
        {"sort": "price_invalid"},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_invalid_query_parameters(params, temp_session_factory):
    """Invalid query parameters should return HTTP 422."""

    # An empty initialized temporary database is sufficient.
    client = TestClient(app)

    response = client.get(
        "/properties/search",
        params=params,
    )

    assert response.status_code == 422
