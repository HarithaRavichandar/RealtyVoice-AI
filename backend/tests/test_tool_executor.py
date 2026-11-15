# -*- coding: utf-8 -*-
"""
Tests for the partial Tool Executor (search_properties & get_property_details).

Uses the `dbsession` fixture from conftest.py and the real `Property` ORM model.
"""

import pytest
from datetime import datetime
from sqlalchemy import text
from backend.orchestrator.services.tool_executor import execute_tool

# Import the Property model
from backend.models.property import Property

@pytest.fixture
def sample_properties(dbsession):
    """Create two Property rows with distinct price ranges for testing.
    The fixture flushes the session so IDs and timestamps are populated.
    """
    p1 = Property(
        location="Mumbai",
        property_type="apartment",
        configuration="2 BHK",
        price_min=1_000_000,
        price_max=2_000_000,
        listing_price=1_500_000,
        possession_date=datetime(2025, 6, 1),
        availability_status="available",
    )
    p2 = Property(
        location="Delhi",
        property_type="villa",
        configuration="3 BHK",
        price_min=3_000_000,
        price_max=5_000_000,
        listing_price=4_000_000,
        possession_date=datetime(2025, 12, 31),
        availability_status="available",
    )
    dbsession.add_all([p1, p2])
    dbsession.flush()
    return [p1, p2]

def test_search_properties_success(dbsession, sample_properties):
    payload = {
        "location": "Mumbai",
        "property_type": "apartment",
        "configuration": "2 BHK",
        "max_budget": "1500000",
        "possession_before": "2025-07-01",
    }
    result = execute_tool(dbsession, "search_properties", payload)
    assert result["tool"] == "search_properties"
    assert result["total"] == 1
    assert isinstance(result["items"], list)
    assert result["items"][0]["location"] == "Mumbai"
    # Verify caller session still usable
    dbsession.execute(text("SELECT 1"))

def test_get_property_details_success(dbsession, sample_properties):
    prop = sample_properties[0]
    payload = {"property_id": prop.property_id}
    result = execute_tool(dbsession, "get_property_details", payload)
    assert result["tool"] == "get_property_details"
    data = result["property"]
    assert data["property_id"] == prop.property_id
    assert data["location"] == prop.location
    dbsession.execute(text("SELECT 1"))

def test_get_property_details_not_found(dbsession):
    payload = {"property_id": 9999}
    result = execute_tool(dbsession, "get_property_details", payload)
    assert result["error"] == "not_found"
    assert result["property_id"] == 9999

def test_search_properties_invalid_payload(dbsession):
    payload = {"unknown": "value"}
    result = execute_tool(dbsession, "search_properties", payload)
    assert result["error"] == "validation"
    assert any("Extra inputs are not permitted" in e["msg"] for e in result["details"])

def test_get_property_details_invalid_id(dbsession):
    payload = {"property_id": -1}
    result = execute_tool(dbsession, "get_property_details", payload)
    assert result["error"] == "validation"
    assert any("greater than or equal to 1" in e["msg"] for e in result["details"])

def test_unknown_tool(dbsession):
    result = execute_tool(dbsession, "nonexistent_tool", {})
    assert result["error"] == "unknown_tool"
    assert result["tool"] == "nonexistent_tool"

def test_search_budget_mapping(dbsession, sample_properties):
    payload = {
        "location": "Mumbai",
        "property_type": "apartment",
        "configuration": "2 BHK",
        "max_budget": "900000",
        "possession_before": "2025-07-01",
    }
    result = execute_tool(dbsession, "search_properties", payload)
    # With max_budget lower than listing_price, the property should be filtered out
    assert result["total"] == 0
    assert result["items"] == []
