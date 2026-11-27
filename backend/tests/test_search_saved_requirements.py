# -*- coding: utf-8 -*-
"""Tests for saved-requirement fallbacks in the search_properties tool.

Saved ``Conversation.temp_requirements`` act as fallbacks for search
criteria the LLM did not pass. Explicit arguments always win, and every
filter that came from saved requirements is reported in ``applied_defaults``.
"""

from datetime import datetime

import pytest

from backend.models import Conversation
from backend.models.property import Property
from backend.orchestrator.services.tool_executor import execute_tool


@pytest.fixture
def props(dbsession):
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
        location="Mumbai",
        property_type="apartment",
        configuration="3 BHK",
        price_min=3_000_000,
        price_max=5_000_000,
        listing_price=4_000_000,
        possession_date=datetime(2025, 12, 31),
        availability_status="under construction",
    )
    p3 = Property(
        location="Delhi",
        property_type="villa",
        configuration="3 BHK",
        price_min=3_000_000,
        price_max=5_000_000,
        listing_price=4_000_000,
        possession_date=datetime(2025, 12, 31),
        availability_status="available",
    )
    dbsession.add_all([p1, p2, p3])
    dbsession.flush()
    return {"p1": p1, "p2": p2, "p3": p3}


def _conversation(dbsession, saved):
    conv = Conversation(
        channel="browser_text",
        external_session_id="test-session",
        temp_requirements=saved,
    )
    dbsession.add(conv)
    dbsession.flush()
    return conv.conversation_id


def _ids(result):
    return {item["property_id"] for item in result["items"]}


def _search(dbsession, payload, conversation_id):
    return execute_tool(
        dbsession, "search_properties", payload, conversation_id=conversation_id
    )


# 1. Multi-turn: earlier turns saved budget and location; this turn only
#    adds configuration.
def test_saved_budget_and_location_applied_across_turns(dbsession, props):
    cid = _conversation(
        dbsession, {"budget_max": "2000000", "location": "Mumbai"}
    )
    result = _search(dbsession, {"configuration": "2 BHK"}, cid)

    assert _ids(result) == {props["p1"].property_id}
    assert result["applied_defaults"] == {
        "max_budget": "2000000",
        "location": "Mumbai",
    }


# 2. Explicit configuration overrides the saved one (2 BHK -> 3 BHK).
def test_explicit_configuration_overrides_saved(dbsession, props):
    cid = _conversation(
        dbsession, {"configuration": "2 BHK", "location": "Mumbai"}
    )
    result = _search(dbsession, {"configuration": "3 BHK"}, cid)

    assert _ids(result) == {props["p2"].property_id}
    assert "configuration" not in result["applied_defaults"]
    assert result["applied_defaults"] == {"location": "Mumbai"}


# 3. Saved availability is applied, and an explicit value overrides it.
def test_saved_availability_applied(dbsession, props):
    cid = _conversation(
        dbsession, {"availability": "available", "location": "Mumbai"}
    )
    result = _search(dbsession, {}, cid)

    assert _ids(result) == {props["p1"].property_id}
    assert result["applied_defaults"]["availability"] == "available"


def test_explicit_availability_overrides_saved(dbsession, props):
    cid = _conversation(
        dbsession, {"availability": "available", "location": "Mumbai"}
    )
    result = _search(dbsession, {"availability": "under construction"}, cid)

    assert _ids(result) == {props["p2"].property_id}
    assert "availability" not in result["applied_defaults"]


# 4. No conversation id / no saved requirements: behaviour is unchanged.
def test_no_conversation_id_preserves_behaviour(dbsession, props):
    result = execute_tool(dbsession, "search_properties", {})

    assert result["total"] == 3
    assert result["applied_defaults"] == {}


@pytest.mark.parametrize("saved", [None, {}])
def test_no_saved_requirements_preserves_behaviour(dbsession, props, saved):
    cid = _conversation(dbsession, saved)
    result = _search(dbsession, {"location": "Delhi"}, cid)

    assert _ids(result) == {props["p3"].property_id}
    assert result["applied_defaults"] == {}


def test_unknown_conversation_id_does_not_crash(dbsession, props):
    result = _search(dbsession, {}, 99999)

    assert result["total"] == 3
    assert result["applied_defaults"] == {}


# 5. Invalid or conflicting saved values are ignored, not fatal.
def test_invalid_saved_values_are_ignored(dbsession, props):
    cid = _conversation(
        dbsession,
        {
            "budget_min": "not-a-number",
            "possession_before": "not-a-date",
            "configuration": "   ",
            "location": "Mumbai",
        },
    )
    result = _search(dbsession, {}, cid)

    assert result["applied_defaults"] == {"location": "Mumbai"}
    assert _ids(result) == {props["p1"].property_id, props["p2"].property_id}


def test_saved_budget_conflicting_with_explicit_budget_is_ignored(
    dbsession, props
):
    cid = _conversation(dbsession, {"budget_min": "3000000"})
    result = _search(dbsession, {"max_budget": "1500000"}, cid)

    assert result["applied_defaults"] == {}
    assert _ids(result) == {props["p1"].property_id}


# Explicit null deliberately drops a saved criterion for this search.
def test_explicit_null_skips_saved_value(dbsession, props):
    cid = _conversation(
        dbsession, {"location": "Mumbai", "configuration": "3 BHK"}
    )
    result = _search(dbsession, {"location": None}, cid)

    assert _ids(result) == {props["p2"].property_id, props["p3"].property_id}
    assert result["applied_defaults"] == {"configuration": "3 BHK"}