# -*- coding: utf-8 -*-
"""Tests for availability phrase normalization.

Customer phrasings such as "ready to move in" must map to the dataset's
canonical availability_status values so the search never silently returns
nothing. Unknown values pass through unchanged.
"""

from datetime import datetime

import pytest

from backend.models import Conversation
from backend.models.property import Property
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
from backend.orchestrator.services.tool_executor import execute_tool
from backend.orchestrator.tools.schemas import (
    SearchPropertiesRequest,
    UpdateCustomerRequirementsRequest,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Ready to Move", "Ready to Move"),
        ("ready to move in", "Ready to Move"),
        ("Ready-To-Move-In", "Ready to Move"),
        ("  READY  ", "Ready to Move"),
        ("RTM", "Ready to Move"),
        ("immediate possession", "Ready to Move"),
        ("Under Construction", "Under Construction"),
        ("under-construction", "Under Construction"),
        ("UC", "Under Construction"),
        # Unknown statuses pass through, trimmed but otherwise unchanged.
        ("  Resale ", "Resale"),
        ("available", "available"),
    ],
)
@pytest.mark.parametrize(
    "schema, field",
    [
        (SearchPropertiesRequest, "availability"),
        (UpdateCustomerRequirementsRequest, "availability"),
    ],
)
def test_availability_normalized(schema, field, raw, expected):
    assert getattr(schema(**{field: raw}), field) == expected


@pytest.mark.parametrize(
    "schema", [SearchPropertiesRequest, UpdateCustomerRequirementsRequest]
)
def test_availability_whitespace_rejected(schema):
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        schema(availability="   ")


def test_availability_none_allowed():
    assert SearchPropertiesRequest().availability is None


def test_prompt_lists_valid_availability_values():
    assert "Ready to Move" in SYSTEM_PROMPT
    assert "Under Construction" in SYSTEM_PROMPT


@pytest.fixture
def listings(dbsession):
    ready = Property(
        location="Whitefield",
        property_type="apartment",
        configuration="2 BHK",
        price_min=6_000_000,
        price_max=7_000_000,
        listing_price=6_500_000,
        possession_date=datetime(2025, 6, 1),
        availability_status="Ready to Move",
    )
    building = Property(
        location="Whitefield",
        property_type="apartment",
        configuration="2 BHK",
        price_min=6_000_000,
        price_max=7_000_000,
        listing_price=6_500_000,
        possession_date=datetime(2027, 6, 1),
        availability_status="Under Construction",
    )
    dbsession.add_all([ready, building])
    conv = Conversation(
        channel="browser_text",
        external_session_id="avail-session",
        temp_requirements=None,
    )
    dbsession.add(conv)
    dbsession.flush()
    return ready, building, conv.conversation_id


def test_saved_phrase_finds_ready_to_move_end_to_end(dbsession, listings):
    ready, _building, cid = listings

    # The customer says "ready to move in"; the LLM saves it as stated.
    saved = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"availability": "ready to move in"},
        conversation_id=cid,
    )
    assert "error" not in saved

    # A later search that omits availability applies the saved, canonical value.
    result = execute_tool(
        dbsession, "search_properties", {}, conversation_id=cid
    )

    assert [i["property_id"] for i in result["items"]] == [ready.property_id]
    assert result["applied_defaults"] == {"availability": "Ready to Move"}


def test_explicit_phrase_in_search_is_normalized(dbsession, listings):
    _ready, building, cid = listings

    result = execute_tool(
        dbsession,
        "search_properties",
        {"availability": "under-construction"},
        conversation_id=cid,
    )

    assert [i["property_id"] for i in result["items"]] == [
        building.property_id
    ]