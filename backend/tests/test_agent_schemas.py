import pytest
from datetime import date, datetime, timezone
from decimal import Decimal

from pydantic import ValidationError

from backend.orchestrator.schemas import (
    AgentRequest,
    CustomerRequirements,
    NextAction,
    AgentResponse,
)
from backend.schemas.property_search import PropertySearchItem
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
def test_agent_request_valid():
    req = AgentRequest(session_id="abc123", message="Hello")
    assert req.session_id == "abc123"

@pytest.mark.parametrize("field,value", [("session_id", ""), ("session_id", "   "), ("message", ""), ("message", "   ")])
def test_agent_request_rejects_empty_values(field, value):
    data = {"session_id": "valid", "message": "Hello"}
    data[field] = value
    with pytest.raises(ValidationError):
        AgentRequest(**data)

@pytest.mark.parametrize("field,value", [("session_id", "x" * 129), ("message", "x" * 4001)])
def test_agent_request_rejects_excessive_length(field, value):
    data = {"session_id": "valid", "message": "Hello"}
    data[field] = value
    with pytest.raises(ValidationError):
        AgentRequest(**data)

def test_agent_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentRequest(session_id="abc", message="Hello", channel="browser_text")

def test_agent_request_rejects_client_requirements():
    with pytest.raises(ValidationError):
        AgentRequest(session_id="abc", message="Hello", requirements={"location": "Chennai"})

def test_customer_requirements_valid():
    req = CustomerRequirements(budget_min=Decimal("5000000"), budget_max=Decimal("8000000"), location="Bangalore", property_type="apartment", configuration="2 BHK", possession_before=date(2027, 12, 31), availability="ready")
    assert req.budget_min == Decimal("5000000")

def test_customer_requirements_rejects_negative_minimum():
    with pytest.raises(ValidationError):
        CustomerRequirements(budget_min=Decimal("-1"))

def test_customer_requirements_rejects_negative_maximum():
    with pytest.raises(ValidationError):
        CustomerRequirements(budget_max=Decimal("-1"))

def test_customer_requirements_rejects_minimum_greater_than_maximum():
    with pytest.raises(ValidationError):
        CustomerRequirements(budget_min=Decimal("10"), budget_max=Decimal("5"))

def test_customer_requirements_accepts_partial_values():
    req = CustomerRequirements(location="Chennai")
    assert req.location == "Chennai" and req.budget_min is None

def test_customer_requirements_accepts_empty_values():
    assert CustomerRequirements().model_dump()["location"] is None

@pytest.mark.parametrize("action", ["ask_clarification", "show_properties", "offer_callback", "await_user", "error"])
def test_next_action_accepts_valid_actions(action):
    assert NextAction(action=action).property_ids == []

def test_next_action_rejects_invalid_action():
    with pytest.raises(ValidationError):
        NextAction(action="invalid_action")

def test_agent_response_accepts_empty_property_list():
    resp = AgentResponse(session_id="sid123", turn_id=1, reply="Hello", intent="general_question", next_action=NextAction(action="await_user"))
    assert resp.matched_properties == []

def test_agent_response_rejects_invalid_intent():
    with pytest.raises(ValidationError):
        AgentResponse(session_id="sid", turn_id=0, reply="Hello", intent="invalid", next_action=NextAction(action="await_user"))
def create_test_property(**overrides):
    data = {
        "property_id": 101,
        "project_name": "Test Residency",
        "location": "Chennai",
        "property_type": "Apartment",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "source_name": "kaggle_house_price_india",
        "is_synthetic": False,
        "price_note": "Dataset-based indicative price",
    }
    data.update(overrides)
    return PropertySearchItem(**data)


def test_synthetic_property_metadata_is_preserved():
    property_item = create_test_property(
        is_synthetic=True,
        source_name="synthetic_demo",
    )

    assert property_item.is_synthetic is True
    assert property_item.source_name == "synthetic_demo"


def test_non_synthetic_dataset_is_not_marked_as_verified():
    property_item = create_test_property(is_synthetic=False)

    assert property_item.is_synthetic is False
    assert property_item.source_name == "kaggle_house_price_india"


def test_property_source_metadata_is_preserved_in_agent_response():
    property_item = create_test_property()

    response = AgentResponse(
        session_id="sid123",
        turn_id=1,
        reply="Here are some dataset-based property records.",
        intent="property_search",
        matched_properties=[property_item],
        next_action=NextAction(action="show_properties"),
    )

    serialized = response.model_dump(mode="json")
    result = serialized["matched_properties"][0]

    assert result["source_name"] == "kaggle_house_price_india"
    assert result["is_synthetic"] is False
    assert result["price_note"] == "Dataset-based indicative price"


def test_agent_response_schema_fields_remain_unchanged():
    expected_fields = {
        "session_id",
        "turn_id",
        "reply",
        "spoken_reply",
        "intent",
        "requirements",
        "matched_properties",
        "next_action",
    }

    assert set(AgentResponse.model_fields.keys()) == expected_fields


def test_system_prompt_contains_property_transparency_rules():
    assert "is_synthetic" in SYSTEM_PROMPT
    assert "source_name" in SYSTEM_PROMPT
    assert "price_note" in SYSTEM_PROMPT
    assert "currently listed" in SYSTEM_PROMPT
    assert "live-verified" in SYSTEM_PROMPT

def test_missing_property_provenance_defaults_safely():
    property_item = create_test_property(
        source_name=None,
        is_synthetic=False,
        price_note=None,
    )

    assert property_item.source_name is None
    assert property_item.is_synthetic is False
    assert property_item.price_note is None