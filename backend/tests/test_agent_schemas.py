import pytest
from datetime import date
from decimal import Decimal
from pydantic import ValidationError
from backend.orchestrator.schemas import AgentRequest, CustomerRequirements, NextAction, AgentResponse

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
