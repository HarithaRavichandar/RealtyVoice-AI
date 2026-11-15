
# -*- coding: utf-8 -*-

from uuid import uuid4

from sqlalchemy import text

from backend.models import Conversation
from backend.orchestrator.services.tool_executor import execute_tool


def create_conversation(dbsession, requirements=None):
    conversation = Conversation(
        lead_id=None,
        channel="browser_text",
        external_session_id=uuid4().hex,
        temp_requirements=requirements,
    )
    dbsession.add(conversation)
    dbsession.flush()
    return conversation


def test_update_single_requirement(dbsession):
    conversation = create_conversation(dbsession)

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": "Bangalore"},
        conversation.conversation_id,
    )

    assert result["tool"] == "update_customer_requirements"
    assert result["requirements"]["location"] == "Bangalore"


def test_preserves_untouched_requirements(dbsession):
    conversation = create_conversation(
        dbsession,
        {
            "location": "Chennai",
            "property_type": "Apartment",
            "budget_max": "5000000",
        },
    )

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": "Bangalore"},
        conversation.conversation_id,
    )

    requirements = result["requirements"]

    assert requirements["location"] == "Bangalore"
    assert requirements["property_type"] == "Apartment"
    assert requirements["budget_max"] == "5000000"


def test_explicit_none_overwrites_existing_value(dbsession):
    conversation = create_conversation(
        dbsession,
        {"location": "Chennai"},
    )

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": None},
        conversation.conversation_id,
    )

    assert result["requirements"]["location"] is None


def test_updates_multiple_requirements(dbsession):
    conversation = create_conversation(dbsession)

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {
            "location": "Bangalore",
            "property_type": "Apartment",
            "configuration": "2BHK",
        },
        conversation.conversation_id,
    )

    requirements = result["requirements"]

    assert requirements["location"] == "Bangalore"
    assert requirements["property_type"] == "Apartment"
    assert requirements["configuration"] == "2BHK"


def test_missing_conversation_id(dbsession):
    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": "Bangalore"},
    )

    assert result["error"] == "missing_conversation_id"


def test_nonexistent_conversation_id(dbsession):
    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": "Bangalore"},
        999999,
    )

    assert result["error"] == "conversation_not_found"


def test_invalid_payload(dbsession):
    conversation = create_conversation(dbsession)

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {"budget_min": -100},
        conversation.conversation_id,
    )

    assert result["error"] == "validation"
    assert result["details"]


def test_decimal_and_date_serialization(dbsession):
    conversation = create_conversation(dbsession)

    result = execute_tool(
        dbsession,
        "update_customer_requirements",
        {
            "budget_min": "1234.56",
            "possession_before": "2027-06-30",
        },
        conversation.conversation_id,
    )

    requirements = result["requirements"]

    assert requirements["budget_min"] == "1234.56"
    assert requirements["possession_before"] == "2027-06-30"


def test_session_remains_usable(dbsession):
    conversation = create_conversation(dbsession)

    execute_tool(
        dbsession,
        "update_customer_requirements",
        {"location": "Bangalore"},
        conversation.conversation_id,
    )

    assert dbsession.execute(text("SELECT 1")).scalar() == 1
