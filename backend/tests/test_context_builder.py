"""Tests for the context builder utilities.

These tests use the ``dbsession`` fixture from ``test_migration_orm_compat.py``
which provides an isolated SQLite transaction.
"""

import logging
import pytest

from datetime import date
from decimal import Decimal
from sqlalchemy.orm import Session

from backend.orchestrator.services.context_builder import build_context
from backend.orchestrator.services.conversation import create_conversation, update_temp_requirements
from backend.orchestrator.services.message import add_message, add_tool_message
from backend.orchestrator.schemas import CustomerRequirements


def test_empty_history_and_missing_requirements(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    dbsession.commit()
    ctx = build_context(
        dbsession, conv.external_session_id, incoming_message="Hello", history_limit=20
    )
    assert ctx["history"] == []
    # requirements should be empty CustomerRequirements
    assert isinstance(ctx["requirements"], CustomerRequirements)
    assert ctx["requirements"].dict() == {
        "budget_min": None,
        "budget_max": None,
        "location": None,
        "property_type": None,
        "configuration": None,
        "possession_before": None,
        "availability": None,
    }


def test_chronological_history(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_text")
    # Customer message turn 1
    add_message(dbsession, conv.conversation_id, turn_id=1, content="I want a house", role="customer")
    # Assistant message turn 2
    add_message(dbsession, conv.conversation_id, turn_id=2, content="Sure, tell me budget", role="assistant")
    dbsession.commit()

    ctx = build_context(dbsession, conv.external_session_id, incoming_message="500k", history_limit=20)
    assert len(ctx["history"]) == 2
    assert ctx["history"][0]["turn_id"] == 1
    assert ctx["history"][0]["role"] == "customer"
    assert ctx["history"][1]["turn_id"] == 2
    assert ctx["history"][1]["role"] == "assistant"


def test_history_limit_truncation(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    for i in range(1, 6):
        add_message(dbsession, conv.conversation_id, turn_id=i, content=f"msg{i}", role="customer")
    dbsession.commit()
    ctx = build_context(dbsession, conv.external_session_id, incoming_message="last", history_limit=2)
    assert len(ctx["history"]) == 2
    # Should contain the last two messages (turn 4 and 5)
    assert ctx["history"][0]["turn_id"] == 4
    assert ctx["history"][1]["turn_id"] == 5


def test_zero_history_limit(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    add_message(dbsession, conv.conversation_id, turn_id=1, content="first", role="customer")
    dbsession.commit()
    ctx = build_context(dbsession, conv.external_session_id, incoming_message="now", history_limit=0)
    assert ctx["history"] == []


def test_invalid_history_limit(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    dbsession.commit()
    with pytest.raises(ValueError, match="history_limit must be a non‑negative integer"):
        build_context(dbsession, conv.external_session_id, incoming_message="msg", history_limit=-1)
    with pytest.raises(ValueError, match="history_limit must be a non‑negative integer"):
        build_context(dbsession, conv.external_session_id, incoming_message="msg", history_limit="bad")


def test_valid_stored_requirements(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    req = {
        "budget_min": Decimal("500000"),
        "budget_max": Decimal("800000"),
        "location": "Delhi",
        "property_type": "apartment",
        "configuration": "2 BHK",
        "possession_before": date(2025, 12, 31),
        "availability": "ready",
    }
    update_temp_requirements(dbsession, conv.conversation_id, req)
    dbsession.commit()
    ctx = build_context(dbsession, conv.external_session_id, incoming_message="search", history_limit=20)
    r = ctx["requirements"]
    assert isinstance(r, CustomerRequirements)
    assert r.budget_min == Decimal("500000")
    assert r.budget_max == Decimal("800000")
    assert r.location == "Delhi"
    assert r.property_type == "apartment"
    assert r.configuration == "2 BHK"
    assert r.possession_before == date(2025, 12, 31)
    assert r.availability == "ready"


def test_invalid_requirements_type_and_values(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    # Store a non‑dict value
    update_temp_requirements(dbsession, conv.conversation_id, "not a dict")
    dbsession.commit()
    ctx = build_context(dbsession, conv.external_session_id, incoming_message="msg", history_limit=20)
    # Should fallback to empty requirements
    assert ctx["requirements"].dict() == {
        "budget_min": None,
        "budget_max": None,
        "location": None,
        "property_type": None,
        "configuration": None,
        "possession_before": None,
        "availability": None,
    }

    # Now store a dict with invalid budget values
    bad_req = {"budget_min": -100, "budget_max": -50}
    update_temp_requirements(dbsession, conv.conversation_id, bad_req)
    dbsession.commit()
    ctx2 = build_context(dbsession, conv.external_session_id, incoming_message="msg", history_limit=20)
    # Validation fails, fallback to empty
    assert ctx2["requirements"].dict() == {
        "budget_min": None,
        "budget_max": None,
        "location": None,
        "property_type": None,
        "configuration": None,
        "possession_before": None,
        "availability": None,
    }


def test_incoming_message_validation(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_voice")
    dbsession.commit()
    with pytest.raises(ValueError, match="incoming_message must be a non‑empty string"):
        build_context(dbsession, conv.external_session_id, incoming_message="   ", history_limit=20)


def test_tool_message_separation(dbsession: Session):
    conv = create_conversation(dbsession, channel="browser_text")
    # Regular messages
    add_message(dbsession, conv.conversation_id, turn_id=1, content="hello", role="customer")
    # Tool audit message
    add_tool_message(
        dbsession,
        conv.conversation_id,
        turn_id=2,
        content="tool exec",
        tool_name="search",
        tool_call_id="call-1",
        tool_arguments={"q": "test"},
        tool_status="completed",
        tool_result={"count": 0},
    )
    dbsession.commit()
    ctx = build_context(dbsession, conv.external_session_id, incoming_message="next", history_limit=20)
    # History should contain only the customer message
    assert len(ctx["history"]) == 1
    assert ctx["history"][0]["role"] == "customer"
    # Tool audit should contain the tool message
    assert len(ctx["tool_audit"]) == 1
    assert ctx["tool_audit"][0]["role"] == "tool"


def test_unknown_session_raises(dbsession: Session):
    with pytest.raises(ValueError, match="Conversation not found"):
        build_context(dbsession, "no-such-id", incoming_message="hi", history_limit=20)

"""Note: The tests intentionally avoid checking DB commit behavior because the
fixture rolls back any changes.  The absence of explicit ``session.commit``
calls in the builder is ensured by code review."""

