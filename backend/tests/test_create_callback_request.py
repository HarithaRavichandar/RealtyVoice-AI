
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from backend.models import (
    Lead,
    Conversation,
    ConsentRecord,
    CallbackRequest,
)
from backend.orchestrator.services.tool_executor import execute_tool


def create_lead(dbsession):
    lead = Lead(
        customer_name="Test Customer",
        phone="9876543210",
        email="test@example.com",
        qualification_status="Warm",
        pipeline_status="New",
    )
    dbsession.add(lead)
    dbsession.flush()
    return lead


def create_conversation(dbsession, lead_id=None):
    conversation = Conversation(
        lead_id=lead_id,
        channel="browser_text",
        external_session_id=uuid4().hex,
    )
    dbsession.add(conversation)
    dbsession.flush()
    return conversation


def create_consent(
    dbsession,
    lead_id,
    conversation_id,
    granted=True,
    recorded_at=None,
):
    consent = ConsentRecord(
        lead_id=lead_id,
        conversation_id=conversation_id,
        consent_type="callback_contact",
        granted=granted,
        recorded_at=recorded_at or datetime.now(timezone.utc),
        source="web",
        policy_version="1.0",
    )
    dbsession.add(consent)
    dbsession.flush()
    return consent


def valid_callback_payload():
    return {
        "preferred_contact": "Phone",
        "preferred_date": "2026-11-20",
        "preferred_time": "09:30",
        "reason": "Property enquiry",
    }


def test_successful_callback_with_granted_consent(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=True,
    )

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["tool"] == "create_callback_request"

    callback = result["callback"]

    assert callback["callback_id"] is not None
    assert callback["lead_id"] == lead.lead_id
    assert callback["conversation_id"] == conversation.conversation_id
    assert callback["preferred_contact"] == "Phone"
    assert callback["preferred_date"] == "2026-11-20"
    assert callback["preferred_time"] == "09:30"
    assert callback["reason"] == "Property enquiry"
    assert callback["status"] == "Pending"

    saved_callback = dbsession.get(
        CallbackRequest,
        callback["callback_id"],
    )

    assert saved_callback is not None


def test_callback_rejected_when_consent_missing(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["error"] == "callback_consent_required"

    assert dbsession.query(CallbackRequest).count() == 0


def test_callback_rejected_when_consent_denied(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=False,
    )

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["error"] == "callback_consent_required"

    assert dbsession.query(CallbackRequest).count() == 0


def test_later_revocation_overrides_earlier_granted_consent(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    now = datetime.now(timezone.utc)

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=True,
        recorded_at=now - timedelta(minutes=10),
    )

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=False,
        recorded_at=now,
    )

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["error"] == "callback_consent_required"

    assert dbsession.query(CallbackRequest).count() == 0


def test_later_granted_consent_overrides_earlier_revocation(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    now = datetime.now(timezone.utc)

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=False,
        recorded_at=now - timedelta(minutes=10),
    )

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=True,
        recorded_at=now,
    )

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["tool"] == "create_callback_request"
    assert result["callback"]["callback_id"] is not None


def test_missing_conversation_id(dbsession):
    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
    )

    assert result["error"] == "missing_conversation_id"


def test_nonexistent_conversation(dbsession):
    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        999999,
    )

    assert result["error"] == "conversation_not_found"


def test_conversation_without_lead(dbsession):
    conversation = create_conversation(dbsession)

    result = execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert result["error"] == "lead_required"


def test_invalid_callback_payload(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    payload = valid_callback_payload()
    payload["preferred_time"] = "25:30"

    result = execute_tool(
        dbsession,
        "create_callback_request",
        payload,
        conversation.conversation_id,
    )

    assert result["error"] == "validation"

    assert dbsession.query(CallbackRequest).count() == 0


def test_callback_creation_keeps_session_usable(dbsession):
    lead = create_lead(dbsession)
    conversation = create_conversation(
        dbsession,
        lead.lead_id,
    )

    create_consent(
        dbsession,
        lead.lead_id,
        conversation.conversation_id,
        granted=True,
    )

    execute_tool(
        dbsession,
        "create_callback_request",
        valid_callback_payload(),
        conversation.conversation_id,
    )

    assert dbsession.execute(text("SELECT 1")).scalar() == 1
