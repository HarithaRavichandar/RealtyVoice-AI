# -*- coding: utf-8 -*-
"""Model‑level tests for anonymous conversations, tool audit fields, and cascade behavior.
These tests use an isolated in‑memory SQLite database with foreign‑key enforcement.
"""

from __future__ import annotations

import json
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models.conversation import Conversation
from backend.models.message import Message
from backend.models.callback import CallbackRequest
from backend.models.consent import ConsentRecord
from backend.models.lead import Lead
from backend.models.lead_property_match import LeadPropertyMatch

# ---------------------------------------------------------------------------
# Helper to enable SQLite foreign‑key enforcement for each connection.
# ---------------------------------------------------------------------------
@event.listens_for(create_engine("sqlite://"), "connect")
def _fk_pragma(dbapi_connection, connection_record):  # pragma: no cover
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def engine() -> "Engine":
    """Create a fresh in‑memory SQLite engine with proper pragma and static pool.
    The StaticPool ensures the same connection is reused across threads, which is
    required for the in‑memory database to persist throughout the test.
    """
    eng = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Enable foreign‑key enforcement.
    @event.listens_for(eng, "connect")
    def _fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    return eng

@pytest.fixture(scope="function")
def session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

@pytest.fixture(scope="function")
def dbsession(session_factory):
    sess = session_factory()
    try:
        yield sess
    finally:
        sess.close()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_anonymous_conversation_creation(dbsession):
    conv = Conversation(
        lead_id=None,
        channel="browser_voice",
        external_session_id="sess-1",
        temp_requirements={"budget": 1000000},
    )
    dbsession.add(conv)
    dbsession.commit()
    assert conv.conversation_id is not None
    fetched = dbsession.get(Conversation, conv.conversation_id)
    assert fetched.lead_id is None
    assert fetched.temp_requirements == {"budget": 1000000}

def test_message_tool_audit_fields(dbsession, session_factory):
    # Need a conversation first
    conv = Conversation(
        lead_id=None,
        channel="browser_text",
        external_session_id="sess-2",
    )
    dbsession.add(conv)
    dbsession.commit()
    msg = Message(
        conversation_id=conv.conversation_id,
        turn_id=1,
        role="assistant",
        content="tool response",
        tool_name="search",
        tool_call_id="call-123",
        tool_arguments={"q": "test"},
        tool_status="success",
        tool_result={"items": []},
        tool_error=None,
    )
    dbsession.add(msg)
    dbsession.commit()
    fetched = dbsession.get(Message, msg.message_id)
    assert fetched.tool_name == "search"
    assert fetched.tool_arguments == {"q": "test"}
    assert fetched.tool_result == {"items": []}

def test_callback_nullable_lead(dbsession, session_factory):
    # Create a conversation to link optionally
    conv = Conversation(
        lead_id=None,
        channel="telephone",
        external_session_id="sess-3",
    )
    dbsession.add(conv)
    dbsession.commit()
    cb = CallbackRequest(
        lead_id=None,
        conversation_id=conv.conversation_id,
        preferred_contact="email",
        preferred_date="2024-01-01",
        preferred_time="10:00",
        status="Pending",
    )
    dbsession.add(cb)
    dbsession.commit()
    fetched = dbsession.get(CallbackRequest, cb.callback_id)
    assert fetched.lead_id is None
    assert fetched.conversation_id == conv.conversation_id

def test_deleting_conversation_nulls_callback_and_consent(dbsession, session_factory):
    # Setup lead, conversation, callback, consent
    lead = Lead(
        qualification_status="Hot",
        pipeline_status="New",
    )
    dbsession.add(lead)
    dbsession.flush()
    conv = Conversation(
        lead_id=lead.lead_id,
        channel="browser_voice",
        external_session_id="sess-4",
    )
    dbsession.add(conv)
    dbsession.flush()
    cb = CallbackRequest(
        lead_id=lead.lead_id,
        conversation_id=conv.conversation_id,
        preferred_contact="phone",
        preferred_date="2024-01-02",
        preferred_time="12:00",
        status="Pending",
    )
    consent = ConsentRecord(
        lead_id=lead.lead_id,
        conversation_id=conv.conversation_id,
        consent_type="contact_storage",
        granted=True,
        source="web",
        policy_version="v1",
    )
    dbsession.add_all([cb, consent])
    dbsession.commit()
    # Delete the conversation
    dbsession.delete(conv)
    dbsession.commit()
    # Verify callback still exists and conversation_id is NULL
    cb_fetched = dbsession.get(CallbackRequest, cb.callback_id)
    assert cb_fetched is not None
    assert cb_fetched.conversation_id is None
    # Verify consent still exists and conversation_id is NULL
    consent_fetched = dbsession.get(ConsentRecord, consent.consent_id)
    assert consent_fetched is not None
    assert consent_fetched.conversation_id is None

def test_deleting_lead_nulls_consent(dbsession, session_factory):
    lead = Lead(
        qualification_status="Warm",
        pipeline_status="Contacted",
    )
    dbsession.add(lead)
    dbsession.flush()
    conv = Conversation(
        lead_id=lead.lead_id,
        channel="browser_text",
        external_session_id="sess-5",
    )
    dbsession.add(conv)
    dbsession.flush()
    consent = ConsentRecord(
        lead_id=lead.lead_id,
        conversation_id=conv.conversation_id,
        consent_type="marketing_communications",
        granted=False,
        source="phone",
        policy_version="v2",
    )
    dbsession.add(consent)
    dbsession.commit()
    # Delete lead – this also deletes the conversation because of cascade delete-orphan.
    dbsession.delete(lead)
    dbsession.commit()
    # Consent should still exist, lead_id set to NULL, conversation_id set to NULL (conversation deleted).
    fetched = dbsession.get(ConsentRecord, consent.consent_id)
    assert fetched is not None
    assert fetched.lead_id is None
    assert fetched.conversation_id is None
    # The conversation row should be gone.
    conv_fetched = dbsession.get(Conversation, conv.conversation_id)
    assert conv_fetched is None


def test_lead_property_match_unique_constraint(dbsession, session_factory):
    # Create a lead and a property (minimal fields required)
    lead = Lead(
        qualification_status="Cold",
        pipeline_status="Lost",
    )
    from backend.models.property import Property
    prop = Property(
        location="TestCity",
        property_type="flat",
        configuration="1 BHK",
        source_name="test",
        source_record_id="test-1",
    )
    dbsession.add_all([lead, prop])
    dbsession.flush()
    match1 = LeadPropertyMatch(lead_id=lead.lead_id, property_id=prop.property_id)
    dbsession.add(match1)
    dbsession.commit()
    # Duplicate should raise an IntegrityError
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        dup = LeadPropertyMatch(lead_id=lead.lead_id, property_id=prop.property_id)
        dbsession.add(dup)
        dbsession.flush()

def test_enum_values_unchanged(dbsession):
    # Verify enum string values are as expected by creating minimal rows.
    # Conversation
    conv = Conversation(
        lead_id=None,
        channel="browser_voice",
        external_session_id="sess-6",
    )
    dbsession.add(conv)
    dbsession.flush()  # ensure conv.conversation_id is assigned
    # Lead – to check qualification and pipeline enums
    lead = Lead(
        qualification_status="Hot",
        pipeline_status="New",
    )
    dbsession.add(lead)
    dbsession.flush()
    # Message
    msg = Message(
        conversation_id=conv.conversation_id,
        turn_id=1,
        role="customer",
        content="hi",
    )
    # Callback
    cb = CallbackRequest(
        lead_id=None,
        conversation_id=conv.conversation_id,
        preferred_contact="email",
        preferred_date="2024-01-03",
        preferred_time="09:00",
        status="Completed",
    )
    # Consent
    consent = ConsentRecord(
        lead_id=None,
        conversation_id=conv.conversation_id,
        consent_type="callback_contact",
        granted=True,
        source="web",
        policy_version="v1",
    )
    dbsession.add_all([msg, cb, consent])
    dbsession.commit()
    # Simple assertions – if enums were wrong an exception would have been raised.
    assert conv.channel in {"browser_voice", "browser_text", "telephone"}
    assert msg.role in {"customer", "assistant", "tool"}
    assert cb.status in {"Pending", "Scheduled", "Completed", "Cancelled"}
    assert consent.consent_type in {"contact_storage", "callback_contact", "marketing_communications"}
    assert lead.qualification_status in {"Hot", "Warm", "Cold", "Unqualified"}
    assert lead.pipeline_status in {"New", "Contacted", "Visit Scheduled", "Negotiation", "Converted", "Lost"}



