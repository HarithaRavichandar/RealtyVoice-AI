# -*- coding: utf-8 -*-
"""
End-to-end tests for the Lead → Consent → Callback flow.

Tests the complete callback workflow through the tool executor:
1. register_lead — creates/updates a Lead and links to conversation
2. record_consent — records callback_contact consent
3. create_callback_request — creates the callback (existing tool)

Also tests error paths: missing conversation, missing lead, missing consent,
validation failures, and idempotent lead updates.
"""

import pytest
from datetime import datetime, date

from backend.models import Conversation, Lead, ConsentRecord, CallbackRequest
from backend.orchestrator.services.tool_executor import execute_tool


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def conversation(dbsession):
    """An anonymous conversation with no lead."""
    conv = Conversation(
        lead_id=None,
        channel="browser_text",
        external_session_id="test-session-001",
    )
    dbsession.add(conv)
    dbsession.flush()
    return conv


# ---------------------------------------------------------------
# register_lead tests
# ---------------------------------------------------------------

class TestRegisterLead:

    def test_creates_lead_and_links_to_conversation(self, dbsession, conversation):
        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Priya Sharma",
            "phone": "+91-9876543210",
            "email": "priya@example.com",
        }, conversation_id=conversation.conversation_id)

        assert "error" not in result
        assert result["tool"] == "register_lead"
        assert result["updated"] is False
        assert result["lead"]["customer_name"] == "Priya Sharma"
        assert result["lead"]["phone"] == "+91-9876543210"
        assert result["lead"]["email"] == "priya@example.com"
        assert result["lead"]["pipeline_status"] == "New"

        # Verify the conversation is now linked.
        dbsession.refresh(conversation)
        assert conversation.lead_id == result["lead"]["lead_id"]

    def test_creates_lead_with_name_only(self, dbsession, conversation):
        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Rahul",
        }, conversation_id=conversation.conversation_id)

        assert "error" not in result
        assert result["lead"]["customer_name"] == "Rahul"
        assert result["lead"]["phone"] is None
        assert result["lead"]["email"] is None

    def test_creates_lead_with_phone_only(self, dbsession, conversation):
        result = execute_tool(dbsession, "register_lead", {
            "phone": "+91-1234567890",
        }, conversation_id=conversation.conversation_id)

        assert "error" not in result
        assert result["lead"]["phone"] == "+91-1234567890"

    def test_rejects_empty_contact(self, dbsession, conversation):
        result = execute_tool(dbsession, "register_lead", {},
                              conversation_id=conversation.conversation_id)

        assert result["error"] == "validation"

    def test_rejects_whitespace_only_name(self, dbsession, conversation):
        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "   ",
        }, conversation_id=conversation.conversation_id)

        assert result["error"] == "validation"

    def test_updates_existing_lead(self, dbsession, conversation):
        # First registration.
        result1 = execute_tool(dbsession, "register_lead", {
            "customer_name": "Priya",
            "phone": "+91-1111111111",
        }, conversation_id=conversation.conversation_id)

        assert result1["updated"] is False
        lead_id = result1["lead"]["lead_id"]

        # Second call updates the existing lead.
        result2 = execute_tool(dbsession, "register_lead", {
            "email": "priya@example.com",
        }, conversation_id=conversation.conversation_id)

        assert result2["updated"] is True
        assert result2["lead"]["lead_id"] == lead_id
        assert result2["lead"]["customer_name"] == "Priya"  # preserved
        assert result2["lead"]["phone"] == "+91-1111111111"  # preserved
        assert result2["lead"]["email"] == "priya@example.com"  # added

    def test_copies_temp_requirements_to_lead(self, dbsession, conversation):
        # Set some requirements on the conversation first.
        conversation.temp_requirements = {
            "location": "Mumbai",
            "budget_max": "5000000",
            "property_type": "apartment",
            "configuration": "2 BHK",
        }
        dbsession.flush()

        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Test User",
        }, conversation_id=conversation.conversation_id)

        lead = dbsession.get(Lead, result["lead"]["lead_id"])
        assert lead.preferred_location == "Mumbai"
        assert lead.property_type == "apartment"
        assert lead.configuration == "2 BHK"

    def test_missing_conversation_id(self, dbsession):
        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Test",
        }, conversation_id=None)

        assert result["error"] == "missing_conversation_id"

    def test_conversation_not_found(self, dbsession):
        result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Test",
        }, conversation_id=99999)

        assert result["error"] == "conversation_not_found"


# ---------------------------------------------------------------
# record_consent tests
# ---------------------------------------------------------------

class TestRecordConsent:

    @pytest.fixture
    def conversation_with_lead(self, dbsession, conversation):
        """A conversation with a registered lead."""
        lead = Lead(
            customer_name="Test Lead",
            phone="+91-9999999999",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()
        conversation.lead_id = lead.lead_id
        dbsession.flush()
        return conversation, lead

    def test_records_granted_consent(self, dbsession, conversation_with_lead):
        conv, lead = conversation_with_lead

        result = execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": True,
        }, conversation_id=conv.conversation_id)

        assert "error" not in result
        assert result["tool"] == "record_consent"
        assert result["consent"]["granted"] is True
        assert result["consent"]["consent_type"] == "callback_contact"
        assert result["consent"]["lead_id"] == lead.lead_id

    def test_records_refused_consent(self, dbsession, conversation_with_lead):
        conv, _ = conversation_with_lead

        result = execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": False,
        }, conversation_id=conv.conversation_id)

        assert "error" not in result
        assert result["consent"]["granted"] is False

    def test_requires_lead(self, dbsession, conversation):
        """Consent cannot be recorded without a lead."""
        result = execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": True,
        }, conversation_id=conversation.conversation_id)

        assert result["error"] == "lead_required"

    def test_rejects_invalid_consent_type(self, dbsession, conversation_with_lead):
        conv, _ = conversation_with_lead

        result = execute_tool(dbsession, "record_consent", {
            "consent_type": "invalid_type",
            "granted": True,
        }, conversation_id=conv.conversation_id)

        assert result["error"] == "validation"

    def test_missing_conversation_id(self, dbsession):
        result = execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": True,
        }, conversation_id=None)

        assert result["error"] == "missing_conversation_id"


# ---------------------------------------------------------------
# Full end-to-end flow tests
# ---------------------------------------------------------------

class TestFullCallbackFlow:
    """Test the complete: register_lead → record_consent → create_callback_request."""

    def test_complete_callback_flow(self, dbsession, conversation):
        conv_id = conversation.conversation_id

        # Step 1: Register lead.
        lead_result = execute_tool(dbsession, "register_lead", {
            "customer_name": "Ananya Patel",
            "phone": "+91-8888888888",
        }, conversation_id=conv_id)

        assert "error" not in lead_result
        lead_id = lead_result["lead"]["lead_id"]

        # Step 2: Record consent.
        consent_result = execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": True,
        }, conversation_id=conv_id)

        assert "error" not in consent_result

        # Step 3: Create callback.
        callback_result = execute_tool(dbsession, "create_callback_request", {
            "preferred_contact": "phone",
            "preferred_date": "2026-12-01",
            "preferred_time": "14:00",
            "reason": "Interested in 3 BHK apartments in Whitefield",
        }, conversation_id=conv_id)

        assert "error" not in callback_result
        assert callback_result["tool"] == "create_callback_request"
        assert callback_result["callback"]["lead_id"] == lead_id
        assert callback_result["callback"]["status"] == "Pending"

    def test_callback_fails_without_lead(self, dbsession, conversation):
        """Attempting callback without registering a lead first."""
        result = execute_tool(dbsession, "create_callback_request", {
            "preferred_contact": "phone",
            "preferred_date": "2026-12-01",
            "preferred_time": "14:00",
            "reason": "Test",
        }, conversation_id=conversation.conversation_id)

        assert result["error"] == "lead_required"

    def test_callback_fails_without_consent(self, dbsession, conversation):
        """Lead registered but no consent recorded."""
        conv_id = conversation.conversation_id

        # Register lead.
        execute_tool(dbsession, "register_lead", {
            "customer_name": "Test User",
            "phone": "+91-7777777777",
        }, conversation_id=conv_id)

        # Attempt callback without consent.
        result = execute_tool(dbsession, "create_callback_request", {
            "preferred_contact": "phone",
            "preferred_date": "2026-12-01",
            "preferred_time": "14:00",
            "reason": "Test",
        }, conversation_id=conv_id)

        assert result["error"] == "callback_consent_required"

    def test_callback_fails_with_refused_consent(self, dbsession, conversation):
        """Lead registered, consent explicitly refused."""
        conv_id = conversation.conversation_id

        execute_tool(dbsession, "register_lead", {
            "customer_name": "Test User",
            "phone": "+91-6666666666",
        }, conversation_id=conv_id)

        execute_tool(dbsession, "record_consent", {
            "consent_type": "callback_contact",
            "granted": False,
        }, conversation_id=conv_id)

        result = execute_tool(dbsession, "create_callback_request", {
            "preferred_contact": "phone",
            "preferred_date": "2026-12-01",
            "preferred_time": "14:00",
            "reason": "Test",
        }, conversation_id=conv_id)

        assert result["error"] == "callback_consent_required"
