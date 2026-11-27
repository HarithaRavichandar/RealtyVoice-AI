# -*- coding: utf-8 -*-
"""Consent revocation and callback lifecycle (task 3.5)."""

from datetime import date

import pytest

from backend.models import CallbackRequest, Conversation
from backend.orchestrator.services.tool_executor import execute_tool
from backend.services.callbacks import (
    cancel_open_callbacks,
    create_or_reschedule,
    open_callbacks,
    update_callback_status,
)


@pytest.fixture
def convo(dbsession):
    c = Conversation(channel="browser_text", external_session_id="cb1", temp_requirements=None)
    dbsession.add(c)
    dbsession.flush()
    execute_tool(dbsession, "register_lead", {"phone": "9876543210"}, conversation_id=c.conversation_id)
    execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": True},
                 conversation_id=c.conversation_id)
    return c


def callback(session, convo, day="2026-12-05", time="10:00", contact="phone", reason="Site visit"):
    return execute_tool(
        session, "create_callback_request",
        {"preferred_contact": contact, "preferred_date": day, "preferred_time": time, "reason": reason},
        conversation_id=convo.conversation_id,
    )


def count(session):
    return session.query(CallbackRequest).count()


def test_first_callback_is_created_pending(dbsession, convo):
    r = callback(dbsession, convo)
    assert r["outcome"] == "created"
    assert r["callback"]["status"] == "Pending"
    assert count(dbsession) == 1


def test_identical_request_is_a_duplicate_and_adds_nothing(dbsession, convo):
    first = callback(dbsession, convo)
    again = callback(dbsession, convo)
    assert again["outcome"] == "duplicate"
    assert again["callback"]["callback_id"] == first["callback"]["callback_id"]
    assert count(dbsession) == 1


def test_different_slot_reschedules_the_open_callback(dbsession, convo):
    first = callback(dbsession, convo)
    moved = callback(dbsession, convo, day="2026-12-09", time="16:30", reason="Evening is better")
    assert moved["outcome"] == "rescheduled"
    assert moved["callback"]["callback_id"] == first["callback"]["callback_id"]
    assert (moved["callback"]["preferred_date"], moved["callback"]["preferred_time"]) == ("2026-12-09", "16:30")
    assert moved["callback"]["reason"] == "Evening is better"
    assert count(dbsession) == 1


def test_rescheduling_a_scheduled_callback_returns_it_to_pending(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    update_callback_status(dbsession, cb_id, "Scheduled")
    moved = callback(dbsession, convo, time="11:00")
    assert moved["callback"]["status"] == "Pending"


def test_a_new_callback_is_allowed_after_the_previous_one_is_closed(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    update_callback_status(dbsession, cb_id, "Completed")
    r = callback(dbsession, convo)
    assert r["outcome"] == "created"
    assert r["callback"]["callback_id"] != cb_id
    assert count(dbsession) == 2


def test_revoking_callback_consent_cancels_open_callbacks(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]

    r = execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": False},
                     conversation_id=convo.conversation_id)

    assert r["cancelled_callback_ids"] == [cb_id]
    assert dbsession.get(CallbackRequest, cb_id).status == "Cancelled"
    assert open_callbacks(dbsession, convo.lead_id) == []


def test_after_revocation_a_new_callback_needs_fresh_consent(dbsession, convo):
    callback(dbsession, convo)
    execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": False},
                 conversation_id=convo.conversation_id)

    assert callback(dbsession, convo)["error"] == "callback_consent_required"

    execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": True},
                 conversation_id=convo.conversation_id)
    assert callback(dbsession, convo)["outcome"] == "created"


def test_revoking_other_consents_does_not_touch_callbacks(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    r = execute_tool(dbsession, "record_consent", {"consent_type": "marketing_communications", "granted": False},
                     conversation_id=convo.conversation_id)
    assert "cancelled_callback_ids" not in r
    assert dbsession.get(CallbackRequest, cb_id).status == "Pending"


def test_granting_consent_cancels_nothing(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    r = execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": True},
                     conversation_id=convo.conversation_id)
    assert "cancelled_callback_ids" not in r
    assert dbsession.get(CallbackRequest, cb_id).status == "Pending"


def test_revocation_with_no_open_callback_reports_an_empty_list(dbsession, convo):
    r = execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": False},
                     conversation_id=convo.conversation_id)
    assert r["cancelled_callback_ids"] == []


def test_callback_requesting_intent_follows_the_open_callback(dbsession, convo):
    from backend.models import Lead

    lead = dbsession.get(Lead, convo.lead_id)
    callback(dbsession, convo)
    assert lead.buyer_intent == "callback_requesting"
    assert lead.score_breakdown["callback_request"]["points"] == 15

    execute_tool(dbsession, "record_consent", {"consent_type": "callback_contact", "granted": False},
                 conversation_id=convo.conversation_id)
    assert lead.buyer_intent != "callback_requesting"
    assert lead.score_breakdown["callback_request"]["points"] == 0


@pytest.mark.parametrize(
    "start, target, ok",
    [
        ("Pending", "Scheduled", True), ("Pending", "Cancelled", True), ("Pending", "Completed", True),
        ("Scheduled", "Completed", True), ("Scheduled", "Cancelled", True), ("Scheduled", "Pending", True),
        ("Completed", "Pending", False), ("Completed", "Cancelled", False),
        ("Cancelled", "Scheduled", False), ("Cancelled", "Pending", False),
    ],
)
def test_status_transitions(dbsession, convo, start, target, ok):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    dbsession.get(CallbackRequest, cb_id).status = start
    if ok:
        assert update_callback_status(dbsession, cb_id, target).status == target
    else:
        with pytest.raises(ValueError):
            update_callback_status(dbsession, cb_id, target)
        assert dbsession.get(CallbackRequest, cb_id).status == start


def test_status_update_errors(dbsession, convo):
    cb_id = callback(dbsession, convo)["callback"]["callback_id"]
    with pytest.raises(LookupError):
        update_callback_status(dbsession, 9999, "Scheduled")
    with pytest.raises(ValueError):
        update_callback_status(dbsession, cb_id, "Paused")


def test_service_level_helpers(dbsession, convo):
    cb, outcome = create_or_reschedule(
        dbsession, lead_id=convo.lead_id, conversation_id=convo.conversation_id,
        preferred_contact="phone", preferred_date=date(2026, 12, 5), preferred_time="10:00", reason="x",
    )
    assert outcome == "created"
    assert cancel_open_callbacks(dbsession, convo.lead_id) == [cb.callback_id]
    assert cancel_open_callbacks(dbsession, convo.lead_id) == []
