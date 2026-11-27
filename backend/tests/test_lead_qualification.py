# -*- coding: utf-8 -*-
"""Lead qualification: intent, transparent score, stored fields (tasks 3.1-3.3)."""

import pytest
from sqlalchemy import create_engine, text

from backend.models import Conversation, Lead
from backend.models.consent import ConsentRecord
from backend.models.property import Property
from backend.orchestrator.services.tool_executor import execute_tool
from backend.orchestrator.tools.schemas import UpdateCustomerRequirementsRequest
from backend.services.lead_qualification import (
    Evidence,
    classify_intent,
    score_lead,
    status_for_score,
)
from backend.services.schema_upgrades import ensure_lead_qualification_columns


# ---------------------------------------------------------------------------
# Values accepted for the three new fields (3.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, raw, expected",
    [
        ("purchase_timeline", "immediate", "immediate"),
        ("purchase_timeline", "ASAP", "immediate"),
        ("purchase_timeline", "within 3 months", "within_3_months"),
        ("purchase_timeline", "within-6-months", "within_6_months"),
        ("purchase_timeline", "this year", "within_12_months"),
        ("purchase_timeline", "just looking", "exploring"),
        ("visit_readiness", "ready to visit", "ready_to_visit"),
        ("visit_readiness", "maybe", "considering"),
        ("visit_readiness", "not now", "not_ready"),
        ("contact_preference", "WhatsApp", "whatsapp"),
        ("contact_preference", "call", "phone_call"),
        ("contact_preference", "e-mail", "email"),
    ],
)
def test_qualification_values_are_normalised(field, raw, expected):
    assert getattr(UpdateCustomerRequirementsRequest(**{field: raw}), field) == expected


@pytest.mark.parametrize(
    "field, raw",
    [("purchase_timeline", "someday maybe tuesday"), ("visit_readiness", "banana"), ("contact_preference", "carrier pigeon"), ("purchase_timeline", "  ")],
)
def test_unknown_qualification_values_are_rejected(field, raw):
    with pytest.raises(ValueError):
        UpdateCustomerRequirementsRequest(**{field: raw})


def test_null_is_accepted_to_clear_a_qualification_field():
    assert UpdateCustomerRequirementsRequest(purchase_timeline=None).purchase_timeline is None


# ---------------------------------------------------------------------------
# Intent (3.2)
# ---------------------------------------------------------------------------

def test_intent_stages():
    assert classify_intent(Evidence()) == "browsing"
    assert classify_intent(Evidence(location="Pune")) == "exploring"
    assert classify_intent(Evidence(has_budget_max=True)) == "exploring"
    assert classify_intent(Evidence(location="Pune", has_budget_max=True)) == "exploring"  # no size/type yet
    searching = Evidence(location="Pune", has_budget_max=True, has_configuration=True)
    assert classify_intent(searching) == "actively_searching"
    assert classify_intent(Evidence(location="Pune", has_budget_min=True, has_property_type=True)) == "actively_searching"


def test_visit_ready_needs_readiness_and_a_way_to_reach_them():
    assert classify_intent(Evidence(visit_readiness="ready_to_visit")) == "browsing"
    assert classify_intent(Evidence(visit_readiness="considering", has_phone=True)) == "browsing"
    assert classify_intent(Evidence(visit_readiness="ready_to_visit", has_phone=True)) == "visit_ready"
    assert classify_intent(Evidence(visit_readiness="ready_to_visit", has_email=True)) == "visit_ready"


def test_open_callback_outranks_every_other_stage():
    e = Evidence(location="Pune", has_budget_max=True, has_configuration=True,
                 visit_readiness="ready_to_visit", has_phone=True, open_callback=True)
    assert classify_intent(e) == "callback_requesting"


# ---------------------------------------------------------------------------
# Score (3.3)
# ---------------------------------------------------------------------------

def test_empty_evidence_scores_zero_and_is_unqualified():
    r = score_lead(Evidence())
    assert r["score"] == 0 and r["status"] == "Unqualified"
    assert set(r["signals"]) == {
        "budget_clarity", "location_specificity", "purchase_timeline",
        "contact_completeness", "callback_request",
    }
    assert all(s["reason"] for s in r["signals"].values())


def test_maximum_evidence_scores_one_hundred():
    e = Evidence(has_budget_min=True, has_budget_max=True, location="Whitefield, Bangalore",
                 location_labels=1, purchase_timeline="immediate",
                 has_name=True, has_phone=True, has_email=True, open_callback=True)
    r = score_lead(e)
    assert r["score"] == 100 and r["status"] == "Hot"
    assert sum(s["max"] for s in r["signals"].values()) == 100


@pytest.mark.parametrize(
    "evidence, points",
    [
        (Evidence(has_budget_min=True, has_budget_max=True), 25),
        (Evidence(has_budget_max=True), 15),
        (Evidence(has_budget_min=True), 15),
    ],
)
def test_budget_clarity_points(evidence, points):
    assert score_lead(evidence)["signals"]["budget_clarity"]["points"] == points


@pytest.mark.parametrize(
    "location, labels, points",
    [(None, None, 0), ("Whitefield", 1, 20), ("Bangalore", 6, 12), ("Atlantis", 0, 6), ("Atlantis", None, 6)],
)
def test_location_specificity_points(location, labels, points):
    r = score_lead(Evidence(location=location, location_labels=labels))
    assert r["signals"]["location_specificity"]["points"] == points


@pytest.mark.parametrize(
    "timeline, points",
    [("immediate", 20), ("within_3_months", 16), ("within_6_months", 12),
     ("within_12_months", 8), ("later", 4), ("exploring", 0), (None, 0)],
)
def test_timeline_points(timeline, points):
    assert score_lead(Evidence(purchase_timeline=timeline))["signals"]["purchase_timeline"]["points"] == points


def test_contact_points_add_up():
    assert score_lead(Evidence(has_phone=True))["signals"]["contact_completeness"]["points"] == 10
    assert score_lead(Evidence(has_phone=True, has_email=True, has_name=True))["signals"]["contact_completeness"]["points"] == 20
    assert score_lead(Evidence(has_name=True))["signals"]["contact_completeness"]["points"] == 5


@pytest.mark.parametrize(
    "score, status",
    [(0, "Unqualified"), (14, "Unqualified"), (15, "Cold"), (39, "Cold"),
     (40, "Warm"), (69, "Warm"), (70, "Hot"), (100, "Hot")],
)
def test_status_thresholds(score, status):
    assert status_for_score(score) == status


# ---------------------------------------------------------------------------
# Stored on the lead through the real tools
# ---------------------------------------------------------------------------

@pytest.fixture
def convo(dbsession):
    dbsession.add(Property(location="bangalore", property_type="apartment", configuration="2 BHK"))
    c = Conversation(channel="browser_text", external_session_id="q1", temp_requirements=None)
    dbsession.add(c)
    dbsession.flush()
    return c


def run(session, convo, tool_name, **payload):
    return execute_tool(session, tool_name, payload, conversation_id=convo.conversation_id)


def lead_of(session, convo):
    return session.get(Lead, convo.lead_id)


def test_requirements_saved_before_registration_are_copied_to_the_lead(dbsession, convo):
    run(dbsession, convo, "update_customer_requirements",
        location="Bengaluru", budget_max="8000000", configuration="2 BHK",
        purchase_timeline="within 3 months", visit_readiness="maybe", contact_preference="whatsapp")
    run(dbsession, convo, "register_lead", customer_name="Asha", phone="9876543210")

    lead = lead_of(dbsession, convo)
    assert (lead.purchase_timeline, lead.visit_readiness, lead.contact_preference) == (
        "within_3_months", "considering", "whatsapp")
    assert lead.buyer_intent == "actively_searching"
    # budget one bound 15 + one data label 20 + 3 months 16 + phone 10 + name 5
    assert float(lead.qualification_score) == 66
    assert lead.qualification_status == "Warm"
    assert lead.score_breakdown["budget_clarity"]["points"] == 15


def test_later_updates_refresh_the_stored_score(dbsession, convo):
    run(dbsession, convo, "register_lead", phone="9876543210")
    lead = lead_of(dbsession, convo)
    assert lead.buyer_intent == "browsing"
    assert float(lead.qualification_score) == 10

    result = run(dbsession, convo, "update_customer_requirements", budget_min="5000000", budget_max="8000000")
    assert result["lead_qualification"]["score"] == 35
    assert lead.qualification_status == "Cold"

    run(dbsession, convo, "update_customer_requirements", purchase_timeline="immediate", visit_readiness="ready_to_visit")
    assert lead.buyer_intent == "visit_ready"
    assert float(lead.qualification_score) == 55
    assert lead.qualification_status == "Warm"


def test_clearing_a_field_removes_its_points(dbsession, convo):
    run(dbsession, convo, "register_lead", phone="9876543210")
    run(dbsession, convo, "update_customer_requirements", purchase_timeline="immediate")
    assert lead_of(dbsession, convo).score_breakdown["purchase_timeline"]["points"] == 20
    run(dbsession, convo, "update_customer_requirements", purchase_timeline=None)
    lead = lead_of(dbsession, convo)
    assert lead.purchase_timeline is None
    assert lead.score_breakdown["purchase_timeline"]["points"] == 0


def test_no_qualification_in_the_response_before_a_lead_exists(dbsession, convo):
    result = run(dbsession, convo, "update_customer_requirements", location="Pune")
    assert "lead_qualification" not in result


def test_revoked_contact_storage_consent_removes_contact_points(dbsession, convo):
    run(dbsession, convo, "register_lead", customer_name="Asha", phone="9876543210", email="a@example.com")
    lead = lead_of(dbsession, convo)
    assert lead.score_breakdown["contact_completeness"]["points"] == 20

    result = run(dbsession, convo, "record_consent", consent_type="contact_storage", granted=False)

    assert result["contact_storage_revoked"] is True
    assert lead.score_breakdown["contact_completeness"]["points"] == 0
    assert lead.phone == "9876543210"  # nothing deleted, only no longer counted

    run(dbsession, convo, "record_consent", consent_type="contact_storage", granted=True)
    assert lead.score_breakdown["contact_completeness"]["points"] == 20


# ---------------------------------------------------------------------------
# Schema upgrade for existing databases
# ---------------------------------------------------------------------------

def test_schema_upgrade_adds_missing_columns_once_and_keeps_data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE leads (lead_id INTEGER PRIMARY KEY, customer_name VARCHAR, "
            "qualification_status VARCHAR NOT NULL, pipeline_status VARCHAR NOT NULL)"
        ))
        conn.execute(text("INSERT INTO leads VALUES (1, 'Asha', 'Cold', 'New')"))

    added = ensure_lead_qualification_columns(engine)
    assert set(added) == {"purchase_timeline", "visit_readiness", "contact_preference", "buyer_intent", "score_breakdown"}
    assert ensure_lead_qualification_columns(engine) == []  # idempotent

    with engine.connect() as conn:
        row = conn.execute(text("SELECT customer_name, buyer_intent FROM leads")).one()
    assert tuple(row) == ("Asha", None)


def test_schema_upgrade_ignores_a_database_without_leads_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    assert ensure_lead_qualification_columns(engine) == []


@pytest.mark.parametrize("fragment", [
    "purchase_timeline", "Never guess these or tell the customer their score",
    "call recommend_properties", 'outcome "duplicate"', "cancelled_callback_ids", "contact_storage consent",
])
def test_prompt_covers_phase3_behaviour(fragment):
    from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
    assert fragment in SYSTEM_PROMPT


def test_provider_tool_definitions_include_the_new_tool_and_fields():
    from backend.orchestrator.services.provider_tools import get_provider_tool_definitions
    defs = {d.name: d for d in get_provider_tool_definitions()}
    assert "recommend_properties" in defs
    props = defs["update_customer_requirements"].parameters["properties"]
    assert {"purchase_timeline", "visit_readiness", "contact_preference"} <= set(props)
