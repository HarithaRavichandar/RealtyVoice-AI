# -*- coding: utf-8 -*-
"""Multi-turn conversation tests.

These drive the real ``POST /api/agent/turn`` endpoint, the real agent loop,
the real tool executor and an in-memory database. Only the LLM is scripted,
so each test describes what the model decides and checks what the system does
with it: saved requirements, fallbacks, explicit overrides, suggestions, error
guidance and the audit trail.

Seeded property rows are throwaway test data in an in-memory database.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from itertools import count
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app
from backend.models import Conversation, Message
from backend.models.property import Property
from backend.orchestrator.providers.base import (
    MessageRole,
    ProviderResponse,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------

_call_ids = count(1)


def tool(name, arguments):
    """A scripted model response that requests one tool call."""
    return ProviderResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call-{next(_call_ids)}", function_name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def say(text):
    """A scripted model response with a final reply."""
    return ProviderResponse(content=text)


class ScriptedProvider:
    """Plays back scripted responses and records everything the model saw."""

    def __init__(self):
        self.queue = []
        self.calls = []

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.queue:
            raise AssertionError("The agent asked the model for more responses than were scripted")
        return self.queue.pop(0)

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chat(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    provider = ScriptedProvider()
    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.create_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.AGENT_MAX_TOOL_ITERATIONS", 4
    )

    def prop(location, configuration, low, high, status=None):
        return Property(
            location=location,
            property_type="apartment",
            configuration=configuration,
            price_min=low,
            price_max=high,
            listing_price=(low + high) // 2,
            possession_date=datetime(2026, 6, 1),
            availability_status=status,
        )

    with factory() as db:
        rows = {
            "A": prop("bangalore", "2 BHK", 6_000_000, 7_000_000, "Ready to Move"),
            "B": prop("bangalore", "3 BHK", 9_000_000, 11_000_000),
            "C": prop("mumbai", "2 BHK", 7_000_000, 7_500_000),
            "D": prop("bangalore", "2 BHK", 9_500_000, 10_000_000),
            "E": prop("bangalore", "1 BHK", 2_000_000, 3_000_000),
        }
        db.add_all(rows.values())
        db.commit()
        ids = {key: row.property_id for key, row in rows.items()}

    class Chat:
        def __init__(self):
            self.client = client
            self.llm = provider
            self.ids = ids
            self.session_factory = factory
            self.last_calls = []

        def new_session(self):
            response = client.post("/api/conversations", json={"channel": "browser_text"})
            assert response.status_code == 201
            return response.json()["session_id"]

        def turn(self, sid, message, *script):
            """Send one customer message; the model plays the scripted responses."""
            start = len(provider.calls)
            provider.queue.extend(script)
            response = client.post("/api/agent/turn", json={"session_id": sid, "message": message})
            assert response.status_code == 200, response.text
            assert provider.queue == [], "Not every scripted model response was used"
            self.last_calls = provider.calls[start:]
            return response.json()

        def tool_payloads(self, *, key):
            """Parsed JSON of tool messages in the last turn containing ``key``."""
            found = []
            for call in self.last_calls:
                for m in call["messages"]:
                    if m.role == MessageRole.TOOL and key in (m.content or ""):
                        payload = json.loads(m.content)
                        if payload not in found:
                            found.append(payload)
            return found

    with TestClient(app) as client:
        try:
            yield Chat()
        finally:
            app.dependency_overrides.pop(get_db, None)
            Base.metadata.drop_all(engine)
            engine.dispose()


def _ids(response):
    return {p["property_id"] for p in response["matched_properties"]}


def _setup_two_bhk_in_bengaluru(chat):
    """Turns 1-2 of the main scenario; returns the session id."""
    sid = chat.new_session()
    chat.turn(
        sid,
        "I want a 2 BHK flat under 1 crore",
        tool("update_customer_requirements", {"configuration": "2 BHK", "budget_max": "10000000"}),
        say("Noted. Which city are you looking in?"),
    )
    chat.turn(
        sid,
        "In Bengaluru",
        tool("update_customer_requirements", {"location": "Bengaluru"}),
        tool("search_properties", {}),
        say("Here are two options in Bengaluru."),
    )
    return sid


# ---------------------------------------------------------------------------
# Requirements accumulate and are applied across turns
# ---------------------------------------------------------------------------

def test_requirements_accumulate_and_are_applied_across_turns(chat):
    sid = chat.new_session()

    r1 = chat.turn(
        sid,
        "I want a 2 BHK flat under 1 crore",
        tool("update_customer_requirements", {"configuration": "2 BHK", "budget_max": "10000000"}),
        say("Noted. Which city are you looking in?"),
    )
    assert r1["requirements"]["configuration"] == "2 BHK"
    assert Decimal(str(r1["requirements"]["budget_max"])) == 10_000_000

    r2 = chat.turn(
        sid,
        "In Bengaluru",
        tool("update_customer_requirements", {"location": "Bengaluru"}),
        tool("search_properties", {}),  # the model sends no arguments at all
        say("Here are two options in Bengaluru."),
    )

    # The search used everything saved in earlier turns.
    assert r2["intent"] == "property_search"
    assert _ids(r2) == {chat.ids["A"], chat.ids["D"]}
    assert r2["next_action"]["action"] == "show_properties"

    searched = chat.tool_payloads(key="applied_defaults")[0]
    assert set(searched["applied_defaults"]) == {"configuration", "max_budget", "location"}
    assert searched["applied_defaults"]["location"] == "Bengaluru"


def test_model_sees_saved_requirements_and_history_on_the_next_turn(chat):
    sid = _setup_two_bhk_in_bengaluru(chat)

    chat.turn(sid, "Thanks", say("You're welcome!"))
    first_call = chat.last_calls[0]["messages"]

    system = first_call[0]
    assert system.role == MessageRole.SYSTEM
    assert "Current customer requirements" in system.content
    for expected in ('"configuration": "2 BHK"', '"location": "Bengaluru"'):
        assert expected in system.content

    history = [(m.role, m.content) for m in first_call[1:]]
    assert (MessageRole.USER, "I want a 2 BHK flat under 1 crore") in history
    assert (MessageRole.ASSISTANT, "Noted. Which city are you looking in?") in history
    assert history[-1] == (MessageRole.USER, "Thanks")


# ---------------------------------------------------------------------------
# A changed preference overrides the saved one
# ---------------------------------------------------------------------------

def test_changed_requirement_overrides_the_saved_value(chat):
    sid = _setup_two_bhk_in_bengaluru(chat)

    r3 = chat.turn(
        sid,
        "Actually show me 3 BHK",
        tool("update_customer_requirements", {"configuration": "3 BHK"}),
        tool("search_properties", {"configuration": "3 BHK"}),
        say("Here is a 3 BHK."),
    )

    assert _ids(r3) == {chat.ids["B"]}
    assert r3["requirements"]["configuration"] == "3 BHK"

    searched = chat.tool_payloads(key="applied_defaults")[0]
    assert "configuration" not in searched["applied_defaults"]  # came from the model
    assert set(searched["applied_defaults"]) == {"max_budget", "location"}


def test_explicit_null_drops_a_saved_filter_for_one_search_only(chat):
    sid = _setup_two_bhk_in_bengaluru(chat)

    r3 = chat.turn(
        sid,
        "Ignore my budget and size, show everything in Bengaluru",
        tool("search_properties", {"max_budget": None, "configuration": None}),
        say("Here is everything in Bengaluru."),
    )

    assert _ids(r3) == {chat.ids[k] for k in "ABDE"}
    searched = chat.tool_payloads(key="applied_defaults")[0]
    assert searched["applied_defaults"] == {"location": "Bengaluru"}
    # The saved requirements themselves are untouched.
    assert r3["requirements"]["configuration"] == "2 BHK"
    assert Decimal(str(r3["requirements"]["budget_max"])) == 10_000_000


def test_requirements_do_not_leak_between_conversations(chat):
    _setup_two_bhk_in_bengaluru(chat)

    other = chat.new_session()
    r = chat.turn(other, "Show me anything", tool("search_properties", {}), say("Here you go."))

    assert _ids(r) == set(chat.ids.values())
    assert chat.tool_payloads(key="applied_defaults")[0]["applied_defaults"] == {}


# ---------------------------------------------------------------------------
# Zero results: suggestions reach the model and it can recover
# ---------------------------------------------------------------------------

def test_unavailable_locality_suggests_the_city_and_the_agent_recovers(chat):
    sid = chat.new_session()

    r = chat.turn(
        sid,
        "Flats in Indiranagar, Bengaluru",
        tool("update_customer_requirements", {"location": "Indiranagar, Bengaluru"}),
        tool("search_properties", {}),
        tool("search_properties", {"location": "Bengaluru"}),  # model widens to the city
        say("I have no records for Indiranagar itself, so I searched all of Bangalore."),
    )

    first, second = chat.tool_payloads(key="applied_defaults")
    assert first["total"] == 0
    suggestion = first["location_suggestions"][0]
    assert suggestion["type"] == "parent_city"
    assert suggestion["location"] == "Bangalore"
    assert "does not confirm" in suggestion["note"]

    # The widened search used the model's explicit location, not the saved one.
    assert "location" not in second["applied_defaults"]
    assert _ids(r) == {chat.ids[k] for k in "ABDE"}
    assert r["next_action"]["action"] == "show_properties"


def test_unknown_place_gets_no_invented_suggestions(chat):
    sid = chat.new_session()

    r = chat.turn(
        sid,
        "Anything in Atlantis?",
        tool("search_properties", {"location": "Atlantis"}),
        say("I have no records for that place."),
    )

    payload = chat.tool_payloads(key="applied_defaults")[0]
    assert payload["total"] == 0
    assert "location_suggestions" not in payload
    assert r["matched_properties"] == []
    assert r["next_action"]["action"] == "ask_clarification"


# ---------------------------------------------------------------------------
# Both budget bounds across turns, clearing, and conflicting budgets
# ---------------------------------------------------------------------------

def test_both_budget_bounds_accumulate_across_turns(chat):
    sid = chat.new_session()

    chat.turn(
        sid,
        "My budget starts at 60 lakh",
        tool("update_customer_requirements", {"budget_min": "6000000"}),
        say("Noted. And your upper limit?"),
    )
    r2 = chat.turn(
        sid,
        "Up to 80 lakh, in Bengaluru",
        tool("update_customer_requirements", {"budget_max": "8000000", "location": "Bengaluru"}),
        say("Shall I search?"),
    )
    assert Decimal(str(r2["requirements"]["budget_min"])) == 6_000_000
    assert Decimal(str(r2["requirements"]["budget_max"])) == 8_000_000

    r3 = chat.turn(sid, "Yes please", tool("search_properties", {}), say("One match."))

    # The minimum removes the cheap 1 BHK; the maximum removes the dearer flats.
    assert _ids(r3) == {chat.ids["A"]}
    applied = chat.tool_payloads(key="applied_defaults")[0]["applied_defaults"]
    assert set(applied) == {"min_budget", "max_budget", "location"}
    assert Decimal(str(applied["min_budget"])) == 6_000_000
    assert Decimal(str(applied["max_budget"])) == 8_000_000


def test_withdrawn_requirement_is_cleared_for_good(chat):
    sid = chat.new_session()

    r1 = chat.turn(
        sid,
        "Bengaluru, budget up to 40 lakh",
        tool("update_customer_requirements", {"location": "Bengaluru", "budget_max": "4000000"}),
        tool("search_properties", {}),
        say("One cheap option."),
    )
    assert _ids(r1) == {chat.ids["E"]}

    r2 = chat.turn(
        sid,
        "Forget the budget limit",
        tool("update_customer_requirements", {"budget_max": None}),
        tool("search_properties", {}),
        say("Here is everything in Bengaluru."),
    )
    assert r2["requirements"]["budget_max"] is None
    assert r2["requirements"]["location"] == "Bengaluru"
    assert _ids(r2) == {chat.ids[k] for k in "ABDE"}
    assert set(chat.tool_payloads(key="applied_defaults")[0]["applied_defaults"]) == {"location"}

    # The clearing persists: a later search does not bring the budget back.
    r3 = chat.turn(sid, "Search again", tool("search_properties", {}), say("Same list."))
    assert _ids(r3) == {chat.ids[k] for k in "ABDE"}
    assert set(chat.tool_payloads(key="applied_defaults")[0]["applied_defaults"]) == {"location"}


def test_conflicting_budget_is_rejected_and_the_conversation_stays_healthy(chat):
    sid = chat.new_session()

    chat.turn(
        sid,
        "My budget starts at 90 lakh",
        tool("update_customer_requirements", {"budget_min": "9000000"}),
        say("Noted."),
    )

    # A maximum below the saved minimum is rejected, nothing is stored.
    r2 = chat.turn(
        sid,
        "Actually under 50 lakh",
        tool("update_customer_requirements", {"budget_max": "5000000"}),
        say("That conflicts with your 90 lakh minimum. Which should I keep?"),
    )
    assert Decimal(str(r2["requirements"]["budget_min"])) == 9_000_000
    assert r2["requirements"]["budget_max"] is None

    conflict = chat.tool_payloads(key="requirements_conflict")[0]
    assert conflict["error"] == "requirements_conflict"
    assert conflict["saved_budget_min"] == "9000000"
    assert conflict["saved_budget_max"] is None

    after_tool = chat.last_calls[1]["messages"]
    tool_index = next(i for i, m in enumerate(after_tool) if m.role == MessageRole.TOOL)
    hint = after_tool[tool_index + 1]
    assert hint.role == MessageRole.SYSTEM
    assert "Do not claim the update was saved" in hint.content

    # The customer resolves it; the model clears the bound they dropped.
    r3 = chat.turn(
        sid,
        "Keep the 50 lakh maximum",
        tool("update_customer_requirements", {"budget_min": None, "budget_max": "5000000"}),
        tool("search_properties", {"location": "Bengaluru"}),
        say("Here is the cheap option."),
    )
    assert r3["requirements"]["budget_min"] is None
    assert Decimal(str(r3["requirements"]["budget_max"])) == 5_000_000
    assert _ids(r3) == {chat.ids["E"]}

    # The model saw the saved (valid) requirements at the start of that turn,
    # i.e. nothing was silently dropped.
    assert '"budget_min": "9000000"' in chat.last_calls[0]["messages"][0].content


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_availability_phrase_is_normalized_saved_and_flagged(chat):
    sid = chat.new_session()

    r = chat.turn(
        sid,
        "Ready to move in please, in Bengaluru",
        tool("update_customer_requirements", {"availability": "ready to move in", "location": "Bengaluru"}),
        tool("search_properties", {}),
        say("One ready-to-move option, from limited data."),
    )

    assert r["requirements"]["availability"] == "Ready to Move"
    assert _ids(r) == {chat.ids["A"]}

    payload = chat.tool_payloads(key="applied_defaults")[0]
    assert payload["applied_defaults"]["availability"] == "Ready to Move"
    assert "unknown status are excluded" in payload["availability_note"]


# ---------------------------------------------------------------------------
# Tool errors and guidance
# ---------------------------------------------------------------------------

def test_amenity_error_guidance_reaches_the_model(chat, monkeypatch):
    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.execute_nearby_amenities",
        AsyncMock(
            return_value={
                "tool": "nearby_amenities",
                "error": "coordinates_unavailable",
                "message": "Geocode succeeded but no coordinates were returned.",
            }
        ),
    )
    sid = chat.new_session()

    chat.turn(
        sid,
        "What is near Whitefield?",
        tool("nearby_amenities", {"location": "Whitefield"}),
        say("I can't look up amenities for that exact place. Shall I try Bengaluru?"),
    )

    after_tool = chat.last_calls[1]["messages"]
    tool_index = next(i for i, m in enumerate(after_tool) if m.role == MessageRole.TOOL)
    assert "coordinates_unavailable" in after_tool[tool_index].content
    hint = after_tool[tool_index + 1]
    assert hint.role == MessageRole.SYSTEM
    assert "Do not estimate or invent" in hint.content


def test_successful_amenity_answer_is_not_reported_as_an_error(chat, monkeypatch):
    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.execute_nearby_amenities",
        AsyncMock(
            return_value={
                "tool": "nearby_amenities",
                "location": "Whitefield",
                "radius_m": 3000,
                "amenities": {"hospital": []},
            }
        ),
    )
    sid = chat.new_session()

    r = chat.turn(
        sid,
        "What is near Whitefield?",
        tool("nearby_amenities", {"location": "Whitefield"}),
        say("I found no hospitals within 3000 m."),
    )

    assert r["next_action"]["action"] != "error"
    assert r["intent"] == "general_question"


def test_a_failed_amenity_lookup_is_explained_not_reported_as_an_error(chat, monkeypatch):
    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.execute_nearby_amenities",
        AsyncMock(return_value={"tool": "nearby_amenities", "error": "mappls_api_error",
                                "failure_kind": "timeout", "message": "The service took too long."}),
    )
    sid = chat.new_session()
    r = chat.turn(
        sid, "What is near Whitefield?",
        tool("nearby_amenities", {"location": "Whitefield"}),
        say("I can't check nearby places right now. Would you like to see properties instead?"),
    )
    assert r["next_action"]["action"] != "error"
    assert "can't check nearby places" in r["reply"]


def test_an_unrecognised_tool_result_is_still_reported_as_an_error(chat):
    sid = chat.new_session()
    r = chat.turn(sid, "Do something odd", tool("not_a_real_tool", {}), say("Sorry, that failed."))
    assert r["next_action"]["action"] == "error"


# ---------------------------------------------------------------------------
# Contract and audit trail
# ---------------------------------------------------------------------------

def test_model_is_offered_min_budget_and_availability(chat):
    sid = chat.new_session()
    chat.turn(sid, "Hello", say("Hi! What are you looking for?"))

    definitions = {t.name: t for t in chat.last_calls[0]["tools"]}
    properties = definitions["search_properties"].parameters["properties"]
    assert {"min_budget", "max_budget", "availability", "location", "configuration"} <= set(properties)
    assert "Ready to Move" in properties["availability"]["description"]


def test_audit_trail_records_what_the_model_sent_and_what_was_applied(chat):
    sid = _setup_two_bhk_in_bengaluru(chat)

    with chat.session_factory() as db:
        conversation = db.query(Conversation).filter_by(external_session_id=sid).one()
        rows = (
            db.query(Message)
            .filter_by(conversation_id=conversation.conversation_id)
            .order_by(Message.message_id)
            .all()
        )

    assert {r.turn_id for r in rows} == {1, 2}
    turn2 = [r for r in rows if r.turn_id == 2]
    assert [r.role for r in turn2] == ["customer", "tool", "tool", "assistant"]

    search = turn2[2]
    assert search.tool_name == "search_properties"
    assert search.tool_status == "success"
    assert search.tool_arguments == {}  # exactly what the model sent
    assert set(search.tool_result["applied_defaults"]) == {"configuration", "max_budget", "location"}


# ---------------------------------------------------------------------------
# Phase 3 exit criterion: qualify, match, consent, callback, revoke
# ---------------------------------------------------------------------------

def test_lead_is_qualified_matched_and_taken_through_consent_and_callback(chat):
    from backend.models import CallbackRequest, Lead

    sid = chat.new_session()

    chat.turn(
        sid,
        "2 BHK in Bengaluru, 50 to 80 lakh, I want to buy within 3 months",
        tool("update_customer_requirements", {
            "location": "Bengaluru", "configuration": "2 BHK",
            "budget_min": "5000000", "budget_max": "8000000",
            "purchase_timeline": "within 3 months",
        }),
        say("Great. Shall I share what suits you best?"),
    )

    r2 = chat.turn(
        sid,
        "What would you recommend?",
        tool("recommend_properties", {"limit": 3}),
        say("The best fit is a 2 BHK in Bengaluru."),
    )
    assert _ids(r2) == {chat.ids["A"]}
    rec = chat.tool_payloads(key="match_score")[0]
    assert rec["items"][0]["match_score"] == 100
    assert r2["next_action"]["action"] == "show_properties"

    chat.turn(
        sid,
        "My name is Asha, number 9876543210. Please call me tomorrow at 10.",
        tool("register_lead", {"customer_name": "Asha", "phone": "9876543210"}),
        tool("record_consent", {"consent_type": "callback_contact", "granted": True}),
        tool("create_callback_request", {
            "preferred_contact": "phone_call", "preferred_date": "2026-12-05",
            "preferred_time": "10:00", "reason": "Discuss the 2 BHK",
        }),
        say("Your callback is booked."),
    )

    with chat.session_factory() as db:
        lead = db.query(Lead).one()
        assert lead.buyer_intent == "callback_requesting"
        # budget 25 + one data label 20 + 3 months 16 + phone 10 + name 5 + callback 15
        assert float(lead.qualification_score) == 91
        assert lead.qualification_status == "Hot"
        assert lead.purchase_timeline == "within_3_months"
        assert db.query(CallbackRequest).one().status == "Pending"
        assert len(lead.property_matches) == 1

    # Asking again for the same slot does not create a second callback.
    chat.turn(
        sid,
        "Please call me tomorrow at 10",
        tool("create_callback_request", {
            "preferred_contact": "phone_call", "preferred_date": "2026-12-05",
            "preferred_time": "10:00", "reason": "Discuss the 2 BHK",
        }),
        say("That callback is already booked."),
    )
    assert chat.tool_payloads(key="outcome")[0]["outcome"] == "duplicate"

    # The customer withdraws consent: the callback is cancelled and the score drops.
    chat.turn(
        sid,
        "Actually, please don't call me",
        tool("record_consent", {"consent_type": "callback_contact", "granted": False}),
        say("Understood, I have cancelled the callback."),
    )
    with chat.session_factory() as db:
        lead = db.query(Lead).one()
        assert db.query(CallbackRequest).one().status == "Cancelled"
        assert lead.buyer_intent == "actively_searching"
        assert float(lead.qualification_score) == 76
        assert lead.score_breakdown["callback_request"]["points"] == 0


# ---------------------------------------------------------------------------
# Provider failures reach the customer as distinct, honest errors (4.9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error, status, kind",
    [
        ("GroqAuthenticationError", 502, "auth_failure"),
        ("GroqQuotaExceededError", 503, "quota_exceeded"),
        ("GroqRateLimitError", 429, "rate_limited"),
        ("GroqTimeoutError", 504, "timeout"),
        ("GroqNetworkError", 502, "network_error"),
        ("GroqResponseError", 502, "invalid_response"),
        ("GroqAPIError", 502, "provider_error"),
    ],
)
def test_llm_failures_are_reported_by_kind_without_provider_detail(chat, monkeypatch, error, status, kind):
    from backend.orchestrator.providers import groq_provider
    from backend.orchestrator.providers.failures import CUSTOMER_MESSAGES, FailureKind

    cls = getattr(groq_provider, error)
    exc = cls(500, "SECRET upstream text") if error == "GroqAPIError" else cls("SECRET upstream text")
    monkeypatch.setattr("backend.orchestrator.services.agent_service._PROVIDER_RETRY_DELAY", 0)

    async def fail(messages, tools=None, **kwargs):
        raise exc

    chat.llm.chat_completion = fail
    sid = chat.new_session()
    response = chat.client.post("/api/agent/turn", json={"session_id": sid, "message": "hello"})

    assert response.status_code == status
    assert response.headers["x-failure-kind"] == kind
    assert response.json()["detail"] == CUSTOMER_MESSAGES[FailureKind(kind)]
    assert "SECRET" not in response.text


def test_rate_limited_response_passes_retry_after_through(chat, monkeypatch):
    from backend.orchestrator.providers.groq_provider import GroqRateLimitError

    monkeypatch.setattr("backend.orchestrator.services.agent_service._PROVIDER_RETRY_DELAY", 0)

    async def fail(messages, tools=None, **kwargs):
        raise GroqRateLimitError("slow", retry_after="12")

    chat.llm.chat_completion = fail
    sid = chat.new_session()
    response = chat.client.post("/api/agent/turn", json={"session_id": sid, "message": "hello"})
    assert response.status_code == 429 and response.headers["retry-after"] == "12"


def test_unclassified_errors_stay_a_generic_502_with_no_kind(chat):
    async def fail(messages, tools=None, **kwargs):
        raise RuntimeError("SECRET internal detail")

    chat.llm.chat_completion = fail
    sid = chat.new_session()
    response = chat.client.post("/api/agent/turn", json={"session_id": sid, "message": "hello"})
    assert response.status_code == 502
    assert "x-failure-kind" not in response.headers
    assert "SECRET" not in response.text


def test_a_failed_turn_leaves_no_half_saved_data(chat):
    from backend.orchestrator.providers.groq_provider import GroqTimeoutError

    async def fail(messages, tools=None, **kwargs):
        raise GroqTimeoutError("t")

    sid = chat.new_session()
    chat.llm.chat_completion = fail
    chat.client.post("/api/agent/turn", json={"session_id": sid, "message": "I want a flat"})
    with chat.session_factory() as db:
        assert db.query(Message).count() == 0


# ---------------------------------------------------------------------------
# Sessions cannot see each other's data (4.10)
# ---------------------------------------------------------------------------

def _two_customers(chat):
    """Asha (session A) saves requirements, a lead, consent and a callback."""
    a = chat.new_session()
    chat.turn(
        a,
        "I'm Asha, 9876543210. 3 BHK in Bengaluru, call me tomorrow at 10",
        tool("update_customer_requirements", {"location": "Bengaluru", "configuration": "3 BHK", "budget_max": "12000000"}),
        tool("register_lead", {"customer_name": "Asha Rao", "phone": "9876543210"}),
        tool("record_consent", {"consent_type": "callback_contact", "granted": True}),
        tool("create_callback_request", {
            "preferred_contact": "phone_call", "preferred_date": "2026-12-05",
            "preferred_time": "10:00", "reason": "Asha wants a visit",
        }),
        say("Booked."),
    )
    return a, chat.new_session()


def test_requirements_do_not_leak_into_another_session(chat):
    a, b = _two_customers(chat)

    r = chat.turn(b, "Show me what you have", tool("search_properties", {}), say("Here is everything."))

    assert r["requirements"]["location"] is None and r["requirements"]["budget_max"] is None
    assert chat.tool_payloads(key="applied_defaults")[0]["applied_defaults"] == {}
    assert _ids(r) == set(chat.ids.values())  # nothing filtered by A's saved needs


def test_the_model_for_session_b_never_sees_session_a_text(chat):
    a, b = _two_customers(chat)

    chat.turn(b, "Hello there", say("Hi! How can I help?"))

    seen = " ".join((m.content or "") for call in chat.last_calls for m in call["messages"])
    # (Static prompt text mentions cities and sizes, so check A's own data.)
    for secret in ("Asha", "9876543210", "12000000", "Asha wants a visit"):
        assert secret not in seen


def test_lead_consent_and_callback_do_not_carry_over(chat):
    a, b = _two_customers(chat)

    r = chat.turn(
        b, "Please call me",
        tool("create_callback_request", {
            "preferred_contact": "phone_call", "preferred_date": "2026-12-06",
            "preferred_time": "11:00", "reason": "x",
        }),
        say("I need your details first."),
    )
    assert chat.tool_payloads(key="lead_required")[0]["error"] == "lead_required"

    with chat.session_factory() as db:
        from backend.models import CallbackRequest, ConsentRecord, Lead
        assert db.query(Lead).count() == 1
        assert db.query(CallbackRequest).count() == 1
        assert db.query(ConsentRecord).count() == 1


def test_a_second_customers_consent_does_not_authorise_the_first_customers_callback(chat):
    a, b = _two_customers(chat)
    chat.turn(
        b, "I'm Ravi, 9123456780",
        tool("register_lead", {"customer_name": "Ravi", "phone": "9123456780"}),
        say("Thanks Ravi."),
    )
    chat.turn(
        b, "Call me",
        tool("create_callback_request", {
            "preferred_contact": "phone_call", "preferred_date": "2026-12-06",
            "preferred_time": "11:00", "reason": "x",
        }),
        say("I need your permission first."),
    )
    assert chat.tool_payloads(key="callback_consent_required")[0]["error"] == "callback_consent_required"


def test_revoking_consent_in_one_session_does_not_touch_another(chat):
    from backend.models import CallbackRequest

    a, b = _two_customers(chat)
    chat.turn(
        b, "I'm Ravi", tool("register_lead", {"phone": "9123456780"}),
        tool("record_consent", {"consent_type": "callback_contact", "granted": True}),
        tool("create_callback_request", {
            "preferred_contact": "email", "preferred_date": "2026-12-06",
            "preferred_time": "11:00", "reason": "Ravi",
        }),
        say("Booked."),
    )
    chat.turn(a, "Don't call me", tool("record_consent", {"consent_type": "callback_contact", "granted": False}), say("Cancelled."))

    with chat.session_factory() as db:
        by_reason = {c.reason: c.status for c in db.query(CallbackRequest).all()}
    assert by_reason == {"Asha wants a visit": "Cancelled", "Ravi": "Pending"}


def test_message_history_is_scoped_to_the_session(chat):
    a, b = _two_customers(chat)
    chat.turn(b, "Hello from B", say("Hi B"))

    texts_b = [m["content"] for m in chat.client.get(f"/api/conversations/{b}/messages").json()]
    texts_a = [m["content"] for m in chat.client.get(f"/api/conversations/{a}/messages").json()]
    assert "Hello from B" in texts_b and not any("Asha" in t for t in texts_b)
    assert "Hello from B" not in texts_a


def test_unknown_and_guessed_session_ids_reveal_nothing(chat):
    a, _ = _two_customers(chat)
    for bad in ("1", "0", a[:-1] + ("0" if a[-1] != "0" else "1"), "../" + a, a.upper() + "x"):
        assert chat.client.get(f"/api/conversations/{bad}").status_code in (404, 422)
        assert chat.client.get(f"/api/conversations/{bad}/messages").status_code in (404, 422)
        assert chat.client.post("/api/agent/turn", json={"session_id": bad, "message": "hi"}).status_code == 404


def test_session_ids_are_long_random_and_unique(chat):
    ids = {chat.new_session() for _ in range(25)}
    assert len(ids) == 25
    assert all(len(i) >= 32 and i.isalnum() for i in ids)


def test_tool_audit_rows_belong_only_to_their_own_conversation(chat):
    from backend.models import Conversation

    a, b = _two_customers(chat)
    chat.turn(b, "hi", say("Hello"))
    with chat.session_factory() as db:
        conv_a = db.query(Conversation).filter_by(external_session_id=a).one()
        conv_b = db.query(Conversation).filter_by(external_session_id=b).one()
        tools_a = [m for m in db.query(Message).filter_by(conversation_id=conv_a.conversation_id) if m.role == "tool"]
        tools_b = [m for m in db.query(Message).filter_by(conversation_id=conv_b.conversation_id) if m.role == "tool"]
    assert len(tools_a) == 4 and tools_b == []
