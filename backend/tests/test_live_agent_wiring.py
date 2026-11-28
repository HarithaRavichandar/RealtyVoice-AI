# -*- coding: utf-8 -*-
"""Phase 5 agent wiring: the search_live_properties tool and the durable ledger.

No test here reaches the network: the live provider is either a mock or the
real API.SH provider behind ``httpx.MockTransport`` serving the saved fixture.
"""

import json
import logging
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import config
from backend.database import Base, get_db
from backend.main import app
from backend.models import Conversation, LiveCreditEntry
from backend.orchestrator.providers.base import ProviderResponse, ToolCall
from backend.orchestrator.schemas import CustomerRequirements
from backend.orchestrator.router import _build_agent_response, _collect_matched_properties
from backend.orchestrator.services import tool_executor
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT, run_agent_loop
from backend.orchestrator.services.conversation import create_conversation
from backend.orchestrator.services.provider_tools import (
    TOOL_DESCRIPTIONS,
    get_provider_tool_definitions,
)
from backend.orchestrator.services.tool_executor import execute_tool
from backend.orchestrator.services.tool_registry import TOOL_REGISTRY
from backend.orchestrator.tools.schemas import SearchLivePropertiesRequest
from backend.services.inventory import (
    ApishPropertyProvider,
    InventoryAuthError,
    InventoryNetworkError,
    InventoryTimeoutError,
    MockPropertyProvider,
    ProviderResult,
)
from backend.services.inventory.models import Inventory
from backend.services.inventory.normalize import APISH_SOURCE, APISH_SOURCE_LABEL
from backend.services.inventory.sql_ledger import (
    DUPLICATE,
    RESERVED,
    SqlCreditLedger,
    forget_pending,
    restore_after_rollback,
)
from backend.services.schema_upgrades import ensure_live_credit_ledger_table

FIXTURE = Path(__file__).parent / "fixtures" / "apish_whitefield_search.json"
SECRET = "sk-test-SECRET-key-12345"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat_completion(self, messages, tools=None, **kwargs):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self):
        pass


def live_call(arguments=None, call_id="call-1"):
    return ToolCall(
        id=call_id,
        function_name="search_live_properties",
        arguments=arguments or {"location": "Whitefield, Bangalore"},
    )


def tool_turn(*calls):
    return ProviderResponse(content=None, tool_calls=list(calls))


def final(text="Done."):
    return ProviderResponse(content=text)


class ApishFixture:
    """The real API.SH provider served from the saved response."""

    def __init__(self, status=200, body=None, error=None):
        self.requests = []
        self.status = status
        self.body = body
        self.error = error

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error:
            raise self.error
        body = self.body if self.body is not None else json.loads(FIXTURE.read_text("utf-8"))
        return httpx.Response(self.status, json=body)

    def provider(self):
        return ApishPropertyProvider(
            api_key=SECRET,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


@pytest.fixture
def live(monkeypatch):
    """Configure live search; returns a setter for the provider in use."""
    monkeypatch.setattr(config, "LIVE_CREDITS_PER_CONVERSATION", 3)
    monkeypatch.setattr(tool_executor, "_live_configured", lambda: True)
    holder = SimpleNamespace(provider=None)

    def use(provider):
        holder.provider = provider
        monkeypatch.setattr(tool_executor, "_create_live_provider", lambda: provider)
        return provider

    holder.use = use
    return holder


@pytest.fixture
def conversation(dbsession):
    return create_conversation(dbsession, channel="browser_text")


def ledger_rows(session, conversation):
    return (
        session.query(LiveCreditEntry)
        .filter_by(conversation_id=conversation.conversation_id)
        .order_by(LiveCreditEntry.entry_id)
        .all()
    )


async def run_turn(session, conversation, provider_responses, message="show live listings"):
    return await run_agent_loop(
        session,
        conversation.external_session_id,
        message,
        provider=FakeProvider(provider_responses),
    )


def one_result(result):
    assert len(result.tool_results) == 1
    return result.tool_results[0]["result"]


# ---------------------------------------------------------------------------
# registration, schema and prompt
# ---------------------------------------------------------------------------

def test_tool_is_registered_and_offered_to_the_model():
    assert "search_live_properties" in TOOL_REGISTRY
    assert TOOL_REGISTRY["search_live_properties"].input_schema is SearchLivePropertiesRequest
    names = {d.name for d in get_provider_tool_definitions()}
    assert "search_live_properties" in names and "search_properties" in names
    assert set(TOOL_DESCRIPTIONS) == set(TOOL_REGISTRY)


def test_description_restricts_live_search_to_explicit_requests():
    text = TOOL_DESCRIPTIONS["search_live_properties"]
    assert "ONLY when the customer explicitly" in text
    assert "search_properties instead" in text
    assert "credit" in text
    assert "demo" in TOOL_DESCRIPTIONS["get_property_details"]


def test_ordinary_search_description_unchanged_in_meaning():
    assert "Use this when the customer wants property options." in TOOL_DESCRIPTIONS["search_properties"]


def test_schema_requires_location_and_rejects_extras():
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest()
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="   ")
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", page=2)
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", fresh=True)
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", availability="Ready to Move")


def test_schema_budget_rules():
    ok = SearchLivePropertiesRequest(location="Whitefield", min_budget="5000000", max_budget="9000000")
    assert ok.min_budget == 5000000
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", min_budget=9, max_budget=5)
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", max_budget=-1)
    with pytest.raises(ValidationError):
        SearchLivePropertiesRequest(location="Whitefield", max_budget="abc")


@pytest.mark.parametrize(
    "fragment",
    [
        "LIVE LISTINGS (search_live_properties)",
        "explicitly asks",
        "Never describe demo data as live",
        "Never top up a short or empty",
        "Never invent a price",
        "live_credit_limit_reached",
        "live_search_failed",
        "do\n    not silently use demo data",
    ],
)
def test_prompt_contains_live_rules(fragment):
    squash = lambda text: " ".join(text.split())  # noqa: E731
    assert squash(fragment) in squash(SYSTEM_PROMPT)


def test_prompt_keeps_earlier_rules():
    assert "is_synthetic" in SYSTEM_PROMPT and "Ready to Move" in SYSTEM_PROMPT
    assert "45." in SYSTEM_PROMPT and "46." in SYSTEM_PROMPT and "51." in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# executor: marker, saved defaults
# ---------------------------------------------------------------------------

def test_execute_tool_defers_live_search(dbsession, conversation):
    result = execute_tool(
        dbsession, "search_live_properties",
        {"location": "Whitefield, Bangalore", "max_budget": 9000000},
        conversation_id=conversation.conversation_id,
    )
    assert result["_async_pending"] is True
    assert result["criteria"]["location"] == "Whitefield, Bangalore"
    assert result["criteria"]["max_budget"] == 9000000
    assert "availability" not in result["criteria"]


def test_execute_tool_requires_conversation(dbsession):
    result = execute_tool(dbsession, "search_live_properties", {"location": "Whitefield"})
    assert result["error"] == "missing_conversation_id"


def test_execute_tool_validation_error_for_missing_location(dbsession, conversation):
    result = execute_tool(
        dbsession, "search_live_properties", {}, conversation_id=conversation.conversation_id
    )
    assert result["error"] == "validation"


def test_saved_requirements_fill_gaps_but_never_override(dbsession, conversation):
    conversation.temp_requirements = {
        "location": "Pune",
        "budget_max": "7000000",
        "configuration": "2 BHK",
        "availability": "Ready to Move",
    }
    dbsession.flush()
    result = execute_tool(
        dbsession, "search_live_properties",
        {"location": "Whitefield, Bangalore", "configuration": "3 BHK"},
        conversation_id=conversation.conversation_id,
    )
    criteria = result["criteria"]
    assert criteria["location"] == "Whitefield, Bangalore"
    assert criteria["configuration"] == "3 BHK"
    assert criteria["max_budget"] == 7000000
    assert "availability" not in criteria
    assert "availability" not in result["applied_defaults"]
    assert set(result["applied_defaults"]) == {"max_budget"}


# ---------------------------------------------------------------------------
# durable ledger
# ---------------------------------------------------------------------------

def test_ledger_reserve_settle_and_totals(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    key = str(conversation.conversation_id)
    assert ledger.used(key) == 0
    assert ledger.reserve(key, "apish_99acres", "r1") == RESERVED
    assert ledger.used(key) == 1 and ledger.confirmed(key) == 0  # a hold, not confirmed
    ledger.settle(key, "apish_99acres", "r1", consumed=1, unconfirmed=0, outcome="ok")
    assert ledger.used(key) == 1 and ledger.confirmed(key) == 1
    row = ledger.entries(key)[0]
    assert (row.provider, row.request_id, row.credits_consumed, row.credits_unconfirmed, row.outcome) == (
        "apish_99acres", "r1", 1, 0, "ok")
    assert row.created_at is not None


def test_ledger_duplicate_request_is_not_double_counted(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    key = str(conversation.conversation_id)
    assert ledger.reserve(key, "p", "same") == RESERVED
    ledger.settle(key, "p", "same", consumed=1, unconfirmed=0, outcome="ok")
    assert ledger.reserve(key, "p", "same") == DUPLICATE
    assert ledger.used(key) == 1
    assert len(ledger.entries(key)) == 1
    # Same id under another provider or conversation is a different request.
    assert ledger.reserve(key, "other", "same") == RESERVED


def test_ledger_is_per_conversation(dbsession):
    a = create_conversation(dbsession, channel="browser_text")
    b = create_conversation(dbsession, channel="browser_text")
    ledger = SqlCreditLedger(dbsession)
    ledger.reserve(str(a.conversation_id), "p", "r1")
    assert ledger.used(str(a.conversation_id)) == 1
    assert ledger.used(str(b.conversation_id)) == 0


def test_ledger_validation(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    key = str(conversation.conversation_id)
    with pytest.raises(ValueError):
        ledger.used("not-a-number")
    with pytest.raises(ValueError):
        ledger.reserve(key, "p", "  ")
    with pytest.raises(KeyError):
        ledger.settle(key, "p", "never-reserved", consumed=0, unconfirmed=0, outcome="ok")
    ledger.reserve(key, "p", "r")
    with pytest.raises(ValueError):
        ledger.settle(key, "p", "r", consumed=-1, unconfirmed=0, outcome="ok")


def test_ledger_survives_restart(tmp_path):
    """Rows written and committed are there for a brand-new engine."""
    url = f"sqlite:///{tmp_path / 'ledger.db'}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as s:
        conv = Conversation(external_session_id="restart-1", channel="browser_text")
        s.add(conv)
        s.commit()
        cid = conv.conversation_id
        ledger = SqlCreditLedger(s)
        ledger.reserve(str(cid), "apish_99acres", "r1")
        ledger.settle(str(cid), "apish_99acres", "r1", consumed=2, unconfirmed=0, outcome="ok")
        s.commit()
    engine.dispose()

    engine2 = create_engine(url, future=True)  # "restart"
    with sessionmaker(bind=engine2, future=True)() as s:
        ledger = SqlCreditLedger(s)
        assert ledger.used(str(cid)) == 2 and ledger.confirmed(str(cid)) == 2
    engine2.dispose()


def test_rollback_restore_keeps_spent_credits(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    key = str(conversation.conversation_id)
    dbsession.commit()
    ledger.reserve(key, "p", "r1")
    ledger.settle(key, "p", "r1", consumed=1, unconfirmed=0, outcome="ok")
    dbsession.rollback()  # the router discards a failed turn
    assert ledger.used(key) == 0
    assert restore_after_rollback(dbsession) == 1
    assert ledger.used(key) == 1 and ledger.confirmed(key) == 1
    assert restore_after_rollback(dbsession) == 0  # nothing left to write


def test_restore_never_duplicates_committed_rows(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    key = str(conversation.conversation_id)
    ledger.reserve(key, "p", "r1")
    ledger.settle(key, "p", "r1", consumed=1, unconfirmed=0, outcome="ok")
    dbsession.commit()
    assert restore_after_rollback(dbsession) == 1  # idempotent update, not an insert
    assert ledger.used(key) == 1
    assert len(ledger.entries(key)) == 1


def test_forget_pending_after_commit(dbsession, conversation):
    ledger = SqlCreditLedger(dbsession)
    ledger.reserve(str(conversation.conversation_id), "p", "r1")
    dbsession.commit()
    forget_pending(dbsession)
    assert restore_after_rollback(dbsession) == 0


def test_ledger_table_upgrade_is_additive_and_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE live_credit_ledger")
    assert ensure_live_credit_ledger_table(engine) is True
    assert "live_credit_ledger" in inspect(engine).get_table_names()
    assert ensure_live_credit_ledger_table(engine) is False
    engine.dispose()


# ---------------------------------------------------------------------------
# agent loop end to end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_search_returns_labelled_live_results(dbsession, conversation, live):
    fixture = ApishFixture()
    live.use(fixture.provider())
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)

    assert data["inventory"] == "live"
    assert data["source"] == APISH_SOURCE
    assert data["source_label"] == APISH_SOURCE_LABEL
    assert data["fallback_used"] is False
    assert data["status"] == "ok" and data["total"] == len(data["items"]) == 2
    for item in data["items"]:
        assert item["inventory"] == "live"
        assert item["source"] == APISH_SOURCE_LABEL
        assert item["listing_id"] and item["title"] and item["url"]
        assert "property_id" not in item
    # Projects were counted and dropped, not shown.
    assert data["excluded"].get("project_records") == 25
    # One request, only documented parameters, key only in the header.
    assert len(fixture.requests) == 1
    request = fixture.requests[0]
    assert request.headers["x-api-key"] == SECRET
    assert SECRET not in str(request.url)
    assert not {"page", "fresh", "raw", "fields"} & set(request.url.params)
    assert data["credits"]["limit_per_conversation"] == 3
    assert data["credits"]["used_in_conversation"] == 1
    assert data["credits"]["remaining"] == 2
    assert SECRET not in json.dumps(data, default=str)
    assert result.reply == "Done."


@pytest.mark.asyncio
async def test_live_search_records_confirmed_credit_in_ledger(dbsession, conversation, live):
    live.use(ApishFixture().provider())
    await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    rows = ledger_rows(dbsession, conversation)
    assert len(rows) == 1
    row = rows[0]
    assert row.provider == APISH_SOURCE
    assert row.request_id == "1:call-1"
    # The fixture response states credits_used: that figure is confirmed.
    expected = json.loads(FIXTURE.read_text("utf-8"))["credits_used"]
    assert row.credits_consumed == expected and row.credits_unconfirmed == 0
    assert row.outcome == "ok"


@pytest.mark.asyncio
async def test_unreported_cost_is_counted_but_not_called_confirmed(dbsession, conversation, live):
    body = json.loads(FIXTURE.read_text("utf-8"))
    body.pop("credits_used")
    live.use(ApishFixture(body=body).provider())
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert data["credits"]["confirmed_by_provider_this_search"] == 0
    assert data["credits"]["counted_without_confirmation_this_search"] == 1
    row = ledger_rows(dbsession, conversation)[0]
    assert (row.credits_consumed, row.credits_unconfirmed) == (0, 1)


@pytest.mark.asyncio
async def test_ordinary_search_never_uses_live_inventory(dbsession, conversation, live):
    mock = live.use(MockPropertyProvider())
    result = await run_turn(
        dbsession, conversation,
        [tool_turn(ToolCall(id="c1", function_name="search_properties", arguments={"location": "Whitefield"})),
         final()],
    )
    data = one_result(result)
    assert mock.calls == []
    assert data["tool"] == "search_properties"
    assert data.get("inventory") != "live"
    assert ledger_rows(dbsession, conversation) == []


@pytest.mark.asyncio
async def test_credit_limit_stops_the_fourth_call(dbsession, conversation, live):
    mock = live.use(MockPropertyProvider(ProviderResult(
        source="mock_live", inventory=Inventory.LIVE, credits_used=1)))
    calls = [live_call(call_id=f"c{i}") for i in range(4)]
    result = await run_turn(
        dbsession, conversation, [tool_turn(*calls), final()]
    )
    outcomes = [r["result"] for r in result.tool_results]
    assert [o.get("status") for o in outcomes] == ["no_results", "no_results", "no_results", "credit_limit_reached"]
    assert outcomes[3]["error"] == "live_credit_limit_reached"
    assert outcomes[3]["credits"]["used_in_conversation"] == 3
    assert outcomes[3]["fallback_used"] is False
    assert len(mock.calls) == 3  # the fourth was never sent
    assert SqlCreditLedger(dbsession).used(str(conversation.conversation_id)) == 3


@pytest.mark.asyncio
async def test_limit_is_configurable(dbsession, conversation, live, monkeypatch):
    monkeypatch.setattr(config, "LIVE_CREDITS_PER_CONVERSATION", 1)
    mock = live.use(MockPropertyProvider(ProviderResult(
        source="mock_live", inventory=Inventory.LIVE, credits_used=1)))
    result = await run_turn(
        dbsession, conversation,
        [tool_turn(live_call(call_id="a"), live_call(call_id="b")), final()],
    )
    assert result.tool_results[1]["result"]["error"] == "live_credit_limit_reached"
    assert len(mock.calls) == 1


@pytest.mark.asyncio
async def test_limit_survives_a_new_turn_and_new_session(dbsession, conversation, live):
    mock = live.use(MockPropertyProvider(ProviderResult(
        source="mock_live", inventory=Inventory.LIVE, credits_used=3)))
    await run_turn(dbsession, conversation, [tool_turn(live_call(call_id="a")), final()])
    dbsession.commit()
    result = await run_turn(dbsession, conversation, [tool_turn(live_call(call_id="b")), final()])
    assert one_result(result)["error"] == "live_credit_limit_reached"
    assert len(mock.calls) == 1


@pytest.mark.asyncio
async def test_repeated_tool_call_id_is_not_sent_or_counted_twice(dbsession, conversation, live):
    mock = live.use(MockPropertyProvider(ProviderResult(
        source="mock_live", inventory=Inventory.LIVE, credits_used=1)))
    result = await run_turn(
        dbsession, conversation,
        [tool_turn(live_call(call_id="dup"), live_call(call_id="dup")), final()],
    )
    first, second = (r["result"] for r in result.tool_results)
    assert first["status"] == "no_results"
    assert second["error"] == "live_duplicate_request"
    assert len(mock.calls) == 1
    assert len(ledger_rows(dbsession, conversation)) == 1
    assert SqlCreditLedger(dbsession).used(str(conversation.conversation_id)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, kind",
    [
        (InventoryNetworkError("down"), "network_error"),
        (InventoryTimeoutError("slow"), "timeout"),
    ],
)
async def test_live_failure_is_reported_honestly_with_no_fallback(
    dbsession, conversation, live, error, kind
):
    live.use(MockPropertyProvider(error))
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final("Sorry.")])
    data = one_result(result)  # exactly one tool ran: nothing else was substituted
    assert data["error"] == "live_search_failed"
    assert data["failure_kind"] == kind
    assert data["fallback_used"] is False
    assert "items" not in data and "total" not in data
    assert data["inventory"] == "live"
    assert "message" in data and SECRET not in json.dumps(data)
    # The request was sent, so a credit is held - but not claimed as confirmed.
    row = ledger_rows(dbsession, conversation)[0]
    assert (row.credits_consumed, row.credits_unconfirmed) == (0, 1)
    assert row.outcome == "failed" and row.failure_kind == kind


@pytest.mark.asyncio
async def test_refused_call_is_free_and_not_sent_again(dbsession, conversation, live):
    live.use(MockPropertyProvider(InventoryAuthError("bad key")))
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert data["error"] == "live_search_failed" and data["failure_kind"] == "auth_failure"
    row = ledger_rows(dbsession, conversation)[0]
    assert (row.credits_consumed, row.credits_unconfirmed) == (0, 0)


@pytest.mark.asyncio
async def test_http_failure_through_real_provider_never_leaks_key(dbsession, conversation, live, caplog):
    caplog.set_level(logging.DEBUG)
    live.use(ApishFixture(status=500, body={"success": False, "error": f"boom {SECRET}"}).provider())
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert data["error"] == "live_search_failed"
    assert SECRET not in json.dumps(data, default=str)
    assert SECRET not in caplog.text
    assert SECRET not in json.dumps([m.content for m in _stored(dbsession, conversation)])


def _stored(session, conversation):
    from backend.orchestrator.services.message import get_conversation_messages

    return get_conversation_messages(session, conversation.conversation_id)


@pytest.mark.asyncio
async def test_unexpected_provider_exception_is_contained(dbsession, conversation, live, caplog):
    caplog.set_level(logging.DEBUG)
    live.use(MockPropertyProvider(RuntimeError(f"secret detail {SECRET}")))
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert data["error"] == "live_search_failed" and data["fallback_used"] is False
    assert SECRET not in json.dumps(data)
    assert SECRET not in caplog.text and "RuntimeError" in caplog.text
    # The hold stays: a call may have been made.
    assert SqlCreditLedger(dbsession).used(str(conversation.conversation_id)) == 1


@pytest.mark.asyncio
async def test_not_configured_makes_no_request_and_no_ledger_entry(dbsession, conversation, monkeypatch):
    monkeypatch.setattr(tool_executor, "_live_configured", lambda: False)
    monkeypatch.setattr(config, "LIVE_CREDITS_PER_CONVERSATION", 3)

    def boom():
        raise AssertionError("provider must not be created")

    monkeypatch.setattr(tool_executor, "_create_live_provider", boom)
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert data["error"] == "live_not_configured" and data["fallback_used"] is False
    assert ledger_rows(dbsession, conversation) == []


@pytest.mark.asyncio
async def test_limit_zero_disables_live_search(dbsession, conversation, live, monkeypatch):
    monkeypatch.setattr(config, "LIVE_CREDITS_PER_CONVERSATION", 0)
    mock = live.use(MockPropertyProvider())
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    assert one_result(result)["error"] == "live_search_disabled"
    assert mock.calls == [] and ledger_rows(dbsession, conversation) == []


@pytest.mark.asyncio
async def test_tool_result_is_stored_with_source_in_the_audit_trail(dbsession, conversation, live):
    live.use(ApishFixture().provider())
    await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    tool_messages = [m for m in _stored(dbsession, conversation) if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_name == "search_live_properties"
    assert tool_messages[0].tool_status == "success"
    assert APISH_SOURCE_LABEL in tool_messages[0].content


@pytest.mark.asyncio
async def test_budget_is_enforced_locally_and_unpriced_listings_are_counted(dbsession, conversation, live):
    live.use(ApishFixture().provider())
    result = await run_turn(
        dbsession, conversation,
        [tool_turn(live_call({"location": "Whitefield, Bangalore", "max_budget": 1000})), final()],
    )
    data = one_result(result)
    assert data["total"] == 0 and data["items"] == []
    assert data["excluded"].get("budget_mismatch", 0) + data["excluded"].get("budget_unknown", 0) == 2


@pytest.mark.asyncio
async def test_saved_budget_applies_to_live_search_and_is_reported(dbsession, conversation, live):
    conversation.temp_requirements = {"budget_max": "1000"}
    dbsession.flush()
    live.use(ApishFixture().provider())
    result = await run_turn(dbsession, conversation, [tool_turn(live_call()), final()])
    data = one_result(result)
    assert {k: Decimal(str(v)) for k, v in data["applied_defaults"].items()} == {
        "max_budget": Decimal(1000)}
    assert data["total"] == 0


# ---------------------------------------------------------------------------
# router: live results are separate from demo property records
# ---------------------------------------------------------------------------

LIVE_RESULT = {
    "tool": "search_live_properties", "inventory": "live",
    "items": [{"listing_id": "123", "title": "x", "property_id": 5, "price": 1}],
}


def _response_for(tool_results, reply="ok"):
    result = SimpleNamespace(
        session_id="s1", turn_id=1, tool_results=tool_results,
        requirements=CustomerRequirements(), iteration_limit_reached=False,
        spoken_reply=reply,
    )
    return _build_agent_response(result, reply)


def test_live_items_are_never_reported_as_demo_property_records():
    entries = [{"tool_name": "search_live_properties", "result": LIVE_RESULT}]
    assert _collect_matched_properties(entries) == []


def test_live_search_is_a_property_search_with_no_demo_ids():
    response = _response_for([{"tool_name": "search_live_properties", "result": LIVE_RESULT}])
    assert response.intent == "property_search"
    assert response.next_action.action == "await_user"
    assert not response.next_action.property_ids


def test_live_failure_is_an_error_next_action():
    failed = {"tool": "search_live_properties", "inventory": "live", "error": "live_search_failed"}
    response = _response_for([{"tool_name": "search_live_properties", "result": failed}])
    assert response.intent == "property_search"
    assert response.next_action.action == "error"


# ---------------------------------------------------------------------------
# router: spent credits survive a failed turn
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            yield client, SessionLocal
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()


def _failing_turn(monkeypatch, *, fail):
    async def fake_run(session, session_id, incoming_message):
        conv = session.query(Conversation).filter_by(external_session_id=session_id).one()
        ledger = SqlCreditLedger(session)
        ledger.reserve(str(conv.conversation_id), "apish_99acres", "1:call-x")
        ledger.settle(str(conv.conversation_id), "apish_99acres", "1:call-x",
                      consumed=1, unconfirmed=0, outcome="ok")
        if fail:
            raise RuntimeError("model failed after the live search")
        return SimpleNamespace(
            session_id=session_id, conversation_id=conv.conversation_id, turn_id=1,
            reply="ok", spoken_reply="ok", requirements=CustomerRequirements(),
            tool_results=[], tool_iterations=0, iteration_limit_reached=False,
        )

    monkeypatch.setattr("backend.orchestrator.router.run_agent_loop", fake_run)


def test_credits_stay_recorded_when_the_turn_fails_afterwards(api_client, monkeypatch):
    client, SessionLocal = api_client
    session_id = client.post("/api/conversations", json={"channel": "browser_text"}).json()["session_id"]
    _failing_turn(monkeypatch, fail=True)
    response = client.post("/api/agent/turn", json={"session_id": session_id, "message": "live please"})
    assert response.status_code == 502
    with SessionLocal() as s:
        rows = s.query(LiveCreditEntry).all()
        assert len(rows) == 1 and rows[0].credits_consumed == 1 and rows[0].request_id == "1:call-x"


def test_credits_are_committed_with_a_successful_turn(api_client, monkeypatch):
    client, SessionLocal = api_client
    session_id = client.post("/api/conversations", json={"channel": "browser_text"}).json()["session_id"]
    _failing_turn(monkeypatch, fail=False)
    response = client.post("/api/agent/turn", json={"session_id": session_id, "message": "live please"})
    assert response.status_code == 200
    with SessionLocal() as s:
        assert s.query(LiveCreditEntry).count() == 1


# ---------------------------------------------------------------------------
# the library is unchanged for callers that do not use reservations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_memory_ledger_flow_is_unchanged():
    from backend.services.inventory import CreditPolicy, InMemoryCreditLedger, InventorySearchService, SearchCriteria

    mock = MockPropertyProvider(ProviderResult(source="mock_live", inventory=Inventory.LIVE, credits_used=1))
    service = InventorySearchService(
        {Inventory.LIVE: mock}, ledger=InMemoryCreditLedger(), policy=CreditPolicy(2)
    )
    for _ in range(2):
        outcome = await service.search(Inventory.LIVE, "c", SearchCriteria(location="Pune"))
        assert outcome.credits_charged == 1
    blocked = await service.search(Inventory.LIVE, "c", SearchCriteria(location="Pune"))
    assert blocked.status == "credit_limit_reached" and len(mock.calls) == 2
