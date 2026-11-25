"""Unit tests for router response-building helpers.

Covers _build_agent_response intent detection and _collect_matched_properties
extraction — the two pure functions that sit between the agent loop and the
API response.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, List

from backend.orchestrator.schemas import (
    AgentResponse,
    CustomerRequirements,
    NextAction,
)
from backend.orchestrator.router import (
    _build_agent_response,
    _collect_matched_properties,
)
from backend.orchestrator.services.agent_service import AgentRunResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_result(
    tool_results: list[dict[str, Any]] | None = None,
    reply: str = "Test reply",
    iteration_limit_reached: bool = False,
    requirements: CustomerRequirements | None = None,
) -> AgentRunResult:
    """Build a minimal AgentRunResult for testing."""
    return AgentRunResult(
        session_id="sess-1",
        conversation_id=1,
        turn_id=1,
        reply=reply,
        spoken_reply=reply,
        requirements=requirements or CustomerRequirements(),
        tool_results=tool_results or [],
        tool_iterations=1,
        iteration_limit_reached=iteration_limit_reached,
    )


def _search_tool_result(items: list[dict] | None = None) -> dict:
    return {
        "tool_name": "search_properties",
        "result": {"items": items or []},
    }


def _details_tool_result(prop: dict | None = None) -> dict:
    return {
        "tool_name": "get_property_details",
        "result": {"property": prop} if prop else {"property": None},
    }


def _callback_tool_result(succeeded: bool = True) -> dict:
    if succeeded:
        return {
            "tool_name": "create_callback_request",
            "result": {"callback": {"id": 1, "status": "Pending"}},
        }
    return {
        "tool_name": "create_callback_request",
        "result": {"error": "consent_not_granted"},
    }


SAMPLE_PROPERTY = {
    "property_id": 42,
    "project_name": "Sunrise Towers",
    "location": "Mumbai",
    "property_type": "Apartment",
    "configuration": "2BHK",
    "price_min": 4500000,
    "price_max": 5500000,
    "listing_price": 5000000,
    "is_synthetic": False,
    "source_name": "kaggle",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}


# ==================================================================
# _collect_matched_properties
# ==================================================================

class TestCollectMatchedProperties:

    def test_extracts_from_search_items(self):
        tool_results = [_search_tool_result([SAMPLE_PROPERTY])]
        props = _collect_matched_properties(tool_results)
        assert len(props) == 1
        assert props[0].property_id == 42

    def test_extracts_from_details_property(self):
        tool_results = [_details_tool_result(SAMPLE_PROPERTY)]
        props = _collect_matched_properties(tool_results)
        assert len(props) == 1
        assert props[0].property_id == 42

    def test_deduplicates_across_tools(self):
        """Same property from search and details should appear once."""
        tool_results = [
            _search_tool_result([SAMPLE_PROPERTY]),
            _details_tool_result(SAMPLE_PROPERTY),
        ]
        props = _collect_matched_properties(tool_results)
        assert len(props) == 1

    def test_multiple_distinct_properties(self):
        prop2 = {**SAMPLE_PROPERTY, "property_id": 99, "project_name": "Green Villas"}
        tool_results = [_search_tool_result([SAMPLE_PROPERTY, prop2])]
        props = _collect_matched_properties(tool_results)
        assert len(props) == 2
        ids = {p.property_id for p in props}
        assert ids == {42, 99}

    def test_skips_malformed_entries(self):
        bad = {"property_id": "not-a-number", "title": 123}
        tool_results = [_search_tool_result([bad, SAMPLE_PROPERTY])]
        props = _collect_matched_properties(tool_results)
        assert len(props) == 1
        assert props[0].property_id == 42

    def test_empty_tool_results(self):
        assert _collect_matched_properties([]) == []

    def test_no_items_or_property_keys(self):
        tool_results = [{"tool_name": "search_properties", "result": {}}]
        assert _collect_matched_properties(tool_results) == []

    def test_none_result_is_handled(self):
        tool_results = [{"tool_name": "search_properties", "result": None}]
        assert _collect_matched_properties(tool_results) == []


# ==================================================================
# _build_agent_response — intent detection
# ==================================================================

class TestBuildAgentResponseIntent:

    def test_iteration_limit_sets_unknown_error(self):
        result = _make_result(iteration_limit_reached=True, reply="Gave up")
        resp = _build_agent_response(result, "Gave up")
        assert resp.intent == "unknown"
        assert resp.next_action.action == "error"

    def test_callback_success_intent(self):
        result = _make_result(tool_results=[_callback_tool_result(True)])
        resp = _build_agent_response(result, "Callback scheduled!")
        assert resp.intent == "callback_request"
        assert resp.next_action.action == "await_user"

    def test_callback_failure_intent(self):
        result = _make_result(tool_results=[_callback_tool_result(False)])
        resp = _build_agent_response(result, "Need consent first")
        assert resp.intent == "callback_request"
        assert resp.next_action.action == "ask_clarification"

    def test_property_search_with_results(self):
        result = _make_result(
            tool_results=[_search_tool_result([SAMPLE_PROPERTY])]
        )
        resp = _build_agent_response(result, "Found properties")
        assert resp.intent == "property_search"
        assert resp.next_action.action == "show_properties"
        assert resp.next_action.property_ids == [42]

    def test_property_search_no_results(self):
        result = _make_result(tool_results=[_search_tool_result([])])
        resp = _build_agent_response(result, "No matches found")
        assert resp.intent == "property_search"
        assert resp.next_action.action == "ask_clarification"

    def test_property_details_with_result(self):
        result = _make_result(
            tool_results=[_details_tool_result(SAMPLE_PROPERTY)]
        )
        resp = _build_agent_response(result, "Here are the details")
        assert resp.intent == "property_details"
        assert resp.next_action.action == "show_properties"

    def test_property_details_no_result(self):
        result = _make_result(
            tool_results=[_details_tool_result(None)]
        )
        resp = _build_agent_response(result, "Could not find that property")
        assert resp.intent == "property_details"
        assert resp.next_action.action == "ask_clarification"

    def test_general_question_await(self):
        """No tools called, reply without question mark → await_user."""
        result = _make_result(tool_results=[], reply="Hello, welcome!")
        resp = _build_agent_response(result, "Hello, welcome!")
        assert resp.intent == "general_question"
        assert resp.next_action.action == "await_user"

    def test_general_question_clarification(self):
        """No tools called, reply with question mark → ask_clarification."""
        result = _make_result(tool_results=[], reply="What area are you looking in?")
        resp = _build_agent_response(result, "What area are you looking in?")
        assert resp.intent == "general_question"
        assert resp.next_action.action == "ask_clarification"

    def test_requirements_only_is_general(self):
        """update_customer_requirements alone → general_question."""
        result = _make_result(
            tool_results=[{"tool_name": "update_customer_requirements", "result": {}}]
        )
        resp = _build_agent_response(result, "Got your preferences.")
        assert resp.intent == "general_question"

    def test_register_lead_alone_is_general(self):
        """register_lead alone → general_question."""
        result = _make_result(
            tool_results=[{"tool_name": "register_lead", "result": {}}]
        )
        resp = _build_agent_response(result, "Lead registered.")
        assert resp.intent == "general_question"

    def test_record_consent_alone_is_general(self):
        """record_consent alone → general_question."""
        result = _make_result(
            tool_results=[{"tool_name": "record_consent", "result": {}}]
        )
        resp = _build_agent_response(result, "Consent noted.")
        assert resp.intent == "general_question"

    def test_unknown_tool_triggers_error(self):
        """An unrecognized tool name → unknown + error."""
        result = _make_result(
            tool_results=[{"tool_name": "some_rogue_tool", "result": {}}]
        )
        resp = _build_agent_response(result, "Something weird happened")
        assert resp.intent == "unknown"
        assert resp.next_action.action == "error"

    def test_iteration_limit_takes_priority(self):
        """iteration_limit_reached overrides tool-based detection."""
        result = _make_result(
            tool_results=[_search_tool_result([SAMPLE_PROPERTY])],
            iteration_limit_reached=True,
        )
        resp = _build_agent_response(result, "Hit the limit")
        assert resp.intent == "unknown"
        assert resp.next_action.action == "error"

    def test_callback_takes_priority_over_search(self):
        """When both callback and search tools ran, callback wins."""
        result = _make_result(
            tool_results=[
                _search_tool_result([SAMPLE_PROPERTY]),
                _callback_tool_result(True),
            ]
        )
        resp = _build_agent_response(result, "Done")
        assert resp.intent == "callback_request"

    def test_response_carries_session_and_turn(self):
        result = _make_result()
        resp = _build_agent_response(result, "Hi")
        assert resp.session_id == "sess-1"
        assert resp.turn_id == 1

    def test_response_carries_requirements(self):
        reqs = CustomerRequirements(location="Mumbai", property_type="Apartment")
        result = _make_result(requirements=reqs)
        resp = _build_agent_response(result, "Noted")
        assert resp.requirements.location == "Mumbai"
        assert resp.requirements.property_type == "Apartment"
