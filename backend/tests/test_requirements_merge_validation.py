# -*- coding: utf-8 -*-
"""update_customer_requirements must never store an inconsistent budget range.

Each update is valid on its own; the merged result may not be. A conflicting
update is rejected with ``requirements_conflict`` and nothing is stored, and
clearing a field with ``null`` works.
"""

import json

import pytest

from backend.models import Conversation
from backend.orchestrator.services.agent_service import _get_current_requirements
from backend.orchestrator.services.tool_executor import execute_tool


@pytest.fixture
def conv(dbsession):
    c = Conversation(channel="browser_text", external_session_id="merge", temp_requirements=None)
    dbsession.add(c)
    dbsession.flush()
    return c


def _update(session, conv, **fields):
    return execute_tool(
        session, "update_customer_requirements", fields, conversation_id=conv.conversation_id
    )


def test_maximum_below_saved_minimum_is_rejected_and_not_stored(dbsession, conv):
    _update(dbsession, conv, budget_min="9000000")

    result = _update(dbsession, conv, budget_max="5000000")

    assert result["error"] == "requirements_conflict"
    assert "nothing was saved" in result["message"]
    assert result["saved_budget_min"] == "9000000"
    assert result["saved_budget_max"] is None
    assert conv.temp_requirements == {"budget_min": "9000000"}
    # The stored state is still readable by the agent.
    assert str(_get_current_requirements(dbsession, conv.conversation_id).budget_min) == "9000000"


def test_minimum_above_saved_maximum_is_rejected(dbsession, conv):
    _update(dbsession, conv, budget_max="5000000")

    result = _update(dbsession, conv, budget_min="9000000")

    assert result["error"] == "requirements_conflict"
    assert conv.temp_requirements == {"budget_max": "5000000"}


def test_conflict_result_is_json_serialisable_for_the_audit_log(dbsession, conv):
    _update(dbsession, conv, budget_min="9000000")
    result = _update(dbsession, conv, budget_max="5000000")

    json.dumps(result)  # would raise on Decimal / date objects
    assert all(set(d) == {"loc", "msg", "type"} for d in result["details"])


def test_conflicting_bound_can_be_replaced_in_one_call(dbsession, conv):
    _update(dbsession, conv, budget_min="9000000")

    result = _update(dbsession, conv, budget_min=None, budget_max="5000000")

    assert "error" not in result
    assert conv.temp_requirements == {"budget_min": None, "budget_max": "5000000"}


@pytest.mark.parametrize(
    "first, second",
    [
        ({"budget_min": "5000000"}, {"budget_max": "9000000"}),
        ({"budget_max": "9000000"}, {"budget_min": "5000000"}),
        ({"budget_min": "5000000"}, {"budget_max": "5000000"}),  # equal bounds are fine
    ],
)
def test_consistent_updates_across_turns_are_stored(dbsession, conv, first, second):
    assert "error" not in _update(dbsession, conv, **first)
    assert "error" not in _update(dbsession, conv, **second)
    assert set(conv.temp_requirements) == set(first) | set(second)


def test_null_clears_a_saved_requirement(dbsession, conv):
    _update(dbsession, conv, location="Pune", budget_max="5000000")

    result = _update(dbsession, conv, budget_max=None)

    assert "error" not in result
    assert conv.temp_requirements["budget_max"] is None
    assert conv.temp_requirements["location"] == "Pune"
    assert _get_current_requirements(dbsession, conv.conversation_id).budget_max is None


def test_update_that_repairs_an_already_inconsistent_state_is_allowed(dbsession, conv):
    # State written before this check existed.
    conv.temp_requirements = {"budget_min": "9000000", "budget_max": "5000000"}
    dbsession.flush()

    # Unrelated updates are rejected until the conflict is resolved...
    assert _update(dbsession, conv, location="Pune")["error"] == "requirements_conflict"

    # ...and the clearing update repairs it.
    result = _update(dbsession, conv, budget_min=None)
    assert "error" not in result
    assert _get_current_requirements(dbsession, conv.conversation_id).budget_max is not None
