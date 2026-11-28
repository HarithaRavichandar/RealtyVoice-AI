# -*- coding: utf-8 -*-
"""Non-numeric budget input is a normal validation error, never a crash."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.orchestrator.services.tool_executor import execute_tool
from backend.orchestrator.tools.schemas import SearchPropertiesRequest, UpdateCustomerRequirementsRequest

BAD = ["abc", "", "   ", "50 lakh", "₹5000000", "NaN", "Infinity", "-Infinity", "1,5", True, [], {}]
GOOD = [("5000000", Decimal("5000000")), (5_000_000, Decimal("5000000")), (" 7500000.50 ", Decimal("7500000.50")),
        (1.5, Decimal("1.5")), ("0", Decimal("0")), ("1e7", Decimal("1E+7"))]


@pytest.mark.parametrize("bad", BAD, ids=repr)
@pytest.mark.parametrize("field", ["min_budget", "max_budget"])
def test_search_rejects_non_numeric_budgets_with_a_validation_error(field, bad):
    with pytest.raises(ValidationError):
        SearchPropertiesRequest(**{field: bad})


@pytest.mark.parametrize("bad", BAD, ids=repr)
@pytest.mark.parametrize("field", ["budget_min", "budget_max"])
def test_saving_requirements_rejects_non_numeric_budgets(field, bad):
    with pytest.raises(ValidationError):
        UpdateCustomerRequirementsRequest(**{field: bad})


@pytest.mark.parametrize("raw, expected", GOOD)
def test_valid_numbers_are_still_accepted(raw, expected):
    assert SearchPropertiesRequest(max_budget=raw).max_budget == expected
    assert UpdateCustomerRequirementsRequest(budget_min=raw).budget_min == expected


def test_none_still_means_not_given_or_clear():
    assert SearchPropertiesRequest(max_budget=None).max_budget is None
    assert UpdateCustomerRequirementsRequest(budget_max=None).budget_max is None


@pytest.mark.parametrize("tool_name, args", [
    ("search_properties", {"max_budget": "abc"}),
    ("search_properties", {"min_budget": "NaN"}),
    ("update_customer_requirements", {"budget_max": "lots"}),
])
def test_the_tools_return_a_validation_result_instead_of_raising(dbsession, tool_name, args):
    from backend.models import Conversation

    conv = Conversation(channel="browser_text", external_session_id="bad1", temp_requirements=None)
    dbsession.add(conv)
    dbsession.flush()

    result = execute_tool(dbsession, tool_name, args, conversation_id=conv.conversation_id)

    assert result["error"] == "validation"
    assert conv.temp_requirements in (None, {})


def test_a_rejected_budget_does_not_disturb_saved_requirements(dbsession):
    from backend.models import Conversation

    conv = Conversation(channel="browser_text", external_session_id="bad2", temp_requirements=None)
    dbsession.add(conv)
    dbsession.flush()
    execute_tool(dbsession, "update_customer_requirements", {"budget_max": "5000000"}, conversation_id=conv.conversation_id)

    execute_tool(dbsession, "update_customer_requirements", {"budget_max": "abc"}, conversation_id=conv.conversation_id)

    assert conv.temp_requirements == {"budget_max": "5000000"}
