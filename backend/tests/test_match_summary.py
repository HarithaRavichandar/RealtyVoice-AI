# -*- coding: utf-8 -*-
"""Per-property match facts returned by search_properties, and the prompt
rules that tell the agent to explain them (task 2.9)."""

from datetime import datetime
from decimal import Decimal

import pytest

from backend.models.property import Property
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
from backend.orchestrator.services.tool_executor import execute_tool
from backend.schemas.property_search import PropertySearchItem
from backend.services.match_summary import build_match_summary

NOW = datetime(2026, 1, 1)


def item(**kw):
    base = dict(
        property_id=1, location="bangalore", property_type="apartment",
        configuration="2 BHK", price_min=Decimal("5000000"), price_max=Decimal("6000000"),
        possession_date=datetime(2025, 6, 1), availability_status="Ready to Move",
        created_at=NOW, updated_at=NOW,
    )
    base.update(kw)
    return PropertySearchItem(**base)


def codes(summary):
    return {n["code"] for n in summary["notes"]}


def test_matched_lists_each_applied_filter_with_actual_value():
    s = build_match_summary(
        item(), location="Bengaluru", configuration="2 BHK",
        min_budget=Decimal("4000000"), max_budget=Decimal("7000000"),
        availability="Ready to Move",
    )
    got = {m["filter"]: m for m in s["matched"]}
    assert set(got) == {"location", "configuration", "budget", "availability"}
    assert got["location"]["requested"] == "Bengaluru"
    assert got["location"]["actual"] == "bangalore"
    assert got["budget"]["actual"] == "5,000,000-6,000,000"
    assert s["notes"] == []


def test_unfiltered_search_has_no_matched_entries():
    assert build_match_summary(item())["matched"] == []


def test_price_range_partly_over_and_under_budget_is_noted():
    over = build_match_summary(item(), max_budget=Decimal("5500000"))
    assert codes(over) == {"price_range_above_budget"}
    under = build_match_summary(item(), min_budget=Decimal("5500000"))
    assert codes(under) == {"price_range_below_budget"}


def test_missing_data_is_flagged_not_guessed():
    s = build_match_summary(
        item(price_min=None, price_max=None, possession_date=None, availability_status=None)
    )
    assert codes(s) == {"price_unknown", "possession_unknown", "availability_unknown"}


def test_listing_price_is_used_when_no_range():
    s = build_match_summary(
        item(price_min=None, price_max=None, listing_price=Decimal("5000000")),
        max_budget=Decimal("6000000"),
    )
    assert s["matched"][0]["actual"] == "5,000,000"
    assert "price_unknown" not in codes(s)


@pytest.fixture
def seeded(dbsession):
    dbsession.add_all([
        Property(location="bangalore", property_type="apartment", configuration="2 BHK",
                 price_min=5_000_000, price_max=9_000_000, listing_price=7_000_000,
                 availability_status=None),
        Property(location="bangalore", property_type="apartment", configuration="2 BHK",
                 price_min=None, price_max=None, listing_price=None),
    ])
    dbsession.flush()
    return dbsession


def test_search_tool_attaches_match_to_every_item(seeded):
    result = execute_tool(
        seeded, "search_properties",
        {"location": "Bengaluru", "max_budget": "6000000", "configuration": "2 BHK"},
    )
    assert result["total"] == 1
    match = result["items"][0]["match"]
    assert {m["filter"] for m in match["matched"]} == {"location", "configuration", "budget"}
    assert codes(match) == {"price_range_above_budget", "possession_unknown", "availability_unknown"}


def test_search_tool_items_have_match_even_without_filters(seeded):
    result = execute_tool(seeded, "search_properties", {})
    assert result["total"] == 2
    assert all("match" in i for i in result["items"])


@pytest.mark.parametrize("fragment", ['"match" field', "match \"notes\"", "Never hide a", "Recommend only properties returned"])
def test_prompt_requires_explaining_matches_and_notes(fragment):
    assert fragment in SYSTEM_PROMPT
