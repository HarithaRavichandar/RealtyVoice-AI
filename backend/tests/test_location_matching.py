# -*- coding: utf-8 -*-
"""Tests for tolerant location matching, suggestions and result notes."""

from datetime import datetime

import pytest

from backend.models.property import Property
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
from backend.orchestrator.services.tool_executor import execute_tool
from backend.services.location_normalizer import (
    location_variants,
    squash,
    suggest_locations,
)
from backend.services.property_search import PropertySearchService

LOCATIONS = [
    "bangalore",
    "new-delhi",
    "greater-noida",
    "gurgaon",
    "mumbai",
    "Whitefield, Bangalore",
    "Connaught Place, Delhi",
]


@pytest.fixture
def seeded(dbsession):
    rows = {}
    for name in LOCATIONS:
        prop = Property(
            location=name,
            property_type="apartment",
            configuration="2 BHK",
            price_min=5_000_000,
            price_max=6_000_000,
            listing_price=5_500_000,
            possession_date=datetime(2026, 1, 1),
            availability_status=None,
        )
        dbsession.add(prop)
        rows[name] = prop
    dbsession.flush()
    return rows


def _service(session):
    return PropertySearchService(session_factory=lambda: session, close_session=False)


def _locations(session, query):
    _total, results = _service(session).search(location=query)
    return sorted(p.location for p in results)


# ---- helpers ---------------------------------------------------------------

def test_squash_ignores_case_spaces_and_punctuation():
    assert squash("White Field") == "whitefield"
    assert squash("New-Delhi, India") == "newdelhiindia"


def test_variants_include_alias_spellings():
    variants = location_variants("Whitefield, Bengaluru")
    assert "whitefieldbengaluru" in variants
    assert "whitefieldbangalore" in variants


def test_variants_empty_for_punctuation_only():
    assert location_variants(" ,, - ") == []


def test_suggest_locations_misspelling_and_parts():
    assert "Whitefield" in suggest_locations("Whitefeild", LOCATIONS)
    assert suggest_locations("Zzzzqqq", LOCATIONS) == []


# ---- service matching ------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("Bengaluru", ["Whitefield, Bangalore", "bangalore"]),
        ("bangalore", ["Whitefield, Bangalore", "bangalore"]),
        ("Whitefield, Bengaluru", ["Whitefield, Bangalore"]),
        ("White Field", ["Whitefield, Bangalore"]),
        ("greater noida", ["greater-noida"]),
        ("Greater-Noida", ["greater-noida"]),
        ("Gurugram", ["gurgaon"]),
        ("New Delhi", ["new-delhi"]),
        ("Delhi", ["Connaught Place, Delhi", "new-delhi"]),
        ("mumba", ["mumbai"]),
        ("Bombay", ["mumbai"]),
    ],
)
def test_service_location_matching(dbsession, seeded, query, expected):
    assert _locations(dbsession, query) == sorted(expected)


def test_service_unknown_location_returns_nothing(dbsession, seeded):
    assert _locations(dbsession, "Atlantis") == []


def test_service_punctuation_only_does_not_match_everything(dbsession, seeded):
    total, _ = _service(dbsession).search(location=" ,, ")
    assert total == 0


def test_distinct_locations(dbsession, seeded):
    assert sorted(_service(dbsession).distinct_locations()) == sorted(LOCATIONS)


# ---- tool: suggestions and notes ------------------------------------------

def test_tool_finds_bengaluru_spelling(dbsession, seeded):
    result = execute_tool(dbsession, "search_properties", {"location": "Bengaluru"})
    assert result["total"] == 2
    assert "location_suggestions" not in result


def test_tool_suggests_close_spelling_when_nothing_found(dbsession, seeded):
    result = execute_tool(dbsession, "search_properties", {"location": "Whitefeild"})
    assert result["total"] == 0
    assert "Whitefield" in result["location_suggestions"]


def test_tool_suggests_broader_city_for_unknown_neighbourhood(dbsession, seeded):
    result = execute_tool(
        dbsession, "search_properties", {"location": "Indiranagar, Bengaluru"}
    )
    assert result["total"] == 0
    assert "Bengaluru" in result["location_suggestions"]


def test_tool_no_suggestions_when_nothing_similar(dbsession, seeded):
    result = execute_tool(dbsession, "search_properties", {"location": "Atlantis"})
    assert result["total"] == 0
    assert "location_suggestions" not in result


# ---- tool: availability note ----------------------------------------------

def test_availability_note_present_when_filter_used(dbsession, seeded):
    result = execute_tool(
        dbsession, "search_properties", {"availability": "ready to move in"}
    )
    assert "availability_note" in result
    assert "unknown status are excluded" in result["availability_note"]


def test_availability_note_absent_without_filter(dbsession, seeded):
    result = execute_tool(dbsession, "search_properties", {"location": "mumbai"})
    assert "availability_note" not in result


# ---- prompt ----------------------------------------------------------------

def test_prompt_mentions_location_suggestions_and_availability_note():
    assert "location_suggestions" in SYSTEM_PROMPT
    assert "availability_note" in SYSTEM_PROMPT