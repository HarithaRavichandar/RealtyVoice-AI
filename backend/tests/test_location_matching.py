# -*- coding: utf-8 -*-
"""Tests for India-wide, data-driven location matching.

The test database is seeded with one row per location label that exists in
the real dataset (81 city-level slugs from the source CSV plus the 12
synthetic locality labels). These are throwaway in-memory rows used only to
exercise matching; no product data is created or changed.
"""

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from backend.main import app
from backend.models.property import Property
from backend.orchestrator.services.agent_service import SYSTEM_PROMPT
from backend.orchestrator.services.tool_executor import execute_tool
from backend.routes.property_search import get_property_search_service
from backend.services.location_normalizer import (
    default_alias_groups,
    load_alias_groups,
    location_variants,
    resolve_locations,
    squash,
    suggest_locations,
)
from backend.services.property_search import PropertySearchService

CITY_LABELS = [
    "new-delhi", "bangalore", "kolkata", "gurgaon", "ahmedabad", "hyderabad",
    "chennai", "jaipur", "greater-noida", "faridabad", "vadodara", "surat",
    "pune", "thane", "mumbai", "visakhapatnam", "mohali", "zirakpur",
    "chandigarh", "noida", "goa", "ranchi", "bhiwadi", "kochi", "dehradun",
    "ghaziabad", "lucknow", "kanpur", "patna", "guwahati", "sonipat",
    "coimbatore", "navi-mumbai", "nagpur", "jamshedpur", "kalyan",
    "vijayawada", "palghar", "raipur", "bhubaneswar", "agra", "aurangabad",
    "mangalore", "badlapur", "siliguri", "panchkula", "nashik", "guntur",
    "varanasi", "thrissur", "trivandrum", "allahabad", "ernakulam",
    "haridwar", "mysore", "durgapur", "gwalior", "ludhiana", "vapi",
    "indore", "bhopal", "jabalpur", "trichy", "udaipur", "bhiwandi",
    "rajahmundry", "tirupati", "belgaum", "jodhpur", "kozhikode", "satara",
    "shimla", "udupi", "vrindavan", "ahmadnagar", "madurai", "navsari",
    "nellore", "palakkad", "pondicherry", "solapur",
]
LOCALITY_LABELS = [
    "Bandra West, Mumbai", "Whitefield, Bangalore", "Connaught Place, Delhi",
    "Hitech City, Hyderabad", "Baner, Pune", "Besant Nagar, Chennai",
    "Rajasthan, Jaipur", "Gurgaon, Haryana", "Kolkata, Salt Lake",
    "Andheri East, Mumbai", "Panaji, Goa", "Navi Mumbai, Maharashtra",
]
ALL_LABELS = CITY_LABELS + LOCALITY_LABELS


def test_label_fixture_matches_real_dataset_size():
    assert len(CITY_LABELS) == 81
    assert len(LOCALITY_LABELS) == 12
    assert len(set(ALL_LABELS)) == 93


@pytest.fixture
def seeded(dbsession):
    for label in ALL_LABELS:
        dbsession.add(
            Property(
                location=label,
                property_type="apartment",
                configuration="2 BHK",
                price_min=5_000_000,
                price_max=6_000_000,
                listing_price=5_500_000,
                possession_date=datetime(2026, 1, 1),
                availability_status=None,
            )
        )
    dbsession.flush()
    return dbsession


def _service(session):
    return PropertySearchService(session_factory=lambda: session, close_session=False)


def _found(session, query):
    _total, results = _service(session).search(location=query, limit=100)
    return {p.location for p in results}


# ---------------------------------------------------------------------------
# Helpers and alias configuration
# ---------------------------------------------------------------------------

def test_squash_ignores_case_spaces_and_punctuation():
    assert squash("White Field") == "whitefield"
    assert squash("New-Delhi, India") == "newdelhiindia"


def test_variants_include_alias_spellings():
    variants = location_variants("Whitefield, Bengaluru")
    assert "whitefieldbengaluru" in variants
    assert "whitefieldbangalore" in variants


def test_variants_empty_for_punctuation_only():
    assert location_variants(" ,, - ") == []


def test_variants_do_not_double_up_new_delhi():
    assert location_variants("New Delhi") == ["newdelhi", "delhi"]
    assert location_variants("Delhi") == ["delhi", "newdelhi"]


def test_default_alias_file_is_valid_and_covers_required_cities():
    groups = default_alias_groups()
    flat = [name for group in groups for name in group]
    assert len(flat) == len(set(flat))  # no name in two groups
    assert all(len(group) >= 2 for group in groups)
    for a, b in [
        ("bengaluru", "bangalore"),
        ("gurugram", "gurgaon"),
        ("mumbai", "bombay"),
        ("kolkata", "calcutta"),
        ("chennai", "madras"),
        ("newdelhi", "delhi"),
    ]:
        assert any(a in g and b in g for g in groups), (a, b)


def test_aliases_are_configurable(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"groups": [["Foo City", "Bar Town"]]}))
    groups = load_alias_groups(path)
    assert groups == (("foocity", "bartown"),)
    assert "bartown" in location_variants("Foo City", groups)
    assert resolve_locations("Bar Town", ["foo-city", "mumbai"], groups) == ["foo-city"]
    # Without the custom group the alias is unknown.
    assert resolve_locations("Bar Town", ["foo-city", "mumbai"], ()) == []


def test_alias_file_rejects_duplicate_names_and_tiny_groups(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps({"groups": [["A b", "C d"], ["c d", "E f"]]}))
    with pytest.raises(ValueError, match="appears in groups"):
        load_alias_groups(dup)

    tiny = tmp_path / "tiny.json"
    tiny.write_text(json.dumps({"groups": [["Only One"]]}))
    with pytest.raises(ValueError, match="at least two"):
        load_alias_groups(tiny)


# ---------------------------------------------------------------------------
# Coverage across every label in the dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ALL_LABELS)
def test_every_label_finds_itself(seeded, label):
    assert label in _found(seeded, label)


@pytest.mark.parametrize("label", CITY_LABELS)
def test_every_city_found_by_spoken_form(seeded, label):
    spoken = label.replace("-", " ").title()  # "new-delhi" -> "New Delhi"
    assert label in _found(seeded, spoken)
    assert label in _found(seeded, spoken.upper())
    assert label in _found(seeded, spoken.replace(" ", ""))


# ---------------------------------------------------------------------------
# Multiple Indian cities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("Hyderabad", {"hyderabad", "Hitech City, Hyderabad"}),
        ("Jaipur", {"jaipur", "Rajasthan, Jaipur"}),
        ("Chennai", {"chennai", "Besant Nagar, Chennai"}),
        ("Pune", {"pune", "Baner, Pune"}),
        ("Kolkata", {"kolkata", "Kolkata, Salt Lake"}),
        ("Lucknow", {"lucknow"}),
        ("Guwahati", {"guwahati"}),
        ("Coimbatore", {"coimbatore"}),
        ("Indore", {"indore"}),
        ("Goa", {"goa", "Panaji, Goa"}),
        ("Visakhapatnam", {"visakhapatnam"}),
        ("Chandigarh", {"chandigarh"}),
    ],
)
def test_multiple_cities(seeded, query, expected):
    assert _found(seeded, query) == expected


# ---------------------------------------------------------------------------
# City aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("Bengaluru", {"bangalore", "Whitefield, Bangalore"}),
        ("Bengalooru", {"bangalore", "Whitefield, Bangalore"}),
        ("Bangaluru", {"bangalore", "Whitefield, Bangalore"}),
        ("Gurugram", {"gurgaon", "Gurgaon, Haryana"}),
        ("Bombay", {"mumbai", "Bandra West, Mumbai", "Andheri East, Mumbai"}),
        ("Calcutta", {"kolkata", "Kolkata, Salt Lake"}),
        ("Madras", {"chennai", "Besant Nagar, Chennai"}),
        ("Delhi", {"new-delhi", "Connaught Place, Delhi"}),
        ("New Delhi", {"new-delhi", "Connaught Place, Delhi"}),
        ("Poona", {"pune", "Baner, Pune"}),
        ("Vizag", {"visakhapatnam"}),
        ("Cochin", {"kochi"}),
        ("Baroda", {"vadodara"}),
        ("Mysuru", {"mysore"}),
        ("Mangaluru", {"mangalore"}),
        ("Prayagraj", {"allahabad"}),
        ("Puducherry", {"pondicherry"}),
        ("Thiruvananthapuram", {"trivandrum"}),
        ("Calicut", {"kozhikode"}),
        ("Benares", {"varanasi"}),
        ("Belagavi", {"belgaum"}),
    ],
)
def test_city_aliases(seeded, query, expected):
    assert _found(seeded, query) == expected


# ---------------------------------------------------------------------------
# Hyphenated labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("new-delhi", {"new-delhi", "Connaught Place, Delhi"}),
        ("New Delhi", {"new-delhi", "Connaught Place, Delhi"}),
        ("NewDelhi", {"new-delhi", "Connaught Place, Delhi"}),
        ("greater noida", {"greater-noida"}),
        ("Greater-Noida", {"greater-noida"}),
        ("GreaterNoida", {"greater-noida"}),
        ("navi mumbai", {"navi-mumbai", "Navi Mumbai, Maharashtra"}),
        ("Navi-Mumbai", {"navi-mumbai", "Navi Mumbai, Maharashtra"}),
    ],
)
def test_hyphenated_labels(seeded, query, expected):
    assert _found(seeded, query) == expected


# ---------------------------------------------------------------------------
# Precision: a short city name must not pull in a longer, different place
# ---------------------------------------------------------------------------

def test_noida_does_not_match_greater_noida(seeded):
    assert _found(seeded, "Noida") == {"noida"}


def test_patna_does_not_match_visakhapatnam(seeded):
    assert _found(seeded, "Patna") == {"patna"}


def test_mumbai_does_not_match_navi_mumbai(seeded):
    assert _found(seeded, "Mumbai") == {
        "mumbai", "Bandra West, Mumbai", "Andheri East, Mumbai",
    }


# ---------------------------------------------------------------------------
# Locality variations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("Whitefield", {"Whitefield, Bangalore"}),
        ("White Field", {"Whitefield, Bangalore"}),
        ("Whitefield Bangalore", {"Whitefield, Bangalore"}),
        ("Whitefield, Bengaluru", {"Whitefield, Bangalore"}),
        ("Bangalore Whitefield", {"Whitefield, Bangalore"}),
        ("Hitech City", {"Hitech City, Hyderabad"}),
        ("Hi-tech City", {"Hitech City, Hyderabad"}),
        ("Hi tech city Hyderabad", {"Hitech City, Hyderabad"}),
        ("Salt Lake, Kolkata", {"Kolkata, Salt Lake"}),
        ("Salt Lake Calcutta", {"Kolkata, Salt Lake"}),
        ("Bandra West", {"Bandra West, Mumbai"}),
        ("Bandra West Bombay", {"Bandra West, Mumbai"}),
        ("Besant Nagar", {"Besant Nagar, Chennai"}),
        ("Baner Pune", {"Baner, Pune"}),
        ("Connaught Place", {"Connaught Place, Delhi"}),
        ("Connaught Place New Delhi", {"Connaught Place, Delhi"}),
    ],
)
def test_locality_variations(seeded, query, expected):
    assert _found(seeded, query) == expected


def test_typed_prefix_still_matches_partially(seeded):
    assert "mumbai" in _found(seeded, "mumba")
    assert "Bandra West, Mumbai" in _found(seeded, "Bandra")


# ---------------------------------------------------------------------------
# Speech-to-text spelling variations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, suggested",
    [
        ("Whitefeild", "Whitefield"),
        ("Bangalre", "Bangalore"),
        ("Hydrabad", "Hyderabad"),
        ("Ahmadabad", "Ahmedabad"),
        ("Chenai", "Chennai"),
        ("Gurgoan", "Gurgaon"),
        ("Vishakapatnam", "Visakhapatnam"),
        ("Coimbatur", "Coimbatore"),
    ],
)
def test_misspellings_find_nothing_but_suggest_real_names(seeded, query, suggested):
    assert _found(seeded, query) == set()
    result = execute_tool(seeded, "search_properties", {"location": query})
    assert result["total"] == 0
    similar = [s for s in result["location_suggestions"] if s["type"] == "similar_name"]
    assert suggested in [s["location"] for s in similar]


# ---------------------------------------------------------------------------
# Unknown locations and cities absent from the database
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["Atlantis", "xyz123", "Qwertyuiop"])
def test_unknown_locations_return_nothing_and_no_suggestions(seeded, query):
    result = execute_tool(seeded, "search_properties", {"location": query})
    assert result["total"] == 0
    assert "location_suggestions" not in result


@pytest.mark.parametrize("query", ["Amritsar", "Rajkot", "Srinagar", "Meerut"])
def test_real_cities_absent_from_data_are_not_invented(seeded, query):
    assert squash(query) not in {squash(l) for l in ALL_LABELS}
    result = execute_tool(seeded, "search_properties", {"location": query})
    assert result["total"] == 0
    assert result["items"] == []
    assert "location_suggestions" not in result


def test_alias_of_absent_city_finds_nothing(seeded):
    assert _found(seeded, "Bengal-nowhere") == set()


# ---------------------------------------------------------------------------
# Parent-city suggestions
# ---------------------------------------------------------------------------

def _suggestions(session, query):
    result = execute_tool(session, "search_properties", {"location": query})
    assert result["total"] == 0
    return result["location_suggestions"]


@pytest.mark.parametrize(
    "query, city",
    [
        ("Indiranagar, Bengaluru", "Bangalore"),
        ("Koramangala Bangalore", "Bangalore"),
        ("Powai, Bombay", "Mumbai"),
        ("Gachibowli Hyderabad", "Hyderabad"),
        ("Sector 62 Noida", "Noida"),
        ("Salt Lake City, Calcutta", "Kolkata"),
        ("Anna Nagar, Madras", "Chennai"),
        ("Aundh Poona", "Pune"),
        ("Dwarka, New Delhi", "New Delhi"),
    ],
)
def test_parent_city_suggested_for_unavailable_locality(seeded, query, city):
    parents = [s for s in _suggestions(seeded, query) if s["type"] == "parent_city"]
    assert city in [s["location"] for s in parents]
    note = parents[0]["note"]
    assert "city-level" in note
    assert "does not confirm" in note


def test_greater_noida_locality_suggests_greater_noida_first(seeded):
    parents = [
        s["location"]
        for s in _suggestions(seeded, "Knowledge Park Greater Noida")
        if s["type"] == "parent_city"
    ]
    assert parents[0] == "Greater Noida"


def test_all_suggestions_come_from_real_locations(seeded):
    real_keys = set()
    for label in ALL_LABELS:
        real_keys.add(squash(label))
        real_keys.update(squash(p) for p in label.split(",") if p.strip())

    queries = [
        "Indiranagar, Bengaluru", "Whitefeild", "Hydrabad", "Sector 62 Noida",
        "Koramangala Bangalore", "Bangalre", "Anna Nagar, Madras",
    ]
    for query in queries:
        for suggestion in _suggestions(seeded, query):
            assert squash(suggestion["location"]) in real_keys, suggestion
            assert suggestion["type"] in {"parent_city", "similar_name"}


def test_suggest_locations_pure_function():
    out = suggest_locations("Indiranagar, Bengaluru", ["bangalore", "Whitefield, Bangalore"])
    assert [s["location"] for s in out] == ["Bangalore"]
    assert out[0]["type"] == "parent_city"
    assert suggest_locations("Anywhere", []) == []
    assert suggest_locations(" , ", ["bangalore"]) == []


# ---------------------------------------------------------------------------
# Existing Bengaluru behaviour is preserved
# ---------------------------------------------------------------------------

def test_bengaluru_behaviour_preserved(seeded):
    assert _found(seeded, "Bangalore") == {"bangalore", "Whitefield, Bangalore"}
    assert _found(seeded, "Bengaluru") == {"bangalore", "Whitefield, Bangalore"}
    assert _found(seeded, "Whitefield, Bengaluru") == {"Whitefield, Bangalore"}
    result = execute_tool(seeded, "search_properties", {"location": "Bengaluru"})
    assert result["total"] == 2
    assert "location_suggestions" not in result


def test_bengaluru_stored_label_still_found(dbsession):
    dbsession.add(Property(location="Bengaluru", property_type="flat"))
    dbsession.flush()
    assert _found(dbsession, "Bangalore") == {"Bengaluru"}
    assert _found(dbsession, "Bengaluru") == {"Bengaluru"}


# ---------------------------------------------------------------------------
# REST endpoint and agent tool share the same behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
def client(seeded):
    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_property_search_service] = lambda: _service(seeded)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


@pytest.mark.parametrize(
    "query",
    [
        "Bengaluru", "Bombay", "Gurugram", "New Delhi", "greater noida",
        "White Field", "Salt Lake, Kolkata", "Noida", "Patna", "Amritsar",
        "Whitefeild",
    ],
)
def test_rest_and_agent_tool_return_the_same_properties(seeded, client, query):
    response = client.get("/properties/search", params={"location": query, "limit": 100})
    assert response.status_code == 200
    rest_ids = sorted(i["property_id"] for i in response.json()["items"])

    tool = execute_tool(seeded, "search_properties", {"location": query})
    tool_ids = sorted(i["property_id"] for i in tool["items"])

    service_ids = sorted(
        p.property_id for p in _service(seeded).search(location=query, limit=100)[1]
    )
    assert rest_ids == tool_ids == service_ids


# ---------------------------------------------------------------------------
# Matching stays label-based (no per-row text work)
# ---------------------------------------------------------------------------

def test_location_filter_uses_exact_in_on_distinct_labels(seeded):
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    engine = seeded.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        _service(seeded).search(location="White Field")
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    joined = "\n".join(statements).lower()
    assert "distinct" in joined  # the label lookup
    assert " in (" in joined  # exact IN filter on the matched labels
    assert "replace(" not in joined  # no per-row text rewriting
    assert "like" not in joined  # no per-row substring scan


# ---------------------------------------------------------------------------
# Availability note and prompt (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_availability_note_present_when_filter_used(seeded):
    result = execute_tool(seeded, "search_properties", {"availability": "ready to move in"})
    assert "unknown status are excluded" in result["availability_note"]


def test_availability_note_absent_without_filter(seeded):
    result = execute_tool(seeded, "search_properties", {"location": "mumbai"})
    assert "availability_note" not in result


def test_prompt_describes_suggestion_types():
    assert "location_suggestions" in SYSTEM_PROMPT
    assert "parent_city" in SYSTEM_PROMPT
    assert "similar_name" in SYSTEM_PROMPT
    assert "availability_note" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Electronic City / E-City (roadmap task 2.4)
#
# The current dataset has no Electronic City label (the 93-label fixture above
# mirrors it), so these tests use their own labels. The alias only helps match
# labels that exist; it never creates inventory.
# ---------------------------------------------------------------------------

ECITY_LABELS = ["Electronic City, Bangalore", "Whitefield, Bangalore", "bangalore"]


def test_electronic_city_names_share_one_alias_group():
    groups = default_alias_groups()
    assert any(
        {"electroniccity", "electronicscity", "ecity"} <= set(group) for group in groups
    )
    assert squash("E-City") == squash("E City") == squash("ECity") == "ecity"


@pytest.mark.parametrize(
    "query",
    [
        "Electronic City",
        "Electronics City",
        "E-City",
        "E City",
        "ecity",
        "Electronic City, Bangalore",
        "Electronic City Bengaluru",
        "E-City Bangalore",
        "E-City, Bengaluru",
        "Bengaluru E City",
    ],
)
def test_electronic_city_variants_find_the_stored_label(query):
    assert resolve_locations(query, ECITY_LABELS) == ["Electronic City, Bangalore"]


def test_variants_combine_two_aliased_names():
    variants = location_variants("E-City Bengaluru")
    assert "ecitybengaluru" == variants[0]
    assert "electroniccitybangalore" in variants


def test_single_alias_variants_are_unchanged_by_combining():
    assert location_variants("Whitefield, Bengaluru") == [
        "whitefieldbengaluru",
        "whitefieldbangalore",
        "whitefieldbengalooru",
        "whitefieldbangaluru",
    ]


def test_electronic_city_alias_does_not_match_unrelated_labels():
    assert resolve_locations("E-City", ["Whitefield, Bangalore", "bangalore", "electricity-nagar"]) == []


def test_without_an_electronic_city_label_nothing_is_invented(seeded):
    # The real dataset has no such label, so searches still find nothing.
    assert _found(seeded, "Electronic City") == set()
    assert _found(seeded, "E-City Bengaluru") == set()
