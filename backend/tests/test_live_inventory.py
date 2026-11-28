# -*- coding: utf-8 -*-
"""Tests for the source-aware inventory package (roadmap Phase 5).

``fixtures/apish_whitefield_search.json`` is a real API.SH response (Whitefield,
2 BHK apartments, ``budget_max=15000000``) trimmed to remove ``raw``, images,
long descriptions and advertiser photos. It has 25 project records and 2
individual property records. No test touches the network: HTTP is simulated
with ``httpx.MockTransport``.

What is NOT tested because it is not verified by API.SH documentation: caching,
pagination, retries, ``fresh``/``raw``/``fields`` behaviour, and the exact
meaning of 401/402/403/429/5xx. The status mapping tests pin the conventional
mapping only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from backend.models.property import Property
from backend.orchestrator.providers.failures import FailureKind
from backend.services.inventory import (
    ApishPropertyProvider,
    CreditPolicy,
    InMemoryCreditLedger,
    Inventory,
    InventoryAPIError,
    InventoryAuthError,
    InventoryConfigError,
    InventoryNetworkError,
    InventoryQuotaError,
    InventoryRateLimitError,
    InventoryRequestError,
    InventoryResponseError,
    InventorySearchService,
    InventoryTimeoutError,
    MockPropertyProvider,
    ProviderResult,
    RecordKind,
    SearchCriteria,
    SourceRef,
    SQLiteProvider,
)
from backend.services.inventory.apish_provider import split_location
from backend.services.inventory.dedup import dedupe
from backend.services.inventory.models import NormalizedProperty
from backend.services.inventory.normalize import (
    APISH_SOURCE,
    NormalizationError,
    matched_locality_name,
    normalize_applied_filters,
    normalize_property_record,
    possession_from_tags,
)
from backend.services.inventory.validation import (
    MATCH,
    MISMATCH,
    NOT_REQUESTED,
    UNKNOWN,
    bedrooms_from_configuration,
    check_bedrooms,
    check_budget,
    check_possession,
    check_property_type,
    documented_type,
)
from backend.services.property_search import PropertySearchService

FIXTURE = Path(__file__).parent / "fixtures" / "apish_whitefield_search.json"
SECRET = "sk-test-SECRET-key-12345"


@pytest.fixture
def body():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _properties(body):
    return [r for r in body["results"] if r["kind"] == "property"]


def _provider(handler, key=SECRET):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ApishPropertyProvider(key, client=client)


def _ok(body):
    return lambda request: httpx.Response(200, json=body)


WHITEFIELD = SearchCriteria(
    location="Whitefield, Bangalore",
    property_type="apartment",
    configuration="2 BHK",
    max_budget=Decimal(15_000_000),
)


def _listing(**kw):
    base = dict(
        ref=SourceRef(APISH_SOURCE, RecordKind.PROPERTY, kw.pop("external_id", "1")),
        inventory=Inventory.LIVE,
        source_label="live",
    )
    base.update(kw)
    return NormalizedProperty(**base)


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------

def test_fixture_mixes_projects_and_properties(body):
    kinds = [r["kind"] for r in body["results"]]
    assert kinds.count("project") == 25 and kinds.count("property") == 2
    assert "raw" not in json.dumps(body["results"][0])
    assert "x-api-key" not in FIXTURE.read_text(encoding="utf-8").lower()


def test_fixture_shows_the_type_differences_the_code_must_handle(body):
    project = next(r for r in body["results"] if r["kind"] == "project")
    prop = _properties(body)[0]
    assert isinstance(project["price_text"], dict) and isinstance(prop["price_text"], str)
    assert isinstance(project["price_per_sqft"], str) and isinstance(prop["price_per_sqft"], int)
    assert isinstance(body["location"], dict) and isinstance(project["location"], str)
    assert isinstance(body["applied_filters"]["budget_max"], str)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_property_record_normalizes_verified_fields(body):
    fetched = datetime(2026, 11, 27, 12, 13, tzinfo=timezone.utc)
    p = normalize_property_record(_properties(body)[0], fetched_at=fetched, cached=True)
    assert p.ref.key == "apish_99acres:property:94869686"
    assert p.ref.kind is RecordKind.PROPERTY and p.inventory is Inventory.LIVE
    assert p.prop_id == "T94869686" and p.project_id == "441258"
    assert (p.price, p.price_max) == (Decimal(9_660_000), Decimal(10_200_000))
    assert p.price_text == "96.6 L - 1.02 Cr" and p.price_per_sqft == 8180
    assert p.bedrooms == 2 and p.bathrooms == 0 and p.balconies is None
    assert p.locality == "Whitefield, Bangalore" and p.city == "Bangalore East"
    assert isinstance(p.latitude, float) and isinstance(p.longitude, float)
    assert p.advertiser_type == "Builder" and p.verified is False
    assert p.posted == datetime(2026, 10, 5, 18, 31, 4, tzinfo=timezone.utc)
    assert p.expires_text == "2027-03-05 00:00:00"  # kept verbatim, no zone assumed
    assert p.fetched_at == fetched and p.cached is True
    assert not hasattr(p, "raw")


def test_possession_kept_verbatim_and_never_invented(body):
    p = normalize_property_record(_properties(body)[0])
    assert p.possession_text == "POSSESSION IN MAR 2028"
    assert p.possession_month == (2028, 3)
    assert p.possession_date is None  # no day of the month is made up
    other = normalize_property_record(_properties(body)[1])
    assert "NEW LAUNCH" in other.tags          # other wording stays in tags, uninterpreted
    assert other.possession_text == "POSSESSION IN DEC 2029"
    assert other.possession_month == (2029, 12)


@pytest.mark.parametrize(
    "tags, expected",
    [
        (("READY TO MOVE",), (None, None)),
        (("FEATURED", "POSSESSION IN MAR 2028"), ("POSSESSION IN MAR 2028", (2028, 3))),
        (("possession in dec 2030",), ("possession in dec 2030", (2030, 12))),
        (("POSSESSION IN XYZ 2028",), (None, None)),
        (("POSSESSION SOON",), (None, None)),
        ((), (None, None)),
    ],
)
def test_possession_from_tags(tags, expected):
    assert possession_from_tags(tags) == expected


def test_project_record_is_not_a_property(body):
    project = next(r for r in body["results"] if r["kind"] == "project")
    with pytest.raises(NormalizationError):
        normalize_property_record(project)


def test_record_without_id_is_rejected(body):
    rec = dict(_properties(body)[0], id=None)
    with pytest.raises(NormalizationError):
        normalize_property_record(rec)


def test_wrong_types_become_unknown_not_guesses(body):
    rec = dict(
        _properties(body)[0],
        price={"min": 1},          # an object, as on projects
        price_text={"label": "x"},
        price_per_sqft="13,342 /sqft",
        bedrooms=True,
        latitude="12.9",
        tags="not a list",
        verified="yes",
    )
    p = normalize_property_record(rec)
    assert p.price is None and p.price_text is None and p.price_per_sqft is None
    assert p.bedrooms is None and p.latitude is None and p.tags == () and p.verified is None


def test_applied_filters_strings_are_typed(body):
    applied = normalize_applied_filters(body["applied_filters"])
    assert applied["budget_max"] == 15_000_000 and applied["budget_min"] is None
    assert applied["bedrooms"] == [2] and applied["type"] == ["apartment"]
    assert normalize_applied_filters(None) == {}


def test_matched_locality_accepts_object_and_string():
    assert matched_locality_name({"name": "Whitefield, Bangalore", "locality_id": "368"}) == "Whitefield, Bangalore"
    assert matched_locality_name("Whitefield") == "Whitefield"
    assert matched_locality_name(None) is None and matched_locality_name(12) is None


# ---------------------------------------------------------------------------
# Location splitting and request parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Whitefield, Bangalore", ("bangalore", "Whitefield")),
        ("Whitefield, Bengaluru", ("bangalore", "Whitefield")),
        ("Whitefield Bangalore", ("bangalore", "Whitefield")),
        ("Bangalore Whitefield", ("bangalore", "Whitefield")),
        ("Kolkata, Salt Lake", ("kolkata", "Salt Lake")),
        ("Pune", ("pune", None)),
        ("Gurugram", ("gurgaon", None)),
        ("Whitefield", (None, "Whitefield")),
        ("Indiranagar, Somewhere", (None, "Indiranagar Somewhere")),
        ("Jaipur", (None, "Jaipur")),  # a city not in the documentation: sent as given
        ("", (None, None)),
        (None, (None, None)),
    ],
)
def test_split_location(text, expected):
    assert split_location(text) == expected


@pytest.mark.asyncio
async def test_only_documented_parameters_are_sent_and_key_is_a_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    provider = _provider(handler)
    criteria = dataclasses.replace(WHITEFIELD, min_budget=Decimal("5000000.5"))
    await provider.search(criteria)
    (request,) = seen
    assert request.url.path.endswith("/99acres/v1/search")
    assert dict(request.url.params) == {
        "purpose": "sale",
        "city": "bangalore",
        "location": "Whitefield",
        "type": "apartment",
        "bedrooms": "2",
        "budget_min": "5000000",   # rounded down: the upstream window is widened
        "budget_max": "15000000",
    }
    for banned in ("page", "fresh", "raw", "fields", "url"):
        assert banned not in request.url.params
    assert request.headers["x-api-key"] == SECRET
    assert SECRET not in str(request.url)


@pytest.mark.asyncio
async def test_search_without_a_location_sends_nothing():
    calls = []
    provider = _provider(lambda r: calls.append(r) or httpx.Response(200, json={}))
    with pytest.raises(InventoryRequestError) as info:
        await provider.search(SearchCriteria(max_budget=Decimal(1)))
    assert info.value.request_sent is False and calls == []


@pytest.mark.asyncio
async def test_search_without_a_key_sends_nothing():
    calls = []
    provider = _provider(lambda r: calls.append(r) or httpx.Response(200, json={}), key="")
    with pytest.raises(InventoryConfigError) as info:
        await provider.search(WHITEFIELD)
    assert info.value.request_sent is False and calls == []


# ---------------------------------------------------------------------------
# Provider: the real response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_response_keeps_properties_and_counts_projects(body):
    result = await _provider(_ok(body)).search(WHITEFIELD)
    assert len(result.listings) == 2
    assert {l.ref.kind for l in result.listings} == {RecordKind.PROPERTY}
    assert result.excluded == {"project_records": 25}
    assert result.matched_locality == "Whitefield, Bangalore"
    assert (result.upstream_total, result.upstream_page, result.upstream_next_page) == (1234, 1, 2)
    assert result.cached is True and result.credits_used == 1
    assert result.fetched_at == datetime(2026, 11, 27, 12, 13, 39, 690000, tzinfo=timezone.utc)
    assert all(l.fetched_at == result.fetched_at and l.cached is True for l in result.listings)
    assert result.source == APISH_SOURCE and result.inventory is Inventory.LIVE
    assert SECRET not in json.dumps(dict(result.sent_params))


@pytest.mark.asyncio
async def test_unknown_and_malformed_records_are_counted_not_guessed(body):
    body = copy.deepcopy(body)
    body["results"] = [
        {"kind": "advert", "id": "9"},
        {"id": "no-kind"},
        "not an object",
        _properties(body)[0],
        {"kind": "property", "id": None},
    ]
    result = await _provider(_ok(body)).search(WHITEFIELD)
    assert len(result.listings) == 1
    assert result.excluded == {"unknown_record_kind": 2, "unparseable_records": 2}


@pytest.mark.asyncio
async def test_missing_budget_echo_is_noted(body):
    body = copy.deepcopy(body)
    body["applied_filters"]["budget_max"] = None
    result = await _provider(_ok(body)).search(WHITEFIELD)
    assert any("budget_max" in note for note in result.notes)


@pytest.mark.asyncio
async def test_bad_credits_used_is_unknown(body):
    for value in (None, "1", -1, True, 1.5):
        body = dict(body, credits_used=value)
        assert (await _provider(_ok(body)).search(WHITEFIELD)).credits_used is None


# ---------------------------------------------------------------------------
# Provider: failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "response, error, kind",
    [
        (httpx.Response(401, json={"success": False, "error": "bad key"}), InventoryAuthError, FailureKind.AUTH_FAILURE),
        (httpx.Response(403, json={"success": False, "error": "no"}), InventoryAuthError, FailureKind.AUTH_FAILURE),
        (httpx.Response(402, json={"success": False, "error": "x"}), InventoryQuotaError, FailureKind.QUOTA_EXCEEDED),
        (httpx.Response(429, json={"success": False, "error": "You exceeded your plan limit"}), InventoryQuotaError, FailureKind.QUOTA_EXCEEDED),
        (httpx.Response(429, json={"success": False, "error": "slow down"}), InventoryRateLimitError, FailureKind.RATE_LIMITED),
        (httpx.Response(500, json={"success": False, "error": "boom"}), InventoryAPIError, FailureKind.PROVIDER_ERROR),
        (httpx.Response(200, text="not json"), InventoryResponseError, FailureKind.INVALID_RESPONSE),
        (httpx.Response(200, json=["a list"]), InventoryResponseError, FailureKind.INVALID_RESPONSE),
        (httpx.Response(200, json={"results": "no"}), InventoryResponseError, FailureKind.INVALID_RESPONSE),
        (httpx.Response(200, json={"success": False, "error": "x"}), InventoryAPIError, FailureKind.PROVIDER_ERROR),
    ],
)
@pytest.mark.asyncio
async def test_status_mapping_is_conventional(response, error, kind):
    provider = _provider(lambda request: response)
    with pytest.raises(error) as info:
        await provider.search(WHITEFIELD)
    assert info.value.failure_kind is kind and info.value.request_sent is True


@pytest.mark.asyncio
async def test_http_404_is_no_results_with_credit_left_to_the_caller():
    provider = _provider(lambda r: httpx.Response(404, json={"success": False, "error": "none"}))
    result = await provider.search(WHITEFIELD)
    assert result.listings == () and result.credits_used is None


@pytest.mark.asyncio
async def test_timeout_and_network_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def network(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(InventoryTimeoutError):
        await _provider(timeout).search(WHITEFIELD)
    with pytest.raises(InventoryNetworkError):
        await _provider(network).search(WHITEFIELD)


@pytest.mark.asyncio
async def test_no_retries():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, json={"success": False, "error": "boom"})

    with pytest.raises(InventoryAPIError):
        await _provider(handler).search(WHITEFIELD)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_key_never_appears_in_errors_logs_or_repr(caplog, body):
    leaky = lambda request: httpx.Response(
        500, json={"success": False, "error": f"invalid key {SECRET} supplied"}
    )
    provider = _provider(leaky)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(InventoryAPIError) as info:
            await provider.search(WHITEFIELD)
        await _provider(_ok(body)).search(WHITEFIELD)
    assert SECRET not in str(info.value) and "[redacted]" in str(info.value)
    assert SECRET not in repr(provider)
    assert SECRET not in caplog.text


# ---------------------------------------------------------------------------
# Local checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "price, price_max, lo, hi, expected",
    [
        (9_660_000, 10_200_000, None, 15_000_000, MATCH),
        (9_660_000, 10_200_000, None, 9_000_000, MISMATCH),
        (9_660_000, 10_200_000, None, 9_660_000, MATCH),
        (9_660_000, 10_200_000, 11_000_000, None, MISMATCH),
        (9_660_000, 10_200_000, 10_200_000, None, MATCH),
        (9_660_000, 10_200_000, 9_000_000, 9_500_000, MISMATCH),
        (9_660_000, 10_200_000, 10_000_000, 12_000_000, MATCH),
        (14_600_000, None, None, 15_000_000, MATCH),
        (None, 12_000_000, None, 15_000_000, MATCH),
        (None, None, None, 15_000_000, UNKNOWN),
        (10, 5, None, 15_000_000, UNKNOWN),       # inconsistent range
        (9_660_000, 10_200_000, None, None, NOT_REQUESTED),
    ],
)
def test_check_budget(price, price_max, lo, hi, expected):
    listing = _listing(
        price=None if price is None else Decimal(price),
        price_max=None if price_max is None else Decimal(price_max),
    )
    to_dec = lambda v: None if v is None else Decimal(v)
    assert check_budget(listing, to_dec(lo), to_dec(hi)) == expected


def test_check_bedrooms_and_configuration_parsing():
    assert check_bedrooms(_listing(bedrooms=2), "2 BHK") == MATCH
    assert check_bedrooms(_listing(bedrooms=3), "2 BHK") == MISMATCH
    assert check_bedrooms(_listing(bedrooms=None), "2 BHK") == UNKNOWN
    assert check_bedrooms(_listing(bedrooms=2), "Studio") == UNKNOWN
    assert check_bedrooms(_listing(bedrooms=2), None) == NOT_REQUESTED
    for text in ("2 BHK", "2BHK", "2-BHK", "2–BHK", " 2 bhk "):
        assert bedrooms_from_configuration(text) == 2
    assert bedrooms_from_configuration("Studio") is None


@pytest.mark.parametrize(
    "month, cutoff, expected",
    [
        ((2028, 3), date(2027, 12, 31), MISMATCH),   # whole month after the cutoff
        ((2028, 3), date(2028, 12, 31), MATCH),      # whole month before the cutoff
        ((2028, 3), date(2028, 3, 31), MATCH),       # cutoff is the month's last day
        ((2028, 3), date(2028, 3, 15), UNKNOWN),     # cutoff inside the month
        ((2028, 3), date(2028, 3, 1), UNKNOWN),
        ((2028, 3), date(2028, 2, 29), MISMATCH),
        (None, date(2028, 12, 31), UNKNOWN),         # unknown is not a mismatch
        ((2028, 3), None, NOT_REQUESTED),
    ],
)
def test_check_possession_never_invents_a_day(month, cutoff, expected):
    assert check_possession(_listing(possession_month=month), cutoff) == expected


def test_check_possession_exact_date_for_demo_records():
    assert check_possession(_listing(possession_date=date(2027, 6, 30)), date(2027, 6, 30)) == MATCH
    assert check_possession(_listing(possession_date=date(2027, 7, 1)), date(2027, 6, 30)) == MISMATCH


def test_type_is_never_a_mismatch():
    apartment = _listing(property_type_text="Residential Apartment")
    assert check_property_type(apartment, "Apartment") == MATCH
    assert check_property_type(_listing(property_type_text="Villa"), "apartment") == UNKNOWN
    assert check_property_type(apartment, "villa") == UNKNOWN
    assert check_property_type(apartment, "castle") == UNKNOWN
    assert check_property_type(apartment, None) == NOT_REQUESTED
    assert documented_type("Independent House") == "independent_house"
    assert documented_type("flat") is None


# ---------------------------------------------------------------------------
# Identity and de-duplication
# ---------------------------------------------------------------------------

def test_identity_is_source_plus_kind_plus_id():
    prop = SourceRef("apish_99acres", RecordKind.PROPERTY, "9414")
    project = SourceRef("apish_99acres", RecordKind.PROJECT, "9414")
    other = SourceRef("sqlite", RecordKind.PROPERTY, "9414")
    assert len({prop.key, project.key, other.key}) == 3


def test_dedupe_is_source_and_kind_aware():
    a = _listing(external_id="1")
    same = _listing(external_id="1", title="again")
    other_kind = dataclasses.replace(a, ref=SourceRef(APISH_SOURCE, RecordKind.PROJECT, "1"))
    other_source = dataclasses.replace(a, ref=SourceRef("sqlite", RecordKind.PROPERTY, "1"))
    kept, removed = dedupe([a, same, other_kind, other_source])
    assert removed == 1 and kept == [a, other_kind, other_source]


# ---------------------------------------------------------------------------
# Credits
# ---------------------------------------------------------------------------

def test_credit_policy_and_ledger():
    assert CreditPolicy().limit_per_conversation == 3
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValueError):
            CreditPolicy(bad)
    ledger = InMemoryCreditLedger()
    assert ledger.used("c1") == 0
    assert ledger.add("c1", 2) == 2 and ledger.add("c1", 1) == 3
    assert ledger.used("c2") == 0
    with pytest.raises(ValueError):
        ledger.add("c1", -1)


# ---------------------------------------------------------------------------
# The search service
# ---------------------------------------------------------------------------

def _live_result(body, **kw):
    listings = tuple(normalize_property_record(r) for r in _properties(body))
    return ProviderResult(source=APISH_SOURCE, inventory=Inventory.LIVE, listings=listings, **kw)


def _service(live, demo=None, limit=3):
    providers = {Inventory.LIVE: live}
    if demo is not None:
        providers[Inventory.DEMO] = demo
    return InventorySearchService(providers, policy=CreditPolicy(limit))


@pytest.mark.asyncio
async def test_one_search_is_one_call_and_costs_what_the_source_reports(body):
    live = MockPropertyProvider(_live_result(body, credits_used=2))
    outcome = await _service(live).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert len(live.calls) == 1
    assert outcome.status == "ok" and outcome.credits_charged == 2
    assert outcome.credits_used_total == 2 and outcome.credits_limit == 3


@pytest.mark.asyncio
async def test_credits_are_assumed_to_cost_one_when_not_reported(body):
    live = MockPropertyProvider(_live_result(body, credits_used=None))
    outcome = await _service(live).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert outcome.credits_charged == 1


@pytest.mark.asyncio
async def test_limit_stops_the_fourth_search_without_calling_the_provider(body):
    live = MockPropertyProvider(_live_result(body, credits_used=1))
    service = _service(live, limit=3)
    for _ in range(3):
        assert (await service.search(Inventory.LIVE, "c1", WHITEFIELD)).status == "ok"
    blocked = await service.search(Inventory.LIVE, "c1", WHITEFIELD)
    assert blocked.status == "credit_limit_reached" and blocked.result is None
    assert blocked.credits_used_total == 3 and len(live.calls) == 3
    # Another conversation has its own allowance.
    assert (await service.search(Inventory.LIVE, "c2", WHITEFIELD)).status == "ok"


@pytest.mark.asyncio
async def test_limit_zero_disables_live_search(body):
    live = MockPropertyProvider(_live_result(body))
    outcome = await _service(live, limit=0).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert outcome.status == "credit_limit_reached" and live.calls == []


@pytest.mark.parametrize(
    "error, charge",
    [
        (InventoryAuthError("x"), 0),
        (InventoryQuotaError("x"), 0),
        (InventoryRateLimitError("x"), 0),
        (InventoryTimeoutError("x"), 1),
        (InventoryNetworkError("x"), 1),
        (InventoryResponseError("x"), 1),
        (InventoryAPIError(500, "x"), 1),
    ],
)
@pytest.mark.asyncio
async def test_failed_calls_are_counted_unless_refused(error, charge):
    live = MockPropertyProvider(error)
    outcome = await _service(live).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert outcome.status == "failed" and outcome.failure_kind is error.failure_kind
    assert outcome.credits_charged == charge and outcome.result is None
    assert outcome.message and SECRET not in outcome.message


@pytest.mark.asyncio
async def test_unsent_requests_cost_nothing():
    live = MockPropertyProvider(InventoryRequestError("no location"))
    outcome = await _service(live).search(Inventory.LIVE, "c1", SearchCriteria())
    assert outcome.status == "invalid_request" and outcome.credits_charged == 0
    live = MockPropertyProvider(InventoryConfigError("no key"))
    outcome = await _service(live).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert outcome.status == "not_configured" and outcome.credits_charged == 0
    assert "key" not in outcome.message.lower()


@pytest.mark.asyncio
async def test_a_live_failure_never_falls_back_to_demo(body):
    demo = MockPropertyProvider(
        ProviderResult(source="sqlite", inventory=Inventory.DEMO,
                       listings=(_listing(external_id="d1", inventory=Inventory.DEMO),)),
        source_name="sqlite", inventory=Inventory.DEMO,
    )
    live = MockPropertyProvider(InventoryTimeoutError("slow"))
    outcome = await _service(live, demo).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert outcome.status == "failed" and outcome.result is None
    assert outcome.fallback_used is False and demo.calls == []


@pytest.mark.asyncio
async def test_demo_and_live_are_separate_and_demo_costs_nothing(body):
    demo_listing = _listing(external_id="d1", inventory=Inventory.DEMO)
    demo = MockPropertyProvider(
        ProviderResult(source="sqlite", inventory=Inventory.DEMO, listings=(demo_listing,)),
        source_name="sqlite", inventory=Inventory.DEMO,
    )
    live = MockPropertyProvider(_live_result(body, credits_used=1))
    service = _service(live, demo)
    outcome = await service.search(Inventory.DEMO, "c1", WHITEFIELD)
    assert outcome.inventory is Inventory.DEMO and outcome.source == "sqlite"
    assert outcome.credits_charged == 0 and outcome.credits_used_total is None
    assert live.calls == [] and outcome.result.listings == (demo_listing,)
    with pytest.raises(ValueError):
        await _service(live).search(Inventory.DEMO, "c1", WHITEFIELD)


@pytest.mark.asyncio
async def test_budget_is_enforced_locally_even_when_upstream_ignores_it(body):
    props = [normalize_property_record(r) for r in _properties(body)]
    too_dear = dataclasses.replace(props[0], ref=SourceRef(APISH_SOURCE, RecordKind.PROPERTY, "x1"),
                                   price=Decimal(20_000_000), price_max=Decimal(25_000_000))
    no_price = dataclasses.replace(props[0], ref=SourceRef(APISH_SOURCE, RecordKind.PROPERTY, "x2"),
                                   price=None, price_max=None)
    result = ProviderResult(source=APISH_SOURCE, inventory=Inventory.LIVE,
                            listings=(props[0], props[1], too_dear, no_price), credits_used=1)
    outcome = await _service(MockPropertyProvider(result)).search(Inventory.LIVE, "c1", WHITEFIELD)
    kept = outcome.result.listings
    assert [l.ref.external_id for l in kept] == ["94869686", props[1].ref.external_id]
    assert outcome.result.excluded == {"budget_mismatch": 1, "budget_unknown": 1}
    assert all(l.checks["budget"] == MATCH for l in kept)


@pytest.mark.asyncio
async def test_possession_unknown_is_kept_and_flagged_mismatch_is_dropped(body):
    mar_2028, dec_2029 = [normalize_property_record(r) for r in _properties(body)]
    unknown = dataclasses.replace(
        mar_2028, ref=SourceRef(APISH_SOURCE, RecordKind.PROPERTY, "x9"),
        possession_text=None, possession_month=None,
    )
    result = ProviderResult(source=APISH_SOURCE, inventory=Inventory.LIVE,
                            listings=(mar_2028, dec_2029, unknown), credits_used=1)
    service = _service(MockPropertyProvider(result), limit=10)

    early = dataclasses.replace(WHITEFIELD, possession_before=date(2027, 12, 31))
    out = await service.search(Inventory.LIVE, "c1", early)
    assert [l.ref.external_id for l in out.result.listings] == ["x9"]       # only the unknown one
    assert [l.checks["possession"] for l in out.result.listings] == [UNKNOWN]
    assert out.result.excluded == {"possession_mismatch": 2}

    mid = dataclasses.replace(WHITEFIELD, possession_before=date(2028, 12, 31))
    out = await service.search(Inventory.LIVE, "c1", mid)
    assert [l.checks["possession"] for l in out.result.listings] == [MATCH, UNKNOWN]
    assert out.result.excluded == {"possession_mismatch": 1}                # Dec 2029 is after

    none = await service.search(Inventory.LIVE, "c1", WHITEFIELD)
    assert len(none.result.listings) == 3
    assert {l.checks["possession"] for l in none.result.listings} == {NOT_REQUESTED}


@pytest.mark.asyncio
async def test_bedroom_mismatch_is_dropped(body):
    props = [normalize_property_record(r) for r in _properties(body)]
    three = dataclasses.replace(props[0], ref=SourceRef(APISH_SOURCE, RecordKind.PROPERTY, "x3"), bedrooms=3)
    result = ProviderResult(source=APISH_SOURCE, inventory=Inventory.LIVE, listings=(three, props[1]), credits_used=1)
    out = await _service(MockPropertyProvider(result)).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert len(out.result.listings) == 1 and out.result.excluded == {"bedrooms_mismatch": 1}


@pytest.mark.asyncio
async def test_duplicates_are_removed_and_counted(body):
    props = [normalize_property_record(r) for r in _properties(body)]
    result = ProviderResult(source=APISH_SOURCE, inventory=Inventory.LIVE,
                            listings=(props[0], props[0], props[1]), credits_used=1)
    out = await _service(MockPropertyProvider(result)).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert len(out.result.listings) == 2 and out.result.excluded == {"duplicates": 1}


@pytest.mark.asyncio
async def test_empty_live_result_is_no_results_not_a_demo_answer(body):
    provider = _provider(lambda r: httpx.Response(404, json={"success": False, "error": "none"}))
    out = await _service(provider).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert out.status == "no_results" and out.credits_charged == 1   # a 404 lookup costs a credit
    assert out.fallback_used is False and out.result.listings == ()


@pytest.mark.asyncio
async def test_end_to_end_with_the_real_response(body):
    provider = _provider(_ok(body))
    out = await _service(provider).search(Inventory.LIVE, "c1", WHITEFIELD)
    assert out.status == "ok" and out.credits_charged == 1 and out.credits_used_total == 1
    assert out.result.excluded == {"project_records": 25}
    assert [l.ref.kind for l in out.result.listings] == [RecordKind.PROPERTY] * 2
    assert all(l.source_label.startswith("99acres listing (via the API.SH") for l in out.result.listings)
    assert all("official" not in l.source_label.lower() for l in out.result.listings)

    data = out.to_dict()
    text = json.dumps(data)                      # JSON-safe
    assert data["inventory"] == "live" and data["fallback_used"] is False
    first = data["result"]["listings"][0]
    assert first["ref"] == {"source": "apish_99acres", "kind": "property", "external_id": "94869686"}
    assert first["possession_month"] == [2028, 3] and first["price"] == 9660000
    assert "raw" not in text and SECRET not in text


# ---------------------------------------------------------------------------
# Demo (SQLite) provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sqlite_provider_labels_demo_records_and_matches_the_existing_search(dbsession):
    dbsession.add_all([
        Property(location="Whitefield, Bangalore", property_type="apartment", configuration="2 BHK",
                 price_min=5_000_000, price_max=6_000_000, listing_price=5_500_000,
                 possession_date=datetime(2027, 6, 30), source_name="synthetic", source_record_id="s-1"),
        Property(location="Whitefield, Bangalore", property_type="apartment", configuration="3 BHK",
                 price_min=9_000_000, price_max=9_500_000, source_name="kaggle_house_price_india",
                 source_record_id="k-7"),
        Property(location="Pune", property_type="apartment", configuration="2 BHK",
                 price_min=4_000_000, price_max=4_500_000),
    ])
    dbsession.flush()
    service = PropertySearchService(session_factory=lambda: dbsession, close_session=False)
    demo = SQLiteProvider(service)
    live = MockPropertyProvider()
    out = await InventorySearchService({Inventory.DEMO: demo, Inventory.LIVE: live}).search(
        Inventory.DEMO, "c1", SearchCriteria(location="Whitefield", configuration="2 BHK"))

    direct_total, direct_rows = service.search(location="Whitefield", configuration="2 BHK")
    assert out.status == "ok" and out.result.upstream_total == direct_total == 1
    (item,) = out.result.listings
    assert item.inventory is Inventory.DEMO and item.source_label == "Demo data (synthetic)"
    assert item.ref == SourceRef("synthetic", RecordKind.PROPERTY, "s-1")
    assert item.is_synthetic and item.price_note == "demo data" and item.bedrooms == 2
    assert item.possession_date == date(2027, 6, 30) and item.price == Decimal(5_000_000)
    assert out.credits_charged == 0 and live.calls == []

    kaggle = await demo.search(SearchCriteria(location="Whitefield", configuration="3 BHK"))
    assert kaggle.listings[0].source_label == "Historical dataset (Kaggle)"
    assert kaggle.listings[0].price_note == "historical"
    plain = await demo.search(SearchCriteria(location="Pune"))
    assert plain.listings[0].ref.source == "sqlite" and plain.listings[0].ref.external_id.startswith("row-")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_credit_limit_setting_is_validated(monkeypatch):
    from backend import config

    monkeypatch.delenv("LIVE_CREDITS_PER_CONVERSATION", raising=False)
    assert config._non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3) == 3
    monkeypatch.setenv("LIVE_CREDITS_PER_CONVERSATION", " 5 ")
    assert config._non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3) == 5
    monkeypatch.setenv("LIVE_CREDITS_PER_CONVERSATION", "0")
    assert config._non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3) == 0
    monkeypatch.setenv("LIVE_CREDITS_PER_CONVERSATION", "")
    assert config._non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3) == 3
    for bad in ("-1", "many", "1.5"):
        monkeypatch.setenv("LIVE_CREDITS_PER_CONVERSATION", bad)
        with pytest.raises(ValueError):
            config._non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3)


def test_env_example_documents_the_settings_without_a_key():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert "APISH_API_KEY=" in lines           # present, and empty
    assert "LIVE_CREDITS_PER_CONVERSATION=3" in lines
    assert "APISH_BASE_URL=https://api.api.sh/99acres/v1" in lines
    from backend import config
    for name in ("APISH_API_KEY", "APISH_BASE_URL", "APISH_TIMEOUT_SECONDS", "LIVE_CREDITS_PER_CONVERSATION"):
        assert name in config.__all__
