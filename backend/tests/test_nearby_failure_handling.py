# -*- coding: utf-8 -*-
"""Tests for honest handling of nearby_amenities failures.

Covers coordinates_unavailable and other Mappls errors, per-category lookup
failures (which must never look like "none found"), the radius actually
used, and the guidance the agent receives.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.orchestrator.providers.mappls_provider import (
    GeocodedLocation,
    NearbyPlace,
    NearbyResult,
    clamp_radius,
)
from backend.orchestrator.services.agent_service import (
    SYSTEM_PROMPT,
    _TOOL_ERROR_HINTS,
    _build_tool_error_hint,
)
from backend.orchestrator.services.tool_executor import execute_nearby_amenities


def _geo(with_coordinates: bool = True) -> GeocodedLocation:
    if with_coordinates:
        return GeocodedLocation(
            latitude=12.9716,
            longitude=77.5946,
            formatted_address="Whitefield, Bengaluru",
        )
    return GeocodedLocation(
        latitude=None,
        longitude=None,
        formatted_address="Whitefield, Bengaluru",
        eloc="ABC123",
    )


def _ok(keyword: str, n: int = 1) -> NearbyResult:
    places = [
        NearbyPlace(
            name=f"{keyword} {i}",
            category="X",
            distance=100 * (i + 1),
            latitude=12.97,
            longitude=77.59,
            address="MG Road",
        )
        for i in range(n)
    ]
    return NearbyResult(keyword=keyword, places=places)


async def _run(geo, nearby=None, radius=3000, categories=None):
    provider = AsyncMock()
    provider.geocode.return_value = geo
    provider.nearby_amenities.return_value = nearby or {}
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=False)
    with patch("backend.config.MAPPLS_API_KEY", "test-key"), patch(
        "backend.config.MAPPLS_BASE_URL", "https://api.test"
    ), patch(
        "backend.orchestrator.providers.mappls_provider.MappslsProvider",
        return_value=provider,
    ):
        result = await execute_nearby_amenities(
            location="Whitefield, Bengaluru", categories=categories, radius=radius
        )
    return result, provider


# ---- coordinates_unavailable -------------------------------------------------

@pytest.mark.asyncio
async def test_coordinates_unavailable_is_a_structured_error_and_skips_search():
    result, provider = await _run(_geo(with_coordinates=False))

    assert result["error"] == "coordinates_unavailable"
    assert result["formatted_address"] == "Whitefield, Bengaluru"
    assert "amenities" not in result
    provider.nearby_amenities.assert_not_called()


# ---- failed categories are never presented as "none found" -------------------

@pytest.mark.asyncio
async def test_failed_category_is_reported_not_shown_as_empty():
    nearby = {
        "hospital": NearbyResult(places=[], keyword="hospital", error="timeout"),
        "school": _ok("school", 2),
    }
    result, _ = await _run(_geo(), nearby, categories=["hospital", "school"])

    assert "error" not in result
    assert result["failed_categories"] == ["hospital"]
    assert "hospital" not in result["amenities"]
    assert len(result["amenities"]["school"]) == 2


@pytest.mark.asyncio
async def test_genuinely_empty_category_stays_an_empty_list():
    nearby = {
        "park": NearbyResult(places=[], keyword="park"),  # lookup worked, none found
        "school": _ok("school"),
    }
    result, _ = await _run(_geo(), nearby, categories=["park", "school"])

    assert result["amenities"]["park"] == []
    assert "failed_categories" not in result


@pytest.mark.asyncio
async def test_all_categories_failed_becomes_an_error():
    nearby = {
        "hospital": NearbyResult(places=[], keyword="hospital", error="timeout"),
        "school": NearbyResult(places=[], keyword="school", error="HTTP 500"),
    }
    result, _ = await _run(_geo(), nearby, categories=["hospital", "school"])

    assert result["error"] == "mappls_api_error"
    assert result["failed_categories"] == ["hospital", "school"]
    assert "amenities" not in result


@pytest.mark.asyncio
async def test_failure_details_are_not_leaked_to_the_model():
    nearby = {
        "hospital": NearbyResult(
            places=[], keyword="hospital", error="https://x?access_token=SECRET"
        ),
        "school": _ok("school"),
    }
    result, _ = await _run(_geo(), nearby, categories=["hospital", "school"])

    assert "SECRET" not in str(result)


# ---- radius reporting --------------------------------------------------------

@pytest.mark.parametrize(
    "requested, effective",
    [(100, 500), (499, 500), (500, 500), (3000, 3000), (10_000, 10_000), (20_000, 10_000)],
)
def test_clamp_radius(requested, effective):
    assert clamp_radius(requested) == effective


@pytest.mark.asyncio
async def test_radius_used_is_reported_and_adjustment_is_explained():
    result, provider = await _run(_geo(), {"school": _ok("school")}, radius=20_000)

    assert result["radius_m"] == 10_000
    assert "20000" in result["radius_note"] and "10000" in result["radius_note"]
    # The caller's value is still passed through; the provider clamps itself.
    assert provider.nearby_amenities.call_args.kwargs["radius"] == 20_000


@pytest.mark.asyncio
async def test_no_radius_note_when_radius_is_supported():
    result, _ = await _run(_geo(), {"school": _ok("school")}, radius=5000)

    assert result["radius_m"] == 5000
    assert "radius_note" not in result


# ---- guidance given to the model --------------------------------------------

@pytest.mark.parametrize(
    "code",
    ["mappls_not_configured", "geocode_failed", "coordinates_unavailable", "mappls_api_error"],
)
def test_every_mappls_error_has_a_hint(code):
    assert code in _TOOL_ERROR_HINTS
    hint = _build_tool_error_hint({"error": code})
    assert hint.startswith(f"Tool error: {code}.")
    assert len(hint) > len(f"Tool error: {code}.") + 20


def test_coordinates_unavailable_hint_forbids_guessing_and_offers_wider_area():
    hint = _build_tool_error_hint({"error": "coordinates_unavailable"})
    assert "Do not estimate or invent" in hint
    assert "wider area" in hint


def test_prompt_covers_amenity_failures():
    for term in (
        "failed_categories",
        "radius_note",
        "coordinates_unavailable",
        "Never say there are none",
    ):
        assert term in SYSTEM_PROMPT, term
