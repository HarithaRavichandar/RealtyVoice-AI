# -*- coding: utf-8 -*-
"""Integration tests for the nearby_amenities tool end-to-end flow.

Tests cover:
  - Sync executor returning the _async_pending marker
  - execute_nearby_amenities() async function (geocode + nearby)
  - Error paths: missing API key, geocode failure, Mappls errors
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.orchestrator.tools.schemas import NearbyAmenitiesRequest
from backend.orchestrator.services.tool_executor import (
    execute_tool,
    execute_nearby_amenities,
)
from backend.orchestrator.providers.mappls_provider import (
    GeocodedLocation,
    NearbyPlace,
    NearbyResult,
    MappslsAPIError,
    MappslsAuthenticationError,
    MappslsTimeoutError,
)


# ---- Helpers ----

def _make_session():
    return MagicMock()


def _geo():
    return GeocodedLocation(
        latitude=12.9716,
        longitude=77.5946,
        formatted_address="Bengaluru, Karnataka",
    )


def _nearby_result(keyword: str = "hospital"):
    return NearbyResult(
        keyword=keyword,
        places=[
            NearbyPlace(
                name="Apollo Hospital",
                category="HSPTL",
                distance=450,
                latitude=12.972,
                longitude=77.595,
                address="123 MG Road",
            ),
        ],
    )


# =================================================================
# Test: sync executor returns _async_pending marker
# =================================================================

class TestSyncExecutorMarker:

    def test_nearby_amenities_returns_async_pending(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={"location": "Whitefield, Bengaluru"},
        )
        assert result["_async_pending"] is True
        assert result["tool"] == "nearby_amenities"
        assert result["location"] == "Whitefield, Bengaluru"
        assert result["categories"] is None
        assert result["radius"] == 3000

    def test_nearby_amenities_with_custom_params(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={
                "location": "Koramangala",
                "categories": ["hospital", "school"],
                "radius": 5000,
            },
        )
        assert result["_async_pending"] is True
        assert result["categories"] == ["hospital", "school"]
        assert result["radius"] == 5000

    def test_nearby_amenities_validation_error_empty_location(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={"location": ""},
        )
        assert result.get("error") == "validation"

    def test_nearby_amenities_validation_error_missing_location(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={},
        )
        assert result.get("error") == "validation"

    def test_nearby_amenities_validation_error_extra_field(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={"location": "Indiranagar", "bogus": True},
        )
        assert result.get("error") == "validation"

    def test_nearby_amenities_radius_too_small(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={"location": "Indiranagar", "radius": 50},
        )
        assert result.get("error") == "validation"

    def test_nearby_amenities_radius_too_large(self):
        result = execute_tool(
            session=_make_session(),
            tool_name="nearby_amenities",
            payload={"location": "Indiranagar", "radius": 100000},
        )
        assert result.get("error") == "validation"


# =================================================================
# Test: execute_nearby_amenities async function
# =================================================================

class TestExecuteNearbyAmenities:

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_success(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.return_value = _geo()
        mock_provider.nearby_amenities.return_value = {
            "hospital": _nearby_result("hospital"),
        }
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(
                location="Whitefield, Bengaluru",
            )

        assert result["tool"] == "nearby_amenities"
        assert result["location"] == "Whitefield, Bengaluru"
        assert result["coordinates"]["latitude"] == 12.9716
        assert result["coordinates"]["longitude"] == 77.5946
        assert "hospital" in result["amenities"]
        assert result["amenities"]["hospital"][0]["name"] == "Apollo Hospital"
        assert result["amenities"]["hospital"][0]["distance_m"] == 450

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "")
    async def test_missing_api_key(self):
        result = await execute_nearby_amenities(location="Whitefield")
        assert result["error"] == "mappls_not_configured"

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_geocode_returns_none(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.return_value = None
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(location="Nonexistent Place")

        assert result["error"] == "geocode_failed"

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_mappls_api_error(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.side_effect = MappslsAPIError("Server error", 500)
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(location="Whitefield")

        assert result["error"] == "mappls_api_error"

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_auth_error(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.side_effect = MappslsAuthenticationError()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(location="Whitefield")

        assert result["error"] == "mappls_api_error"

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_timeout_error(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.side_effect = MappslsTimeoutError()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(location="Whitefield")

        assert result["error"] == "mappls_api_error"

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_custom_categories_and_radius(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.return_value = _geo()
        mock_provider.nearby_amenities.return_value = {
            "school": _nearby_result("school"),
            "park": _nearby_result("park"),
        }
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(
                location="Koramangala",
                categories=["school", "park"],
                radius=5000,
            )

        assert "school" in result["amenities"]
        assert "park" in result["amenities"]
        mock_provider.nearby_amenities.assert_called_once_with(
            latitude=12.9716,
            longitude=77.5946,
            categories=["school", "park"],
            radius=5000,
        )

    @pytest.mark.asyncio
    @patch("backend.config.MAPPLS_API_KEY", "test-key")
    @patch("backend.config.MAPPLS_BASE_URL", "https://api.test")
    async def test_multiple_amenity_types_with_places(self):
        mock_provider = AsyncMock()
        mock_provider.geocode.return_value = _geo()
        mock_provider.nearby_amenities.return_value = {
            "hospital": NearbyResult(keyword="hospital", places=[
                NearbyPlace(name="Apollo", category="HSPTL", distance=400,
                            latitude=12.97, longitude=77.59, address="MG Rd"),
                NearbyPlace(name="Manipal", category="HSPTL", distance=800,
                            latitude=12.98, longitude=77.60, address="HAL Rd"),
            ]),
            "metro": NearbyResult(keyword="metro", places=[]),
        }
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "backend.orchestrator.providers.mappls_provider.MappslsProvider",
            return_value=mock_provider,
        ):
            result = await execute_nearby_amenities(location="Whitefield")

        assert len(result["amenities"]["hospital"]) == 2
        assert result["amenities"]["metro"] == []
