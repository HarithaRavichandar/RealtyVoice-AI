"""Tests for MappslsProvider — geocoding and nearby-places search.

Uses httpx MockTransport to simulate Mappls API responses without
hitting the network.
"""

from __future__ import annotations

import json
import pytest
import httpx

from backend.orchestrator.providers.mappls_provider import (
    MappslsProvider,
    GeocodedLocation,
    NearbyPlace,
    NearbyResult,
    AMENITY_CATEGORIES,
    MappslsAuthenticationError,
    MappslsRateLimitError,
    MappslsTimeoutError,
    MappslsAPIError,
    MappslsResponseError,
    DEFAULT_BASE_URL,
    DEFAULT_RADIUS,
    DEFAULT_RESULT_LIMIT,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_GEOCODE_RESPONSE = {
    "copResults": {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "formatted_address": "Bengaluru, Karnataka, India",
        "eLoc": "ABC123",
        "locality": "MG Road",
        "city": "Bengaluru",
        "state": "Karnataka",
        "pincode": "560001",
    }
}

SAMPLE_NEARBY_RESPONSE = {
    "suggestedLocations": [
        {
            "placeName": "Apollo Hospital",
            "categoryCode": "HSPTL",
            "distance": 500,
            "latitude": 12.972,
            "longitude": 77.595,
            "placeAddress": "Bannerghatta Road, Bengaluru",
            "eLoc": "XYZ789",
            "phone": "+91-80-12345678",
        },
        {
            "placeName": "Fortis Hospital",
            "categoryCode": "HSPTL",
            "distance": 1200,
            "latitude": 12.973,
            "longitude": 77.596,
            "placeAddress": "Richmond Road, Bengaluru",
            "eLoc": "DEF456",
            "phone": "",
        },
    ]
}


def make_transport(
    status: int = 200,
    body: dict | str | None = None,
    *,
    timeout: bool = False,
    connect_error: bool = False,
):
    """Create a mock transport returning a fixed response."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if timeout:
            raise httpx.ReadTimeout("mock timeout")
        if connect_error:
            raise httpx.ConnectError("mock connection refused")

        content = b""
        if body is not None:
            if isinstance(body, dict):
                content = json.dumps(body).encode()
            else:
                content = body.encode() if isinstance(body, str) else body

        return httpx.Response(
            status_code=status,
            content=content,
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


def make_provider(transport, **kwargs):
    """Build a provider backed by a mock transport."""
    client = httpx.AsyncClient(transport=transport)
    return MappslsProvider(
        api_key="test-token-123",
        client=client,
        **kwargs,
    )


# ===================================================================
# Init
# ===================================================================


class TestMappslsProviderInit:
    """Initialization and configuration."""

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            MappslsProvider(api_key="")

    def test_requires_non_whitespace_key(self):
        with pytest.raises(ValueError, match="api_key"):
            MappslsProvider(api_key="   ")

    def test_strips_base_url_trailing_slash(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport, base_url="https://example.com/api/")
        assert provider._base_url == "https://example.com/api"

    def test_default_base_url(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)
        assert provider._base_url == DEFAULT_BASE_URL

    def test_custom_timeout(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport, timeout=30.0)
        assert provider._timeout == 30.0

    def test_external_client_not_owned(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)
        assert provider._owns_client is False

    def test_own_client_created_when_none(self):
        provider = MappslsProvider(api_key="key123")
        assert provider._owns_client is True


# ===================================================================
# Context Manager
# ===================================================================


class TestContextManager:
    """Async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)
        async with provider as p:
            assert p is provider

    @pytest.mark.asyncio
    async def test_external_client_not_closed(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        client = httpx.AsyncClient(transport=transport)
        provider = MappslsProvider(api_key="key", client=client)
        await provider.close()
        # External client should still be open.
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.asyncio
    async def test_owned_client_closed(self):
        provider = MappslsProvider(api_key="key")
        client = provider._client
        await provider.close()
        assert client.is_closed


# ===================================================================
# Geocode
# ===================================================================


class TestGeocode:
    """Geocoding API tests."""

    @pytest.mark.asyncio
    async def test_geocode_success(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)

        result = await provider.geocode("Bengaluru")

        assert isinstance(result, GeocodedLocation)
        assert result.latitude == 12.9716
        assert result.longitude == 77.5946
        assert result.city == "Bengaluru"
        assert result.state == "Karnataka"
        assert result.pincode == "560001"
        assert result.eloc == "ABC123"

    @pytest.mark.asyncio
    async def test_geocode_empty_address_raises(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)

        with pytest.raises(ValueError, match="address"):
            await provider.geocode("")

    @pytest.mark.asyncio
    async def test_geocode_whitespace_address_raises(self):
        transport = make_transport(body=SAMPLE_GEOCODE_RESPONSE)
        provider = make_provider(transport)

        with pytest.raises(ValueError, match="address"):
            await provider.geocode("   ")

    @pytest.mark.asyncio
    async def test_geocode_no_results(self):
        transport = make_transport(body={"copResults": {}})
        provider = make_provider(transport)

        result = await provider.geocode("xyznonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_geocode_list_response(self):
        """Some endpoints return a list of results."""
        body = {
            "copResults": [
                {
                    "latitude": 12.97,
                    "longitude": 77.59,
                    "formatted_address": "Test",
                }
            ]
        }
        transport = make_transport(body=body)
        provider = make_provider(transport)

        result = await provider.geocode("Test")
        assert result is not None
        assert result.latitude == 12.97

    @pytest.mark.asyncio
    async def test_geocode_empty_list(self):
        transport = make_transport(body={"copResults": []})
        provider = make_provider(transport)

        result = await provider.geocode("nothing")
        assert result is None

    @pytest.mark.asyncio
    async def test_geocode_auth_error(self):
        transport = make_transport(status=401)
        provider = make_provider(transport)

        with pytest.raises(MappslsAuthenticationError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_rate_limit(self):
        transport = make_transport(status=429)
        provider = make_provider(transport)

        with pytest.raises(MappslsRateLimitError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_timeout(self):
        transport = make_transport(timeout=True)
        provider = make_provider(transport)

        with pytest.raises(MappslsTimeoutError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_connection_error(self):
        transport = make_transport(connect_error=True)
        provider = make_provider(transport)

        with pytest.raises(MappslsAPIError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_server_error(self):
        transport = make_transport(status=500, body="Internal Server Error")
        provider = make_provider(transport)

        with pytest.raises(MappslsAPIError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_invalid_json(self):
        transport = make_transport(body="not json at all {{{")
        provider = make_provider(transport)

        with pytest.raises(MappslsResponseError):
            await provider.geocode("test")

    @pytest.mark.asyncio
    async def test_geocode_missing_coordinates(self):
        body = {"copResults": {"formatted_address": "Some place"}}
        transport = make_transport(body=body)
        provider = make_provider(transport)

        result = await provider.geocode("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_geocode_alt_field_names(self):
        """Handles 'lat'/'lng' instead of 'latitude'/'longitude'."""
        body = {
            "copResults": {
                "lat": 13.0,
                "lng": 77.6,
                "formatted_address": "Alt format",
                "area": "Whitefield",
                "district": "Bengaluru Urban",
            }
        }
        transport = make_transport(body=body)
        provider = make_provider(transport)

        result = await provider.geocode("whitefield")
        assert result is not None
        assert result.latitude == 13.0
        assert result.longitude == 77.6
        assert result.locality == "Whitefield"
        assert result.city == "Bengaluru Urban"


# ===================================================================
# Nearby
# ===================================================================


class TestNearby:
    """Nearby places API tests."""

    @pytest.mark.asyncio
    async def test_nearby_success(self):
        transport = make_transport(body=SAMPLE_NEARBY_RESPONSE)
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "HSPTL")

        assert isinstance(result, NearbyResult)
        assert result.keyword == "HSPTL"
        assert len(result.places) == 2
        assert result.places[0].name == "Apollo Hospital"
        assert result.places[0].distance == 500
        assert result.places[1].name == "Fortis Hospital"

    @pytest.mark.asyncio
    async def test_nearby_place_fields(self):
        transport = make_transport(body=SAMPLE_NEARBY_RESPONSE)
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "HSPTL")
        place = result.places[0]

        assert isinstance(place, NearbyPlace)
        assert place.category == "HSPTL"
        assert place.latitude == 12.972
        assert place.longitude == 77.595
        assert place.address == "Bannerghatta Road, Bengaluru"
        assert place.eloc == "XYZ789"
        assert place.phone == "+91-80-12345678"

    @pytest.mark.asyncio
    async def test_nearby_empty_keywords_raises(self):
        transport = make_transport(body=SAMPLE_NEARBY_RESPONSE)
        provider = make_provider(transport)

        with pytest.raises(ValueError, match="keywords"):
            await provider.nearby(12.97, 77.59, "")

    @pytest.mark.asyncio
    async def test_nearby_no_results(self):
        transport = make_transport(body={"suggestedLocations": []})
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "NONEXISTENT")
        assert result.places == []

    @pytest.mark.asyncio
    async def test_nearby_missing_locations_key(self):
        transport = make_transport(body={})
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "test")
        assert result.places == []

    @pytest.mark.asyncio
    async def test_nearby_respects_limit(self):
        many_places = {
            "suggestedLocations": [
                {
                    "placeName": f"Place {i}",
                    "categoryCode": "TEST",
                    "distance": i * 100,
                    "latitude": 12.97,
                    "longitude": 77.59,
                }
                for i in range(20)
            ]
        }
        transport = make_transport(body=many_places)
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "TEST", limit=5)
        assert len(result.places) == 5

    @pytest.mark.asyncio
    async def test_nearby_auth_error(self):
        transport = make_transport(status=401)
        provider = make_provider(transport)

        with pytest.raises(MappslsAuthenticationError):
            await provider.nearby(12.97, 77.59, "HSPTL")

    @pytest.mark.asyncio
    async def test_nearby_rate_limit(self):
        transport = make_transport(status=429)
        provider = make_provider(transport)

        with pytest.raises(MappslsRateLimitError):
            await provider.nearby(12.97, 77.59, "HSPTL")

    @pytest.mark.asyncio
    async def test_nearby_timeout(self):
        transport = make_transport(timeout=True)
        provider = make_provider(transport)

        with pytest.raises(MappslsTimeoutError):
            await provider.nearby(12.97, 77.59, "HSPTL")

    @pytest.mark.asyncio
    async def test_nearby_connection_error(self):
        transport = make_transport(connect_error=True)
        provider = make_provider(transport)

        with pytest.raises(MappslsAPIError):
            await provider.nearby(12.97, 77.59, "HSPTL")

    @pytest.mark.asyncio
    async def test_nearby_invalid_json(self):
        transport = make_transport(body="broken{json")
        provider = make_provider(transport)

        with pytest.raises(MappslsResponseError):
            await provider.nearby(12.97, 77.59, "test")

    @pytest.mark.asyncio
    async def test_nearby_skips_bad_entries(self):
        """Entries with invalid lat/lng are skipped."""
        body = {
            "suggestedLocations": [
                {
                    "placeName": "Good Place",
                    "distance": 100,
                    "latitude": 12.97,
                    "longitude": 77.59,
                },
                {
                    "placeName": "Bad Place",
                    "distance": "not a number",
                    "latitude": "bad",
                    "longitude": "bad",
                },
            ]
        }
        transport = make_transport(body=body)
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "test")
        assert len(result.places) == 1
        assert result.places[0].name == "Good Place"

    @pytest.mark.asyncio
    async def test_nearby_alt_field_names(self):
        """Handles 'name'/'address' instead of 'placeName'/'placeAddress'."""
        body = {
            "suggestedLocations": [
                {
                    "name": "Alt Place",
                    "address": "Some Road",
                    "distance": 200,
                    "latitude": 12.97,
                    "longitude": 77.59,
                }
            ]
        }
        transport = make_transport(body=body)
        provider = make_provider(transport)

        result = await provider.nearby(12.97, 77.59, "test")
        assert result.places[0].name == "Alt Place"
        assert result.places[0].address == "Some Road"

    @pytest.mark.asyncio
    async def test_nearby_caps_radius(self):
        """Radius is capped at 50km."""
        requests_seen = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(request)
            return httpx.Response(
                200,
                json={"suggestedLocations": []},
            )

        transport = httpx.MockTransport(handler)
        provider = make_provider(transport)

        await provider.nearby(12.97, 77.59, "test", radius=100000)
        assert "radius=50000" in str(requests_seen[0].url)


# ===================================================================
# Nearby Amenities (multi-category)
# ===================================================================


class TestNearbyAmenities:
    """Multi-category convenience method."""

    @pytest.mark.asyncio
    async def test_default_categories(self):
        transport = make_transport(body={"suggestedLocations": []})
        provider = make_provider(transport)

        results = await provider.nearby_amenities(12.97, 77.59)

        expected = {"hospital", "school", "metro", "park", "restaurant", "supermarket"}
        assert set(results.keys()) == expected

    @pytest.mark.asyncio
    async def test_custom_categories(self):
        transport = make_transport(body={"suggestedLocations": []})
        provider = make_provider(transport)

        results = await provider.nearby_amenities(
            12.97, 77.59, categories=["hospital", "atm"]
        )
        assert set(results.keys()) == {"hospital", "atm"}

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self):
        """If one category fails, others still return."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            # Fail every other request.
            if call_count % 2 == 0:
                return httpx.Response(500, content=b"error")
            return httpx.Response(
                200,
                json=SAMPLE_NEARBY_RESPONSE,
            )

        transport = httpx.MockTransport(handler)
        provider = make_provider(transport)

        results = await provider.nearby_amenities(
            12.97, 77.59, categories=["hospital", "school", "park"]
        )

        assert len(results) == 3
        # At least one should have results, others may be empty.
        has_results = sum(1 for r in results.values() if r.places)
        has_empty = sum(1 for r in results.values() if not r.places)
        assert has_results >= 1
        assert has_empty >= 1

    @pytest.mark.asyncio
    async def test_uses_category_code_mapping(self):
        """Category names are mapped to Mappls category codes."""
        requests_seen = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(str(request.url))
            return httpx.Response(
                200,
                json={"suggestedLocations": []},
            )

        transport = httpx.MockTransport(handler)
        provider = make_provider(transport)

        await provider.nearby_amenities(
            12.97, 77.59, categories=["hospital"]
        )

        assert len(requests_seen) == 1
        assert "HSPTL" in requests_seen[0]

    @pytest.mark.asyncio
    async def test_unknown_category_passes_through(self):
        """Unknown category names are used as-is."""
        requests_seen = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(str(request.url))
            return httpx.Response(
                200,
                json={"suggestedLocations": []},
            )

        transport = httpx.MockTransport(handler)
        provider = make_provider(transport)

        await provider.nearby_amenities(
            12.97, 77.59, categories=["coworking_space"]
        )

        assert "coworking_space" in requests_seen[0]


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    """Verify exported constants."""

    def test_amenity_categories_has_common_types(self):
        for key in ("hospital", "school", "metro", "park", "restaurant", "atm"):
            assert key in AMENITY_CATEGORIES

    def test_category_codes_are_strings(self):
        for code in AMENITY_CATEGORIES.values():
            assert isinstance(code, str)
            assert len(code) > 0

    def test_default_radius(self):
        assert DEFAULT_RADIUS == 3000

    def test_default_result_limit(self):
        assert DEFAULT_RESULT_LIMIT == 10

    def test_default_base_url(self):
        assert "atlas.mappls.com" in DEFAULT_BASE_URL
