"""Mappls location provider — geocoding and nearby place search.

Wraps the Mappls (MapMyIndia) REST APIs:
  • Geocoding:  GET /places/geocode?address=...
  • Nearby:     GET /places/nearby/json?keywords=...&refLocation=lat,lng

Authentication uses an OAuth2 bearer token passed via the
``Authorization: bearer <token>`` header.

Bengaluru-focused category codes are provided as constants for
common amenity types relevant to real-estate searches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://atlas.mappls.com/api"
DEFAULT_RADIUS = 3000  # metres
DEFAULT_RESULT_LIMIT = 10

# ---------------------------------------------------------------------------
# Bengaluru-relevant POI category keywords
# ---------------------------------------------------------------------------

AMENITY_CATEGORIES: dict[str, str] = {
    "hospital": "HSPTL",
    "school": "SCHCOL",
    "metro": "TRNMET",
    "park": "PRKNFR",
    "restaurant": "FODRES",
    "atm": "FINATM",
    "pharmacy": "HLTHPH",
    "supermarket": "SHPGSM",
    "gym": "HLTHGM",
    "mall": "SHPGML",
    "bus_stop": "TRNBUS",
    "bank": "FINBNK",
    "temple": "RLGTEM",
    "mosque": "RLGMOS",
    "church": "RLGCHR",
    "police": "GOVPOL",
    "petrol": "TRNPET",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeocodedLocation:
    """Result of a geocoding lookup."""

    latitude: float
    longitude: float
    formatted_address: str
    eloc: str = ""  # Mappls place code
    locality: str = ""
    city: str = ""
    state: str = ""
    pincode: str = ""


@dataclass(frozen=True, slots=True)
class NearbyPlace:
    """A single nearby point-of-interest."""

    name: str
    category: str
    distance: int  # metres
    latitude: float
    longitude: float
    address: str = ""
    eloc: str = ""
    phone: str = ""


@dataclass(frozen=True, slots=True)
class NearbyResult:
    """Container for a nearby-places search."""

    places: list[NearbyPlace] = field(default_factory=list)
    keyword: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MappslsError(Exception):
    """Base error for Mappls API calls."""


class MappslsAuthenticationError(MappslsError):
    """Invalid or expired bearer token."""


class MappslsRateLimitError(MappslsError):
    """Rate limit exceeded."""


class MappslsTimeoutError(MappslsError):
    """Request timed out."""


class MappslsAPIError(MappslsError):
    """Non-success HTTP status."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


class MappslsResponseError(MappslsError):
    """Unexpected response body."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MappslsProvider:
    """Async client for Mappls geocoding and nearby-places APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required")

        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> "MappslsProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    # -- auth header --------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self._api_key}"}

    # -- error handling -----------------------------------------------------

    def _handle_error(self, response: httpx.Response) -> None:
        """Raise a typed exception for non-2xx responses."""
        status = response.status_code

        if status == 401:
            raise MappslsAuthenticationError(
                "Invalid or expired Mappls bearer token"
            )
        if status == 429:
            raise MappslsRateLimitError("Mappls rate limit exceeded")
        if status >= 400:
            detail = response.text[:300] if response.text else ""
            raise MappslsAPIError(status, detail)

    # -- geocode ------------------------------------------------------------

    async def geocode(self, address: str) -> Optional[GeocodedLocation]:
        """Geocode an address string to coordinates.

        Returns the best match or ``None`` if no results.
        """
        if not address or not address.strip():
            raise ValueError("address cannot be empty")

        url = f"{self._base_url}/places/geocode"

        try:
            response = await self._client.get(
                url,
                params={"address": address.strip(), "region": "IND"},
                headers=self._auth_headers(),
            )
        except httpx.TimeoutException as exc:
            raise MappslsTimeoutError(
                f"Geocode request timed out: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise MappslsAPIError(0, f"Connection failed: {exc}") from exc

        self._handle_error(response)

        try:
            data = response.json()
        except Exception as exc:
            raise MappslsResponseError(
                f"Invalid JSON in geocode response: {exc}"
            ) from exc

        # Mappls returns {"copResults": {...}} for geocode
        result = data.get("copResults") or data

        # Could be a dict (single result) or list
        if isinstance(result, list):
            if not result:
                return None
            result = result[0]

        if not isinstance(result, dict):
            return None

        lat = result.get("latitude") or result.get("lat")
        lng = result.get("longitude") or result.get("lng")

        if lat is None or lng is None:
            return None

        return GeocodedLocation(
            latitude=float(lat),
            longitude=float(lng),
            formatted_address=result.get("formatted_address", ""),
            eloc=result.get("eLoc", ""),
            locality=result.get("locality", result.get("area", "")),
            city=result.get("city", result.get("district", "")),
            state=result.get("state", ""),
            pincode=result.get("pincode", ""),
        )

    # -- nearby -------------------------------------------------------------

    async def nearby(
        self,
        latitude: float,
        longitude: float,
        keywords: str,
        *,
        radius: int = DEFAULT_RADIUS,
        limit: int = DEFAULT_RESULT_LIMIT,
        sort_by: str = "dist:asc",
    ) -> NearbyResult:
        """Search for nearby places around a coordinate.

        ``keywords`` can be a category code (e.g. ``"HSPTL"``) or a
        free-text keyword (e.g. ``"hospital"``).
        """
        if not keywords or not keywords.strip():
            raise ValueError("keywords cannot be empty")

        url = f"{self._base_url}/places/nearby/json"

        params: dict[str, Any] = {
            "keywords": keywords.strip(),
            "refLocation": f"{latitude},{longitude}",
            "radius": min(radius, 50000),
            "page": 1,
            "region": "IND",
            "sortBy": sort_by,
        }

        try:
            response = await self._client.get(
                url,
                params=params,
                headers=self._auth_headers(),
            )
        except httpx.TimeoutException as exc:
            raise MappslsTimeoutError(
                f"Nearby request timed out: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise MappslsAPIError(0, f"Connection failed: {exc}") from exc

        self._handle_error(response)

        try:
            data = response.json()
        except Exception as exc:
            raise MappslsResponseError(
                f"Invalid JSON in nearby response: {exc}"
            ) from exc

        raw_places = data.get("suggestedLocations") or []

        places: list[NearbyPlace] = []

        for item in raw_places[:limit]:
            try:
                lat = float(item.get("latitude", 0))
                lng = float(item.get("longitude", 0))
                dist = int(item.get("distance", 0))
            except (TypeError, ValueError):
                continue

            places.append(
                NearbyPlace(
                    name=item.get("placeName", item.get("name", "")),
                    category=item.get("categoryCode", ""),
                    distance=dist,
                    latitude=lat,
                    longitude=lng,
                    address=item.get("placeAddress", item.get("address", "")),
                    eloc=item.get("eLoc", ""),
                    phone=item.get("phone", ""),
                )
            )

        return NearbyResult(places=places, keyword=keywords.strip())

    # -- convenience: multi-category nearby ---------------------------------

    async def nearby_amenities(
        self,
        latitude: float,
        longitude: float,
        categories: Optional[list[str]] = None,
        *,
        radius: int = DEFAULT_RADIUS,
        limit_per_category: int = 5,
    ) -> dict[str, NearbyResult]:
        """Search for multiple amenity categories around a point.

        ``categories`` defaults to a real-estate-relevant set:
        hospital, school, metro, park, restaurant, supermarket.

        Returns a dict mapping each category name to its NearbyResult.
        """
        if categories is None:
            categories = [
                "hospital",
                "school",
                "metro",
                "park",
                "restaurant",
                "supermarket",
            ]

        results: dict[str, NearbyResult] = {}

        for cat in categories:
            keyword = AMENITY_CATEGORIES.get(cat.lower(), cat)

            try:
                result = await self.nearby(
                    latitude=latitude,
                    longitude=longitude,
                    keywords=keyword,
                    radius=radius,
                    limit=limit_per_category,
                )
                results[cat] = result

            except MappslsError as exc:
                logger.warning(
                    "Nearby search for '%s' failed: %s", cat, exc
                )
                results[cat] = NearbyResult(places=[], keyword=keyword)

        return results


__all__ = [
    "MappslsProvider",
    "GeocodedLocation",
    "NearbyPlace",
    "NearbyResult",
    "AMENITY_CATEGORIES",
    "MappslsError",
    "MappslsAuthenticationError",
    "MappslsRateLimitError",
    "MappslsTimeoutError",
    "MappslsAPIError",
    "MappslsResponseError",
    "DEFAULT_BASE_URL",
    "DEFAULT_RADIUS",
    "DEFAULT_RESULT_LIMIT",
]
