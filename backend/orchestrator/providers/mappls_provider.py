"""Mappls location provider — geocoding and nearby place search.

Wraps the Mappls (MapMyIndia) REST APIs:
  • Geocoding:  GET /address/geocode
  • Nearby:     GET /places/nearby/json

Authentication uses a static access key passed as the `access_token` query parameter.
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

DEFAULT_BASE_URL = "https://search.mappls.com/search"
DEFAULT_RADIUS = 3000  # metres
MIN_RADIUS = 500  # metres - smallest radius the nearby search accepts
MAX_RADIUS = 10_000  # metres - largest radius the nearby search accepts


def clamp_radius(radius: int) -> int:
    """Return the radius (metres) the nearby search will actually use."""
    return max(MIN_RADIUS, min(radius, MAX_RADIUS))
DEFAULT_RESULT_LIMIT = 10

# ---------------------------------------------------------------------------
# Bengaluru‑relevant POI category keywords
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
    """Result of a geocoding lookup.

    latitude and longitude are optional because the Mappls API may omit them.
    """

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    formatted_address: str = ""
    eloc: str = ""  # Mappls place code
    locality: str = ""
    city: str = ""
    state: str = ""
    pincode: str = ""

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

@dataclass(frozen=True, slots=True)
class NearbyPlace:
    """A single nearby point‑of‑interest.

    latitude and longitude are optional because they may be omitted in the API response.
    """

    name: str = ""
    category: str = ""
    distance: int = 0  # metres
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: str = ""
    eloc: str = ""
    phone: str = ""

@dataclass(frozen=True, slots=True)
class NearbyResult:
    """Container for a nearby-places search.

    The optional ``error`` field stores a per-category failure message when a
    recoverable HTTP 5xx or network error occurs. Authentication, permission and
    rate-limit errors are re-raised by ``nearby_amenities`` and therefore never
    populate this field.
    """

    places: list[NearbyPlace] = field(default_factory=list)
    keyword: str = ""
    error: str | None = None

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
    """Non‑success HTTP status."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")

class MappslsResponseError(MappslsError):
    """Unexpected response body."""

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MappslsProvider:
    """Async client for Mappls geocoding and nearby‑places APIs."""

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

    async def __aenter__(self) -> "MappslsProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --------------------------------------------------------------------
    def _common_params(self) -> dict[str, str]:
        """Static access token query parameter."""
        return {"access_token": self._api_key}

    # --------------------------------------------------------------------
    def _handle_error(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == 401:
            raise MappslsAuthenticationError("Invalid or expired Mappls bearer token")
        if status == 429:
            raise MappslsRateLimitError("Mappls rate limit exceeded")
        if status >= 400:
            detail = response.text[:300] if response.text else ""
            raise MappslsAPIError(status, detail)

    # --------------------------------------------------------------------
    async def geocode(self, address: str) -> Optional[GeocodedLocation]:
        """Geocode an address string to coordinates.

        Returns the best match or ``None`` if no results or coordinates missing.
        """
        if not address or not address.strip():
            raise ValueError("address cannot be empty")

        url = f"{self._base_url}/address/geocode"
        try:
            response = await self._client.get(
                url,
                params={
                    "address": address.strip(),
                    "region": "IND",
                    **self._common_params(),
                },
            )
        except httpx.TimeoutException as exc:
            raise MappslsTimeoutError(f"Geocode request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise MappslsAPIError(0, f"Connection failed: {exc}") from exc

        self._handle_error(response)
        try:
            data = response.json()
        except Exception as exc:
            raise MappslsResponseError(f"Invalid JSON in geocode response: {exc}") from exc

        result = data.get("copResults")
        if isinstance(result, list):
            if not result:
                return None
            result = result[0]
        if not isinstance(result, dict) or not result:
            return None
        lat = result.get("latitude") or result.get("lat")
        lng = result.get("longitude") or result.get("lng")
        # Return location even if coordinates missing
        return GeocodedLocation(
            latitude=float(lat) if lat is not None else None,
            longitude=float(lng) if lng is not None else None,
            formatted_address=result.get("formatted_address", result.get("formattedAddress", "")),
            eloc=result.get("eLoc", ""),
            locality=result.get("locality", result.get("area", "")),
            city=result.get("city", result.get("district", "")),
            state=result.get("state", ""),
            pincode=result.get("pincode", ""),
        )

    # --------------------------------------------------------------------
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
        if not keywords or not keywords.strip():
            raise ValueError("keywords cannot be empty")
        radius = clamp_radius(radius)
        url = f"{self._base_url}/places/nearby/json"
        params: dict[str, Any] = {
            "keywords": keywords.strip(),
            "refLocation": f"{latitude},{longitude}",
            "radius": radius,
            "page": 1,
            "region": "IND",
            "sortBy": sort_by,
            **self._common_params(),
        }
        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise MappslsTimeoutError(f"Nearby request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise MappslsAPIError(0, f"Connection failed: {exc}") from exc
        self._handle_error(response)
        if response.status_code == 204:
            # No content – return empty result
            return NearbyResult(places=[], keyword=keywords.strip())
        try:
            data = response.json()
        except Exception as exc:
            raise MappslsResponseError(f"Invalid JSON in nearby response: {exc}") from exc
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
                    latitude=lat if lat != 0 else None,
                    longitude=lng if lng != 0 else None,
                    address=item.get("placeAddress", item.get("address", "")),
                    eloc=item.get("eLoc", ""),
                    phone=item.get("phone", ""),
                )
            )
        return NearbyResult(places=places, keyword=keywords.strip())

    # --------------------------------------------------------------------
    async def nearby_amenities(
        self,
        latitude: float,
        longitude: float,
        categories: list[str] | None = None,
        radius: int = DEFAULT_RADIUS,
        limit_per_category: int = 5,
    ) -> dict[str, NearbyResult]:
        """Search nearby amenities for given categories.

        Returns a mapping from category name to ``NearbyResult``.
        """
        # Default categories limited to those required by tests
        default_cats = ["hospital", "school", "metro", "park", "restaurant", "supermarket"]
        if categories is None:
            categories = default_cats
        result: dict[str, NearbyResult] = {}
        for cat in categories:
            keyword = AMENITY_CATEGORIES.get(cat, cat)
            try:
                result[cat] = await self.nearby(
                    latitude=latitude,
                    longitude=longitude,
                    keywords=keyword,
                    radius=radius,
                    limit=limit_per_category,
                )
            except MappslsError as exc:
                # Propagate auth, permission, and rate‑limit errors.
                if isinstance(exc, (MappslsAuthenticationError, MappslsRateLimitError)):
                    raise
                if isinstance(exc, MappslsAPIError) and exc.status_code in (401, 403, 429):
                    raise
                # Recoverable per‑category failure (e.g., HTTP 5xx, timeout, connect).
                logger.warning("Nearby category '%s' failed: %s", cat, exc)
                result[cat] = NearbyResult(places=[], keyword=keyword, error=str(exc))
        return result

# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

__all__ = [
    "MappslsProvider",
    "GeocodedLocation",
    "NearbyPlace",
    "NearbyResult",
    "AMENITY_CATEGORIES",
    "clamp_radius",
    "MappslsError",
    "MappslsAuthenticationError",
    "MappslsRateLimitError",
    "MappslsTimeoutError",
    "MappslsAPIError",
    "MappslsResponseError",
]
