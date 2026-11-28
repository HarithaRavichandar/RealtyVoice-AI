# -*- coding: utf-8 -*-
"""API.SH provider: individual property listings relayed from 99acres.

Contract used (from the API.SH documentation and a real response):

* ``GET {base}/search`` with the key in the ``x-api-key`` header.
* Parameters sent here: ``purpose``, ``city``, ``location``, ``type``,
  ``bedrooms``, ``budget_min``, ``budget_max``. Nothing else.
* Never sent: ``page``, ``fresh``, ``raw``, ``fields``, ``url`` and the other
  filters. Their cost and behaviour are not documented to us, so they are not
  used. One search is one call, with no retries.
* Errors return ``{"success": false, "error": "..."}`` with an HTTP status; a
  lookup that found nothing is HTTP 404 and is documented as costing a credit.

NOT verified, so marked as convention only: how API.SH uses 401/402/403/429
and its 5xx codes. The status mapping below follows normal HTTP meaning and
the shared quota rule; it is not confirmed by API.SH documentation.

Only ``kind == "property"`` records are individual listings. Project records
are counted and dropped; their price ranges are never read as available units.

The key is sent only in the request header. It is never logged, never put in
an error message, and not shown by ``repr``.
"""

from __future__ import annotations

import logging
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Dict, Optional, Tuple

import httpx

from backend.orchestrator.providers.failures import is_quota_response
from backend.services.location_normalizer import location_variants, squash

from .base import PropertyProvider
from .errors import (
    InventoryAPIError,
    InventoryAuthError,
    InventoryConfigError,
    InventoryNetworkError,
    InventoryQuotaError,
    InventoryRateLimitError,
    InventoryRequestError,
    InventoryResponseError,
    InventoryTimeoutError,
)
from .models import (
    EXCLUDED_PROJECT_RECORDS,
    EXCLUDED_UNKNOWN_KIND,
    EXCLUDED_UNPARSEABLE,
    Inventory,
    ProviderResult,
    RecordKind,
    SearchCriteria,
)
from .normalize import (
    APISH_SOURCE,
    APISH_SOURCE_LABEL,
    NormalizationError,
    matched_locality_name,
    normalize_applied_filters,
    normalize_property_record,
    parse_iso_datetime,
)
from .validation import bedrooms_from_configuration, documented_type

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.api.sh/99acres/v1"
DEFAULT_TIMEOUT_SECONDS = 15.0

# Cities named in the API.SH documentation for the ``city`` parameter. The
# list there is introduced with "e.g.", so other cities may work but are not
# verified; an unlisted name is sent as a locality instead of guessing a city.
DOCUMENTED_CITIES = (
    "mumbai", "pune", "bangalore", "hyderabad", "chennai",
    "kolkata", "gurgaon", "noida", "thane",
)
_DOCUMENTED_CITY_KEYS = {squash(c): c for c in DOCUMENTED_CITIES}


def _city_of(text: str) -> Optional[str]:
    """The documented city token for ``text`` (aliases included), or ``None``."""
    for variant in location_variants(text):
        if variant in _DOCUMENTED_CITY_KEYS:
            return _DOCUMENTED_CITY_KEYS[variant]
    return None


def split_location(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a customer's place into ``(city, locality)`` for the two parameters.

    ``"Whitefield, Bangalore"`` -> ``("bangalore", "Whitefield")``;
    ``"Pune"`` -> ``("pune", None)``; ``"Whitefield"`` -> ``(None, "Whitefield")``
    (the documentation says a locality works best with a city, so a place with
    no recognisable city is sent as given and the matched locality is reported
    back for the customer to confirm).
    """
    if not text or not text.strip():
        return None, None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        for index, part in enumerate(parts):
            city = _city_of(part)
            if city:
                rest = " ".join(p for i, p in enumerate(parts) if i != index)
                return city, rest or None
        return None, " ".join(parts)
    only = parts[0]
    city = _city_of(only)
    if city:
        return city, None
    tokens = only.split()
    if len(tokens) >= 2:
        for index in (len(tokens) - 1, 0):  # trailing city, then leading city
            city = _city_of(tokens[index])
            if city:
                rest = " ".join(t for i, t in enumerate(tokens) if i != index)
                return city, rest or None
    return None, only


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "[redacted]") if secret else text


class ApishPropertyProvider(PropertyProvider):
    """Live individual listings from API.SH (99acres)."""

    source_name = APISH_SOURCE
    inventory = Inventory.LIVE

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=timeout)
            self._owns_client = True

    def __repr__(self) -> str:  # never show the key
        return f"ApishPropertyProvider(base_url={self._base_url!r}, configured={bool(self._api_key)})"

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "ApishPropertyProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    def build_params(self, criteria: SearchCriteria) -> Dict[str, str]:
        """The query parameters for ``criteria`` (documented ones only)."""
        city, locality = split_location(criteria.location)
        if not city and not locality:
            raise InventoryRequestError("a location is required for a live search")
        params: Dict[str, str] = {"purpose": "sale"}
        if city:
            params["city"] = city
        if locality:
            params["location"] = locality
        type_token = documented_type(criteria.property_type)
        if type_token:
            params["type"] = type_token
        bedrooms = bedrooms_from_configuration(criteria.configuration)
        if bedrooms is not None:
            params["bedrooms"] = str(bedrooms)
        # Widen the upstream window to whole rupees; the exact rule is applied
        # locally afterwards.
        if criteria.min_budget is not None:
            params["budget_min"] = str(int(Decimal(criteria.min_budget).to_integral_value(ROUND_FLOOR)))
        if criteria.max_budget is not None:
            params["budget_max"] = str(int(Decimal(criteria.max_budget).to_integral_value(ROUND_CEILING)))
        return params

    # ------------------------------------------------------------------
    def _raise_for_status(self, response: httpx.Response) -> None:
        """Status mapping. Conventional HTTP meaning; not confirmed by API.SH."""
        status = response.status_code
        if status in (401, 403):
            raise InventoryAuthError("the live listings service rejected the credentials")
        if is_quota_response(status, response.text):
            raise InventoryQuotaError("the live listings service usage limit was reached")
        if status == 429:
            raise InventoryRateLimitError("the live listings service rate limit was hit")
        if status >= 400:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("error"), str):
                    detail = _redact(body["error"][:200], self._api_key)
            except ValueError:
                pass
            raise InventoryAPIError(status, detail)

    # ------------------------------------------------------------------
    async def search(self, criteria: SearchCriteria) -> ProviderResult:
        params = self.build_params(criteria)  # may raise InventoryRequestError
        if not self._api_key:
            raise InventoryConfigError("APISH_API_KEY is not configured")

        started = time.monotonic()
        try:
            response = await self._client.get(
                f"{self._base_url}/search",
                params=params,
                headers={"x-api-key": self._api_key},
                timeout=self._timeout,
            )
        except httpx.TimeoutException:
            raise InventoryTimeoutError("the live listings service timed out") from None
        except httpx.HTTPError as exc:
            # The exception text can include request details; keep only its type.
            raise InventoryNetworkError(
                f"could not reach the live listings service ({type(exc).__name__})"
            ) from None

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "apish search finished",
            extra={"status_code": response.status_code, "elapsed_ms": elapsed_ms},
        )

        if response.status_code == 404:
            # Documented: a lookup that ran but found nothing. It still costs a
            # credit, so ``credits_used`` is left unset for the caller to count.
            return ProviderResult(
                source=self.source_name,
                inventory=self.inventory,
                sent_params=dict(params),
                notes=("the service reported no match (HTTP 404)",),
            )
        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError:
            raise InventoryResponseError("the live listings response was not valid JSON") from None
        if not isinstance(body, dict):
            raise InventoryResponseError("the live listings response was not an object")
        if body.get("success") is False:
            raise InventoryAPIError(response.status_code, "service reported failure")
        results = body.get("results")
        if not isinstance(results, list):
            raise InventoryResponseError("the live listings response had no results list")

        fetched_at = parse_iso_datetime(body.get("fetched_at"))
        cached = body.get("cached") if isinstance(body.get("cached"), bool) else None
        listings = []
        excluded: Dict[str, int] = {}

        def bump(reason: str) -> None:
            excluded[reason] = excluded.get(reason, 0) + 1

        for record in results:
            kind = record.get("kind") if isinstance(record, dict) else None
            if kind == RecordKind.PROJECT.value:
                bump(EXCLUDED_PROJECT_RECORDS)
            elif kind == RecordKind.PROPERTY.value:
                try:
                    listings.append(
                        normalize_property_record(
                            record,
                            source=self.source_name,
                            source_label=APISH_SOURCE_LABEL,
                            fetched_at=fetched_at,
                            cached=cached,
                        )
                    )
                except NormalizationError:
                    bump(EXCLUDED_UNPARSEABLE)
            elif isinstance(record, dict):
                bump(EXCLUDED_UNKNOWN_KIND)  # not a kind we understand: not guessed
            else:
                bump(EXCLUDED_UNPARSEABLE)

        notes = []
        applied = normalize_applied_filters(body.get("applied_filters"))
        for sent_key in ("budget_min", "budget_max"):
            if sent_key in params and applied.get(sent_key) != int(params[sent_key]):
                notes.append(f"the service did not report applying {sent_key} as sent")

        credits_used = body.get("credits_used")
        if isinstance(credits_used, bool) or not isinstance(credits_used, int) or credits_used < 0:
            credits_used = None

        def _page(name: str) -> Optional[int]:
            value = body.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        return ProviderResult(
            source=self.source_name,
            inventory=self.inventory,
            listings=tuple(listings),
            excluded=excluded,
            upstream_total=_page("total"),
            upstream_page=_page("page"),
            upstream_next_page=_page("next_page"),
            fetched_at=fetched_at,
            cached=cached,
            credits_used=credits_used,
            matched_locality=matched_locality_name(body.get("location")),
            sent_params=dict(params),
            notes=tuple(notes),
        )
