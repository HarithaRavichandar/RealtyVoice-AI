# -*- coding: utf-8 -*-
"""Source-aware property types shared by every inventory provider.

Two kinds of inventory exist and are never mixed silently:

* ``demo``  - the local SQLite dataset (synthetic and historical Kaggle rows),
  kept for development and testing.
* ``live``  - listings fetched from an external service (API.SH, which
  relays 99acres data). Live results are transient: nothing here writes them
  to the ``properties`` table.

Every listing carries a :class:`SourceRef` (source + record kind + external id)
so a project id can never be mistaken for a property id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class Inventory(str, Enum):
    """Which pool of records a result came from."""

    LIVE = "live"
    DEMO = "demo"


class RecordKind(str, Enum):
    """What a source record represents.

    API.SH returns both individual listings (``property``) and project-level
    records (``project``). Only ``property`` records are individual listings.
    """

    PROPERTY = "property"
    PROJECT = "project"


@dataclass(frozen=True)
class SourceRef:
    """Exact identity of a record: source + record kind + external id."""

    source: str
    kind: RecordKind
    external_id: str

    @property
    def key(self) -> str:
        return f"{self.source}:{self.kind.value}:{self.external_id}"


@dataclass(frozen=True)
class SearchCriteria:
    """What the customer asked for (already merged by the caller)."""

    location: Optional[str] = None
    property_type: Optional[str] = None
    configuration: Optional[str] = None
    min_budget: Optional[Decimal] = None
    max_budget: Optional[Decimal] = None
    possession_before: Optional[date] = None


@dataclass(frozen=True)
class NormalizedProperty:
    """One individual property, in a provider-independent shape.

    Only fields seen in real records are modelled. ``None`` means "the source
    did not say", never "zero" or "no".
    """

    ref: SourceRef
    inventory: Inventory
    source_label: str

    title: Optional[str] = None
    url: Optional[str] = None
    purpose: Optional[str] = None
    property_type_text: Optional[str] = None

    price: Optional[Decimal] = None
    price_max: Optional[Decimal] = None
    currency: Optional[str] = None
    price_text: Optional[str] = None
    price_per_sqft: Optional[int] = None

    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    balconies: Optional[int] = None
    area_text: Optional[str] = None
    carpet_sqft: Optional[float] = None
    super_builtup_sqft: Optional[float] = None
    builtup_sqft: Optional[float] = None
    floor: Optional[str] = None
    total_floors: Optional[int] = None
    furnishing: Optional[str] = None
    transaction: Optional[str] = None
    ownership: Optional[str] = None
    tags: Tuple[str, ...] = ()
    society: Optional[str] = None

    # Link to a project record. Kept as given; it is a link, not an identity.
    project_id: Optional[str] = None
    building_id: Optional[str] = None
    prop_id: Optional[str] = None

    locality: Optional[str] = None
    city: Optional[str] = None
    city_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    verified: Optional[bool] = None
    advertiser_type: Optional[str] = None
    advertiser_name: Optional[str] = None

    posted: Optional[datetime] = None
    updated: Optional[datetime] = None
    # Kept verbatim: the source gives no time zone for it.
    expires_text: Optional[str] = None

    # Possession, never invented. ``possession_text`` is the verbatim source
    # wording; ``possession_month`` is (year, month) only when the wording
    # follows an exact known pattern; ``possession_date`` is an exact date
    # (demo records only).
    possession_text: Optional[str] = None
    possession_month: Optional[Tuple[int, int]] = None
    possession_date: Optional[date] = None
    availability_status: Optional[str] = None

    is_synthetic: bool = False
    price_note: Optional[str] = None

    # When the source says it fetched the data (live only), and whether the
    # relay served it from its own cache.
    fetched_at: Optional[datetime] = None
    cached: Optional[bool] = None

    # Filled by the search service: criterion -> "match" | "unknown" |
    # "not_requested". Mismatches never reach this point.
    checks: Mapping[str, str] = field(default_factory=dict)


# Keys of ``ProviderResult.excluded``.
EXCLUDED_PROJECT_RECORDS = "project_records"
EXCLUDED_UNKNOWN_KIND = "unknown_record_kind"
EXCLUDED_UNPARSEABLE = "unparseable_records"
EXCLUDED_DUPLICATES = "duplicates"
EXCLUDED_BUDGET_MISMATCH = "budget_mismatch"
EXCLUDED_BUDGET_UNKNOWN = "budget_unknown"
EXCLUDED_BEDROOMS_MISMATCH = "bedrooms_mismatch"
EXCLUDED_POSSESSION_MISMATCH = "possession_mismatch"


@dataclass(frozen=True)
class ProviderResult:
    """What one provider call returned, labelled with where it came from."""

    source: str
    inventory: Inventory
    listings: Tuple[NormalizedProperty, ...] = ()
    # Counts of records that were dropped, by reason.
    excluded: Mapping[str, int] = field(default_factory=dict)
    upstream_total: Optional[int] = None
    upstream_page: Optional[int] = None
    upstream_next_page: Optional[int] = None
    fetched_at: Optional[datetime] = None
    cached: Optional[bool] = None
    # Credits the source reported for this call; None when it said nothing.
    credits_used: Optional[int] = None
    # The locality the source says it matched (may differ from the request).
    matched_locality: Optional[str] = None
    # Parameters actually sent (never includes credentials).
    sent_params: Mapping[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()
