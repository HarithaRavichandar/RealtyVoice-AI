# -*- coding: utf-8 -*-
"""Demo inventory: the existing SQLite ``properties`` table.

A thin adapter over :class:`PropertySearchService`; the search logic is not
duplicated or changed. Results are labelled as demo inventory with the source
each row already records (synthetic or historical Kaggle data).
"""

from __future__ import annotations

from typing import Optional

from backend.models.property import Property
from backend.services.property_search import PropertySearchService

from .base import PropertyProvider
from .models import (
    Inventory,
    NormalizedProperty,
    ProviderResult,
    RecordKind,
    SearchCriteria,
    SourceRef,
)
from .validation import bedrooms_from_configuration

DEMO_SOURCE = "sqlite"


def _label(source_name: Optional[str]) -> str:
    if source_name == "synthetic":
        return "Demo data (synthetic)"
    if source_name == "kaggle_house_price_india":
        return "Historical dataset (Kaggle)"
    return f"Stored record ({source_name or 'unknown source'})"


def to_normalized(row: Property) -> NormalizedProperty:
    """Map one ``Property`` row, without inventing anything."""
    source = row.source_name or DEMO_SOURCE
    external_id = (
        str(row.source_record_id)
        if row.source_record_id not in (None, "")
        else f"row-{row.property_id}"
    )
    low = row.price_min if row.price_min is not None else (
        row.listing_price if row.listing_price is not None else row.price_max
    )
    high = row.price_max if row.price_max is not None else (
        row.listing_price if row.listing_price is not None else row.price_min
    )
    synthetic = row.source_name == "synthetic"
    return NormalizedProperty(
        ref=SourceRef(source, RecordKind.PROPERTY, external_id),
        inventory=Inventory.DEMO,
        source_label=_label(row.source_name),
        title=row.project_name,
        url=row.source_url,
        property_type_text=row.property_type,
        price=low,
        price_max=high,
        currency="INR" if low is not None or high is not None else None,
        bedrooms=bedrooms_from_configuration(row.configuration),
        locality=row.location,
        possession_date=row.possession_date.date() if row.possession_date else None,
        availability_status=row.availability_status,
        is_synthetic=synthetic,
        price_note="demo data" if synthetic else "historical",
    )


class SQLiteProvider(PropertyProvider):
    """Demo records from the local database."""

    source_name = DEMO_SOURCE
    inventory = Inventory.DEMO

    def __init__(self, service: PropertySearchService, *, limit: int = 20) -> None:
        self._service = service
        self._limit = limit

    async def search(self, criteria: SearchCriteria) -> ProviderResult:
        # The database call is synchronous, as everywhere else in the app.
        total, rows = self._service.search(
            location=criteria.location,
            property_type=criteria.property_type,
            configuration=criteria.configuration,
            min_price=criteria.min_budget,
            max_price=criteria.max_budget,
            possession_before=criteria.possession_before,
            limit=self._limit,
        )
        return ProviderResult(
            source=self.source_name,
            inventory=self.inventory,
            listings=tuple(to_normalized(row) for row in rows),
            upstream_total=total,
        )
