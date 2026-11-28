# -*- coding: utf-8 -*-
"""Turn one API.SH search record into a :class:`NormalizedProperty`.

Only fields seen in real ``kind == "property"`` records are read, and only
when they have the expected type; anything else becomes ``None`` rather than
being guessed. ``raw`` (the source's own payload), images and the long
description are deliberately not copied.

Project-level records (``kind == "project"``) are not individual listings and
are rejected here; the provider counts them instead.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Optional, Tuple

from .models import (
    Inventory,
    NormalizedProperty,
    RecordKind,
    SourceRef,
)

APISH_SOURCE = "apish_99acres"
# Deliberately not "official": the data is relayed by a third-party service.
APISH_SOURCE_LABEL = "99acres listing (via the API.SH data service)"


class NormalizationError(ValueError):
    """A record cannot be normalized (missing id or wrong kind)."""


# ---------------------------------------------------------------------------
# Small typed getters: a wrong type is "unknown", never a guess.
# ---------------------------------------------------------------------------

def _text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _money(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float) and value == value and value not in (
        float("inf"),
        float("-inf"),
    ):
        return Decimal(str(value))
    return None


def _id_text(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return _text(value)


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp that states its own offset (``...Z``)."""
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None  # no stated offset: do not assume one
    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Possession
# ---------------------------------------------------------------------------

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_POSSESSION_TAG = re.compile(r"^POSSESSION IN ([A-Z]{3}) (\d{4})$", re.IGNORECASE)


def possession_from_tags(tags: Tuple[str, ...]) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Find possession wording in a listing's tags.

    Returns ``(verbatim tag, (year, month))``. The month is set only when the
    tag matches ``POSSESSION IN MON YYYY`` exactly. Other wording (for example
    ``READY TO MOVE``) is left in ``tags`` and is not interpreted.
    """
    for tag in tags:
        match = _POSSESSION_TAG.match(tag.strip())
        if match:
            month = _MONTHS.get(match.group(1).upper())
            if month:
                return tag.strip(), (int(match.group(2)), month)
    return None, None


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------

def normalize_property_record(
    record: Mapping[str, Any],
    *,
    source: str = APISH_SOURCE,
    source_label: str = APISH_SOURCE_LABEL,
    fetched_at: Optional[datetime] = None,
    cached: Optional[bool] = None,
) -> NormalizedProperty:
    """Normalize one ``kind == "property"`` search record."""
    if not isinstance(record, Mapping):
        raise NormalizationError("record is not an object")
    if record.get("kind") != RecordKind.PROPERTY.value:
        raise NormalizationError("not an individual property record")
    external_id = _id_text(record.get("id"))
    if not external_id:
        raise NormalizationError("record has no id")

    raw_tags = record.get("tags")
    tags = tuple(
        t.strip() for t in raw_tags if isinstance(t, str) and t.strip()
    ) if isinstance(raw_tags, list) else ()
    possession_text, possession_month = possession_from_tags(tags)

    advertiser = record.get("advertiser")
    advertiser = advertiser if isinstance(advertiser, Mapping) else {}

    # ``price_text`` is a display string for properties. If a source ever sends
    # an object here it is a different shape: ignore it, do not parse it.
    price_text = _text(record.get("price_text"))

    return NormalizedProperty(
        ref=SourceRef(source, RecordKind.PROPERTY, external_id),
        inventory=Inventory.LIVE,
        source_label=source_label,
        title=_text(record.get("title")),
        url=_text(record.get("url")),
        purpose=_text(record.get("purpose")),
        property_type_text=_text(record.get("type")),
        price=_money(record.get("price")),
        price_max=_money(record.get("price_max")),
        currency=_text(record.get("currency")),
        price_text=price_text,
        # A number on properties; a formatted string on projects. Strings are
        # not parsed.
        price_per_sqft=_int(record.get("price_per_sqft")),
        bedrooms=_int(record.get("bedrooms")),
        bathrooms=_int(record.get("bathrooms")),
        balconies=_int(record.get("balconies")),
        area_text=_text(record.get("area_text")),
        carpet_sqft=_number(record.get("carpet_sqft")),
        super_builtup_sqft=_number(record.get("super_builtup_sqft")),
        builtup_sqft=_number(record.get("builtup_sqft")),
        floor=_id_text(record.get("floor")),
        total_floors=_int(record.get("total_floors")),
        furnishing=_text(record.get("furnishing")),
        transaction=_text(record.get("transaction")),
        ownership=_text(record.get("ownership")),
        tags=tags,
        society=_text(record.get("society")),
        project_id=_id_text(record.get("project_id")),
        building_id=_id_text(record.get("building_id")),
        prop_id=_id_text(record.get("prop_id")),
        # For a property, ``locality`` is a string such as "Whitefield,
        # Bangalore". (The response's top-level ``location`` is an object and
        # is handled by the provider.)
        locality=_text(record.get("locality")),
        city=_text(record.get("city")),
        city_id=_id_text(record.get("city_id")),
        latitude=_number(record.get("latitude")),
        longitude=_number(record.get("longitude")),
        verified=record.get("verified") if isinstance(record.get("verified"), bool) else None,
        advertiser_type=_text(advertiser.get("type")),
        advertiser_name=_text(advertiser.get("name")),
        posted=parse_iso_datetime(record.get("posted")),
        updated=parse_iso_datetime(record.get("updated")),
        expires_text=_text(record.get("expires")),
        possession_text=possession_text,
        possession_month=possession_month,
        fetched_at=fetched_at,
        cached=cached,
    )


# ---------------------------------------------------------------------------
# Response-level helpers
# ---------------------------------------------------------------------------

def normalize_applied_filters(filters: Any) -> dict:
    """Type the ``applied_filters`` echo.

    Real responses give some values as strings (``"budget_max": "15000000"``)
    and some as lists (``"bedrooms": [2]``). Numbers become ints; anything that
    does not parse stays ``None``.
    """
    if not isinstance(filters, Mapping):
        return {}
    out: dict = {}
    for key in ("budget_min", "budget_max", "area_min", "area_max"):
        value = filters.get(key)
        if isinstance(value, bool):
            out[key] = None
        elif isinstance(value, int):
            out[key] = value
        elif isinstance(value, str) and value.strip().isdigit():
            out[key] = int(value.strip())
        else:
            out[key] = None
    bedrooms = filters.get("bedrooms")
    if isinstance(bedrooms, list):
        out["bedrooms"] = [b for b in (_int(x) for x in bedrooms) if b is not None]
    for key in ("type", "furnishing", "status", "posted_by"):
        value = filters.get(key)
        out[key] = (
            [v for v in value if isinstance(v, str)] if isinstance(value, list) else None
        )
    out["verified"] = filters.get("verified") if isinstance(filters.get("verified"), bool) else None
    return out


def matched_locality_name(location: Any) -> Optional[str]:
    """The locality API.SH says it matched.

    The top-level ``location`` is an object ``{name, locality_id, city_id}`` in
    real responses; inside project records it is a plain string. Both are
    accepted here; anything else is unknown.
    """
    if isinstance(location, Mapping):
        return _text(location.get("name"))
    return _text(location)
