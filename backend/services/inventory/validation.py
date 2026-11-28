# -*- coding: utf-8 -*-
"""Local checks of live listings against what the customer asked for.

API.SH filters on its side, but its filters are not trusted: in a real
response for ``budget_max=15000000`` several project records had a cheapest
2 BHK above that cap. Every live listing is therefore re-checked here.

Each check returns one of:

* ``match``          the listing satisfies the criterion;
* ``mismatch``       the listing clearly does not (it is dropped);
* ``unknown``        the listing does not say enough to tell;
* ``not_requested``  the customer did not ask for it.

Unknown is never treated as a mismatch, and never as a match.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from decimal import Decimal
from typing import Optional

from .models import NormalizedProperty

MATCH = "match"
MISMATCH = "mismatch"
UNKNOWN = "unknown"
NOT_REQUESTED = "not_requested"

# Property types API.SH documents for its ``type`` filter.
DOCUMENTED_TYPES = (
    "apartment",
    "independent_house",
    "villa",
    "builder_floor",
    "plot",
    "studio",
    "serviced_apartment",
    "farm_house",
    "agricultural_land",
)
# The one type string seen in a real record for a documented type.
_VERIFIED_TYPE_TEXT = {"apartment": "Residential Apartment"}

_BHK = re.compile(r"^\s*(\d+)\s*[-‑–— ]?\s*bhk\s*$", re.IGNORECASE)


def documented_type(property_type: Optional[str]) -> Optional[str]:
    """The documented ``type`` token for a customer's wording, or ``None``."""
    if not property_type:
        return None
    token = re.sub(r"[\s\-]+", "_", property_type.strip().lower())
    return token if token in DOCUMENTED_TYPES else None


def bedrooms_from_configuration(configuration: Optional[str]) -> Optional[int]:
    """``"2 BHK"`` -> 2. Anything else (for example ``"Studio"``) -> ``None``."""
    if not configuration:
        return None
    match = _BHK.match(configuration)
    return int(match.group(1)) if match else None


def check_budget(
    listing: NormalizedProperty,
    min_budget: Optional[Decimal],
    max_budget: Optional[Decimal],
) -> str:
    """Interval overlap, the same rule the SQL search uses for demo data."""
    if min_budget is None and max_budget is None:
        return NOT_REQUESTED
    low = listing.price if listing.price is not None else listing.price_max
    high = listing.price_max if listing.price_max is not None else listing.price
    if low is None or high is None or low > high:
        return UNKNOWN
    if max_budget is not None and low > max_budget:
        return MISMATCH
    if min_budget is not None and high < min_budget:
        return MISMATCH
    return MATCH


def check_bedrooms(listing: NormalizedProperty, configuration: Optional[str]) -> str:
    if not configuration:
        return NOT_REQUESTED
    wanted = bedrooms_from_configuration(configuration)
    if wanted is None or listing.bedrooms is None:
        return UNKNOWN
    return MATCH if listing.bedrooms == wanted else MISMATCH


def check_property_type(listing: NormalizedProperty, property_type: Optional[str]) -> str:
    """Type is never a mismatch: only one wording is verified (apartment)."""
    if not property_type:
        return NOT_REQUESTED
    token = documented_type(property_type)
    verified = _VERIFIED_TYPE_TEXT.get(token or "")
    if verified and listing.property_type_text == verified:
        return MATCH
    return UNKNOWN


def check_possession(listing: NormalizedProperty, cutoff: Optional[date]) -> str:
    """"On or before" ``cutoff``, without inventing a day of the month."""
    if cutoff is None:
        return NOT_REQUESTED
    if listing.possession_date is not None:
        return MATCH if listing.possession_date <= cutoff else MISMATCH
    if listing.possession_month is None:
        return UNKNOWN
    year, month = listing.possession_month
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    if first > cutoff:
        return MISMATCH  # the whole month is after the cutoff
    if last <= cutoff:
        return MATCH  # the whole month is on or before the cutoff
    return UNKNOWN  # the cutoff falls inside that month
