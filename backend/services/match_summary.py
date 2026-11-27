# -*- coding: utf-8 -*-
"""Per-property explanation of how a search result relates to the filters.

Search applies exact filters, so every returned property satisfies them. This
module records *which* filters were applied (with the property's own value)
and adds factual notes where a result only partly fits or lacks data, so the
agent can phrase an explanation without working it out itself.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from backend.schemas.property_search import PropertySearchItem


def _fmt(amount: Decimal) -> str:
    return f"{int(amount):,}"


def build_match_summary(
    item: PropertySearchItem,
    *,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    configuration: Optional[str] = None,
    min_budget: Optional[Decimal] = None,
    max_budget: Optional[Decimal] = None,
    possession_before: Any = None,
    availability: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    matched: List[Dict[str, Any]] = []
    notes: List[Dict[str, str]] = []

    def hit(name: str, requested: Any, actual: Any) -> None:
        matched.append({"filter": name, "requested": str(requested), "actual": str(actual)})

    def note(code: str, message: str) -> None:
        notes.append({"code": code, "message": message})

    if location:
        hit("location", location, item.location)
    if property_type:
        hit("property_type", property_type, item.property_type)
    if configuration:
        hit("configuration", configuration, item.configuration)
    if availability:
        hit("availability", availability, item.availability_status)
    if possession_before:
        hit("possession_before", possession_before, item.possession_date.date())

    # Same preference order as the search filter's price interval.
    low = next((v for v in (item.price_min, item.listing_price, item.price_max) if v is not None), None)
    high = next((v for v in (item.price_max, item.listing_price, item.price_min) if v is not None), None)
    if low is not None and high is not None:
        actual = _fmt(low) if low == high else f"{_fmt(low)}-{_fmt(high)}"
        if min_budget is not None or max_budget is not None:
            requested = (
                f"{_fmt(min_budget) if min_budget is not None else 'any'}"
                f" to {_fmt(max_budget) if max_budget is not None else 'any'}"
            )
            hit("budget", requested, actual)
        if max_budget is not None and high > max_budget:
            note(
                "price_range_above_budget",
                f"Prices run up to {_fmt(high)}, above the budget maximum of "
                f"{_fmt(max_budget)}; only the lower part of the range fits.",
            )
        if min_budget is not None and low < min_budget:
            note(
                "price_range_below_budget",
                f"Prices start at {_fmt(low)}, below the budget minimum of "
                f"{_fmt(min_budget)}; only the upper part of the range fits.",
            )
    else:
        note("price_unknown", "No price is recorded for this property.")

    if item.possession_date is None:
        note("possession_unknown", "No possession date is recorded.")
    if item.availability_status is None:
        note("availability_unknown", "Availability status is not recorded.")

    return {"matched": matched, "notes": notes}
