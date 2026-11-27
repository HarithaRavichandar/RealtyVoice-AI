# -*- coding: utf-8 -*-
"""Rank properties against a customer's saved requirements (task 3.4).

``search_properties`` applies exact filters. Matching instead grades each
property on every requirement the customer has stated and returns the best
fits with an explanation per criterion, so near-misses (for example slightly
over budget) can be offered honestly and clearly labelled.

Rules
-----
* Location is a gate: when a location is saved, only properties in a matching
  data label are considered. Nothing outside it is ever recommended.
* Every other saved requirement is scored. The final score is 0-100, computed
  only over the requirements the customer actually stated.
* Missing property data is "unknown" and earns partial credit, never a match.
* Properties scoring below ``MIN_SCORE`` are not returned.

Weights: configuration 25, budget 35, property type 15, possession 10,
availability 15.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from backend.models.lead_property_match import LeadPropertyMatch
from backend.models.property import Property
from backend.services.location_normalizer import resolve_locations
from backend.services.property_search import PropertySearchService

WEIGHTS = {
    "configuration": 25,
    "budget": 35,
    "property_type": 15,
    "possession": 10,
    "availability": 15,
}
MIN_SCORE = 50
POOL_SIZE = 300
UNKNOWN_CREDIT = 0.3
PARTIAL_BUDGET_CREDIT = 0.6
NEAR_BUDGET_CREDIT = 0.25
NEAR_BUDGET_MARGIN = Decimal("0.10")  # up to 10% beyond the stated bounds


def _norm_config(value: Optional[str]) -> str:
    if not value:
        return ""
    for dash in ("‑", "–", "—", "-"):
        value = value.replace(dash, " ")
    return " ".join(value.lower().split())


def _fmt(amount: Decimal) -> str:
    return f"{int(amount):,}"


@dataclass
class Criterion:
    name: str
    requested: str
    actual: Optional[str]
    status: str  # match | partial | mismatch | unknown
    credit: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.name,
            "requested": self.requested,
            "actual": self.actual,
            "status": self.status,
        }


@dataclass
class MatchResult:
    prop: Property
    score: int
    criteria: List[Criterion] = field(default_factory=list)


def _price_bounds(prop: Property) -> tuple[Optional[Decimal], Optional[Decimal]]:
    low = next((v for v in (prop.price_min, prop.listing_price, prop.price_max) if v is not None), None)
    high = next((v for v in (prop.price_max, prop.listing_price, prop.price_min) if v is not None), None)
    return (Decimal(low) if low is not None else None, Decimal(high) if high is not None else None)


def _budget_criterion(prop: Property, bmin: Optional[Decimal], bmax: Optional[Decimal]) -> Criterion:
    requested = f"{_fmt(bmin) if bmin is not None else 'any'} to {_fmt(bmax) if bmax is not None else 'any'}"
    low, high = _price_bounds(prop)
    if low is None or high is None:
        return Criterion("budget", requested, None, "unknown", UNKNOWN_CREDIT)
    actual = _fmt(low) if low == high else f"{_fmt(low)}-{_fmt(high)}"
    within_low = bmin is None or low >= bmin
    within_high = bmax is None or high <= bmax
    if within_low and within_high:
        return Criterion("budget", requested, actual, "match", 1.0)
    overlaps = (bmax is None or low <= bmax) and (bmin is None or high >= bmin)
    if overlaps:
        return Criterion("budget", requested, actual, "partial", PARTIAL_BUDGET_CREDIT)
    # No overlap: slightly over the maximum / under the minimum is a labelled near miss.
    if bmax is not None and low > bmax and low <= bmax * (1 + NEAR_BUDGET_MARGIN):
        return Criterion("budget", requested, actual, "partial", NEAR_BUDGET_CREDIT)
    if bmin is not None and high < bmin and high >= bmin * (1 - NEAR_BUDGET_MARGIN):
        return Criterion("budget", requested, actual, "partial", NEAR_BUDGET_CREDIT)
    return Criterion("budget", requested, actual, "mismatch", 0.0)


def evaluate(prop: Property, req: Dict[str, Any]) -> MatchResult:
    """Score one property against saved requirements (pure, no database)."""
    criteria: List[Criterion] = []

    if req.get("configuration"):
        want = req["configuration"]
        have = prop.configuration
        if not have:
            criteria.append(Criterion("configuration", want, None, "unknown", UNKNOWN_CREDIT))
        elif _norm_config(have) == _norm_config(want):
            criteria.append(Criterion("configuration", want, have, "match", 1.0))
        else:
            criteria.append(Criterion("configuration", want, have, "mismatch", 0.0))

    bmin, bmax = req.get("budget_min"), req.get("budget_max")
    if bmin is not None or bmax is not None:
        criteria.append(_budget_criterion(
            prop,
            Decimal(str(bmin)) if bmin is not None else None,
            Decimal(str(bmax)) if bmax is not None else None,
        ))

    if req.get("property_type"):
        want = req["property_type"]
        have = prop.property_type
        if not have:
            criteria.append(Criterion("property_type", want, None, "unknown", UNKNOWN_CREDIT))
        elif have.strip().lower() == want.strip().lower():
            criteria.append(Criterion("property_type", want, have, "match", 1.0))
        else:
            criteria.append(Criterion("property_type", want, have, "mismatch", 0.0))

    if req.get("possession_before"):
        cutoff = req["possession_before"]
        if isinstance(cutoff, str):
            cutoff = date.fromisoformat(cutoff)
        have = prop.possession_date
        if have is None:
            criteria.append(Criterion("possession", str(cutoff), None, "unknown", UNKNOWN_CREDIT))
        elif have < datetime.combine(cutoff, time.min) + timedelta(days=1):
            criteria.append(Criterion("possession", str(cutoff), str(have.date()), "match", 1.0))
        else:
            criteria.append(Criterion("possession", str(cutoff), str(have.date()), "mismatch", 0.0))

    if req.get("availability"):
        want = req["availability"]
        have = prop.availability_status
        if not have:
            criteria.append(Criterion("availability", want, None, "unknown", UNKNOWN_CREDIT))
        elif have.strip().lower() == want.strip().lower():
            criteria.append(Criterion("availability", want, have, "match", 1.0))
        else:
            criteria.append(Criterion("availability", want, have, "mismatch", 0.0))

    if not criteria:
        # Location-only requirements: everything in the area fits equally.
        return MatchResult(prop, 100, [])
    total = sum(WEIGHTS[c.name] for c in criteria)
    earned = sum(WEIGHTS[c.name] * c.credit for c in criteria)
    return MatchResult(prop, round(100 * earned / total), criteria)


def _pool_rank_expr(req: Dict[str, Any]):
    """Cheap SQL approximation used only to choose which rows to score."""
    parts = []
    if req.get("property_type"):
        parts.append(case((func.lower(Property.property_type) == req["property_type"].strip().lower(), WEIGHTS["property_type"]), else_=0))
    if req.get("configuration"):
        col = func.lower(func.trim(func.replace(func.replace(Property.configuration, "-", " "), "–", " ")))
        parts.append(case((col == _norm_config(req["configuration"]), WEIGHTS["configuration"]), else_=0))
    low = func.coalesce(Property.price_min, Property.listing_price, Property.price_max)
    high = func.coalesce(Property.price_max, Property.listing_price, Property.price_min)
    bmin, bmax = req.get("budget_min"), req.get("budget_max")
    if bmin is not None or bmax is not None:
        # Widen by the near-miss margin so near misses reach the scorer.
        conds = [low.isnot(None), high.isnot(None)]
        if bmax is not None:
            conds.append(low <= Decimal(str(bmax)) * (1 + NEAR_BUDGET_MARGIN))
        if bmin is not None:
            conds.append(high >= Decimal(str(bmin)) * (1 - NEAR_BUDGET_MARGIN))
        parts.append(case((and_(*conds), WEIGHTS["budget"]), else_=0))
    if not parts:
        return None
    expr = parts[0]
    for p in parts[1:]:
        expr = expr + p
    return expr


def match_properties(
    session: Session, requirements: Dict[str, Any], limit: int = 5
) -> List[MatchResult]:
    """Return up to ``limit`` best matches (score >= MIN_SCORE), best first."""
    stmt = select(Property)
    location = requirements.get("location")
    if location:
        service = PropertySearchService(session_factory=lambda: session, close_session=False)
        labels = resolve_locations(location, service.distinct_locations())
        if not labels:
            return []
        stmt = stmt.where(Property.location.in_(labels))
    rank = _pool_rank_expr(requirements)
    if rank is not None:
        stmt = stmt.order_by(rank.desc(), Property.property_id)
    else:
        stmt = stmt.order_by(Property.property_id)
    pool = session.execute(stmt.limit(POOL_SIZE)).scalars().all()

    results = [evaluate(p, requirements) for p in pool]
    results = [r for r in results if r.score >= MIN_SCORE]
    results.sort(key=lambda r: (-r.score, r.prop.property_id))
    return results[:limit]


def save_matches(session: Session, lead_id: int, results: List[MatchResult]) -> None:
    """Upsert LeadPropertyMatch rows for the returned matches."""
    for r in results:
        row = (
            session.query(LeadPropertyMatch)
            .filter_by(lead_id=lead_id, property_id=r.prop.property_id)
            .one_or_none()
        )
        if row is None:
            session.add(LeadPropertyMatch(lead_id=lead_id, property_id=r.prop.property_id, match_score=float(r.score)))
        else:
            row.match_score = float(r.score)
    session.flush()
