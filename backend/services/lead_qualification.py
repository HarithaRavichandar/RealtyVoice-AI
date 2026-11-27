# -*- coding: utf-8 -*-
"""Transparent lead qualification: buyer intent, score and status.

Everything here is deterministic and based on measurable evidence already in
the conversation (saved requirements, contact details, open callbacks). The
score is the sum of five signals, and the per-signal points are stored with
the lead so a score can always be explained.

    budget clarity        25  both bounds 25, one bound 15
    location specificity  20  one data label 20, several labels 12, unmatched 6
    purchase timeline     20  immediate 20 ... exploring/unknown 0
    contact completeness  20  phone 10, email 5, name 5
    callback request      15  an open (Pending/Scheduled) callback exists

Status: Hot >= 70, Warm >= 40, Cold >= 15, otherwise Unqualified.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.callback import CallbackRequest
from backend.models.consent import ConsentRecord
from backend.models.conversation import Conversation
from backend.models.lead import Lead

OPEN_CALLBACK_STATUSES = ("Pending", "Scheduled")

INTENTS = ("browsing", "exploring", "actively_searching", "visit_ready", "callback_requesting")

TIMELINE_POINTS = {
    "immediate": 20,
    "within_3_months": 16,
    "within_6_months": 12,
    "within_12_months": 8,
    "later": 4,
    "exploring": 0,
}

MAX_POINTS = {
    "budget_clarity": 25,
    "location_specificity": 20,
    "purchase_timeline": 20,
    "contact_completeness": 20,
    "callback_request": 15,
}

HOT_THRESHOLD = 70
WARM_THRESHOLD = 40
COLD_THRESHOLD = 15


@dataclass
class Evidence:
    """Facts the intent and score are computed from."""

    has_budget_min: bool = False
    has_budget_max: bool = False
    location: Optional[str] = None
    location_labels: Optional[int] = None  # distinct data labels the location matches
    has_configuration: bool = False
    has_property_type: bool = False
    purchase_timeline: Optional[str] = None
    visit_readiness: Optional[str] = None
    has_name: bool = False
    has_phone: bool = False
    has_email: bool = False
    open_callback: bool = False


def classify_intent(e: Evidence) -> str:
    """Highest applicable stage wins: callback > visit > searching > exploring."""
    if e.open_callback:
        return "callback_requesting"
    if e.visit_readiness == "ready_to_visit" and (e.has_phone or e.has_email):
        return "visit_ready"
    if (
        e.location
        and (e.has_budget_min or e.has_budget_max)
        and (e.has_configuration or e.has_property_type)
    ):
        return "actively_searching"
    if (
        e.location or e.has_budget_min or e.has_budget_max
        or e.has_configuration or e.has_property_type
    ):
        return "exploring"
    return "browsing"


def _signal(points: int, maximum: int, reason: str) -> Dict[str, Any]:
    return {"points": points, "max": maximum, "reason": reason}


def score_lead(e: Evidence) -> Dict[str, Any]:
    """Return ``{"score", "status", "signals"}``; signals explain each point."""
    signals: Dict[str, Dict[str, Any]] = {}

    if e.has_budget_min and e.has_budget_max:
        signals["budget_clarity"] = _signal(25, 25, "both a minimum and a maximum budget are known")
    elif e.has_budget_min or e.has_budget_max:
        signals["budget_clarity"] = _signal(15, 25, "only one budget bound is known")
    else:
        signals["budget_clarity"] = _signal(0, 25, "no budget given")

    if not e.location:
        signals["location_specificity"] = _signal(0, 20, "no location given")
    elif e.location_labels == 1:
        signals["location_specificity"] = _signal(20, 20, "location matches one specific area in the data")
    elif e.location_labels and e.location_labels > 1:
        signals["location_specificity"] = _signal(12, 20, "location is broad (matches several areas)")
    else:
        signals["location_specificity"] = _signal(6, 20, "location given but not found in the data")

    timeline = e.purchase_timeline
    signals["purchase_timeline"] = _signal(
        TIMELINE_POINTS.get(timeline, 0),
        20,
        f"timeline is {timeline}" if timeline else "no timeline given",
    )

    contact = (10 if e.has_phone else 0) + (5 if e.has_email else 0) + (5 if e.has_name else 0)
    have = [n for n, ok in (("phone", e.has_phone), ("email", e.has_email), ("name", e.has_name)) if ok]
    signals["contact_completeness"] = _signal(
        contact, 20, f"have {', '.join(have)}" if have else "no contact details"
    )

    signals["callback_request"] = _signal(
        15 if e.open_callback else 0,
        15,
        "an open callback request exists" if e.open_callback else "no open callback request",
    )

    score = sum(s["points"] for s in signals.values())
    return {"score": score, "status": status_for_score(score), "signals": signals}


def status_for_score(score: int) -> str:
    if score >= HOT_THRESHOLD:
        return "Hot"
    if score >= WARM_THRESHOLD:
        return "Warm"
    if score >= COLD_THRESHOLD:
        return "Cold"
    return "Unqualified"


# ---------------------------------------------------------------------------
# Evidence gathering and persistence
# ---------------------------------------------------------------------------

def _contact_storage_revoked(session: Session, lead_id: int) -> bool:
    latest = (
        session.query(ConsentRecord)
        .filter(
            ConsentRecord.lead_id == lead_id,
            ConsentRecord.consent_type == "contact_storage",
        )
        .order_by(ConsentRecord.recorded_at.desc(), ConsentRecord.consent_id.desc())
        .first()
    )
    return latest is not None and not latest.granted


def build_evidence(
    session: Session,
    lead: Lead,
    requirements: Optional[Dict[str, Any]],
    location_labels: Optional[int] = None,
) -> Evidence:
    req = requirements or {}
    revoked = _contact_storage_revoked(session, lead.lead_id)
    open_callback = (
        session.query(CallbackRequest.callback_id)
        .filter(
            CallbackRequest.lead_id == lead.lead_id,
            CallbackRequest.status.in_(OPEN_CALLBACK_STATUSES),
        )
        .first()
        is not None
    )
    return Evidence(
        has_budget_min=req.get("budget_min") is not None,
        has_budget_max=req.get("budget_max") is not None,
        location=req.get("location"),
        location_labels=location_labels,
        has_configuration=bool(req.get("configuration")),
        has_property_type=bool(req.get("property_type")),
        purchase_timeline=req.get("purchase_timeline"),
        visit_readiness=req.get("visit_readiness"),
        # Contact the customer withdrew storage consent for does not count.
        has_name=bool(lead.customer_name) and not revoked,
        has_phone=bool(lead.phone) and not revoked,
        has_email=bool(lead.email) and not revoked,
        open_callback=open_callback,
    )


def count_location_labels(session: Session, location: Optional[str]) -> Optional[int]:
    """How many distinct data labels the saved location matches."""
    if not location:
        return None
    from backend.services.location_normalizer import resolve_locations
    from backend.services.property_search import PropertySearchService

    service = PropertySearchService(session_factory=lambda: session, close_session=False)
    return len(resolve_locations(location, service.distinct_locations()))


def sync_lead_from_requirements(lead: Lead, requirements: Dict[str, Any]) -> None:
    """Copy the qualification fields the customer has stated onto the lead."""
    for field in ("purchase_timeline", "visit_readiness", "contact_preference"):
        if field in requirements:
            setattr(lead, field, requirements.get(field))


def refresh_lead_qualification(
    session: Session, conversation: Conversation
) -> Optional[Dict[str, Any]]:
    """Recompute and store intent, score and status. ``None`` if no lead yet."""
    if conversation.lead_id is None:
        return None
    lead = session.get(Lead, conversation.lead_id)
    if lead is None:
        return None
    requirements = conversation.temp_requirements or {}
    sync_lead_from_requirements(lead, requirements)
    evidence = build_evidence(
        session, lead, requirements,
        location_labels=count_location_labels(session, requirements.get("location")),
    )
    result = score_lead(evidence)
    intent = classify_intent(evidence)
    lead.buyer_intent = intent
    lead.qualification_score = Decimal(result["score"])
    lead.qualification_status = result["status"]
    lead.score_breakdown = result["signals"]
    session.flush()
    return {
        "buyer_intent": intent,
        "score": result["score"],
        "status": result["status"],
        "signals": result["signals"],
    }
