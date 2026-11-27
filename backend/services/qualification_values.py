# -*- coding: utf-8 -*-
"""Allowed values (and customer phrasings) for the lead qualification fields.

Shared by the tool schemas and the stored-requirements schema so the model
cannot save a value the scoring code does not understand.
"""

from typing import Dict, Optional, Tuple

PURCHASE_TIMELINES: Tuple[str, ...] = (
    "immediate",        # within about a month
    "within_3_months",
    "within_6_months",
    "within_12_months",
    "later",            # more than a year away
    "exploring",        # no timeline yet
)
VISIT_READINESS: Tuple[str, ...] = ("ready_to_visit", "considering", "not_ready")
CONTACT_PREFERENCES: Tuple[str, ...] = ("phone_call", "whatsapp", "sms", "email")

_SYNONYMS: Dict[str, Dict[str, str]] = {
    "purchase_timeline": {
        "now": "immediate", "asap": "immediate", "right away": "immediate",
        "this month": "immediate", "within a month": "immediate",
        "1 month": "immediate", "one month": "immediate",
        "3 months": "within_3_months", "three months": "within_3_months",
        "within 3 months": "within_3_months", "next quarter": "within_3_months",
        "6 months": "within_6_months", "six months": "within_6_months",
        "within 6 months": "within_6_months", "half a year": "within_6_months",
        "12 months": "within_12_months", "one year": "within_12_months",
        "1 year": "within_12_months", "within a year": "within_12_months",
        "this year": "within_12_months", "within 1 year": "within_12_months",
        "next year": "later", "later": "later", "more than a year": "later",
        "no timeline": "exploring", "not sure": "exploring",
        "just looking": "exploring", "just exploring": "exploring",
    },
    "visit_readiness": {
        "ready": "ready_to_visit", "yes": "ready_to_visit",
        "ready to visit": "ready_to_visit", "wants to visit": "ready_to_visit",
        "can visit": "ready_to_visit", "site visit": "ready_to_visit",
        "maybe": "considering", "thinking": "considering",
        "considering": "considering", "not sure": "considering",
        "no": "not_ready", "not ready": "not_ready", "later": "not_ready",
        "not now": "not_ready",
    },
    "contact_preference": {
        "phone": "phone_call", "call": "phone_call", "phone call": "phone_call",
        "voice": "phone_call", "whatsapp": "whatsapp", "wa": "whatsapp",
        "text": "sms", "text message": "sms", "sms": "sms",
        "mail": "email", "e mail": "email", "email": "email",
    },
}

_ALLOWED = {
    "purchase_timeline": PURCHASE_TIMELINES,
    "visit_readiness": VISIT_READINESS,
    "contact_preference": CONTACT_PREFERENCES,
}


def _normalize(field: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} cannot be empty")
    key = " ".join(value.lower().replace("-", " ").replace("_", " ").split())
    canonical = value.strip().lower().replace(" ", "_").replace("-", "_")
    if canonical in _ALLOWED[field]:
        return canonical
    mapped = _SYNONYMS[field].get(key)
    if mapped is not None:
        return mapped
    raise ValueError(f"{field} must be one of: {', '.join(_ALLOWED[field])}")


def normalize_purchase_timeline(value: Optional[str]) -> Optional[str]:
    return _normalize("purchase_timeline", value)


def normalize_visit_readiness(value: Optional[str]) -> Optional[str]:
    return _normalize("visit_readiness", value)


def normalize_contact_preference(value: Optional[str]) -> Optional[str]:
    return _normalize("contact_preference", value)
