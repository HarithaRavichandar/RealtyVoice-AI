# -*- coding: utf-8 -*-
"""Callback lifecycle: duplicate prevention, rescheduling, cancel, status.

A lead has at most one open callback (status Pending or Scheduled). Asking
again for the same slot returns the existing one; asking for a different slot
reschedules it instead of creating a second request.
"""

from datetime import date
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.models.callback import CallbackRequest

OPEN_STATUSES = ("Pending", "Scheduled")

ALLOWED_TRANSITIONS = {
    "Pending": {"Scheduled", "Cancelled", "Completed"},
    "Scheduled": {"Pending", "Completed", "Cancelled"},
    "Completed": set(),
    "Cancelled": set(),
}


def open_callbacks(session: Session, lead_id: int) -> List[CallbackRequest]:
    return (
        session.query(CallbackRequest)
        .filter(CallbackRequest.lead_id == lead_id, CallbackRequest.status.in_(OPEN_STATUSES))
        .order_by(CallbackRequest.callback_id)
        .all()
    )


def create_or_reschedule(
    session: Session,
    *,
    lead_id: int,
    conversation_id: Optional[int],
    preferred_contact: str,
    preferred_date: date,
    preferred_time: str,
    reason: str,
) -> Tuple[CallbackRequest, str]:
    """Return ``(callback, outcome)`` where outcome is created, duplicate or rescheduled."""
    iso = preferred_date.isoformat()
    existing = open_callbacks(session, lead_id)
    if existing:
        cb = existing[0]
        if (cb.preferred_contact, cb.preferred_date, cb.preferred_time) == (preferred_contact, iso, preferred_time):
            return cb, "duplicate"
        cb.preferred_contact = preferred_contact
        cb.preferred_date = iso
        cb.preferred_time = preferred_time
        cb.reason = reason
        cb.conversation_id = conversation_id
        cb.status = "Pending"  # a changed slot needs confirming again
        session.flush()
        return cb, "rescheduled"
    cb = CallbackRequest(
        lead_id=lead_id,
        conversation_id=conversation_id,
        preferred_contact=preferred_contact,
        preferred_date=iso,
        preferred_time=preferred_time,
        reason=reason,
        status="Pending",
    )
    session.add(cb)
    session.flush()
    return cb, "created"


def cancel_open_callbacks(session: Session, lead_id: int) -> List[int]:
    """Cancel every open callback for the lead; return the cancelled ids."""
    cancelled = []
    for cb in open_callbacks(session, lead_id):
        cb.status = "Cancelled"
        cancelled.append(cb.callback_id)
    session.flush()
    return cancelled


def update_callback_status(session: Session, callback_id: int, new_status: str) -> CallbackRequest:
    """Move a callback to ``new_status`` if the transition is allowed."""
    cb = session.get(CallbackRequest, callback_id)
    if cb is None:
        raise LookupError(f"callback {callback_id} not found")
    if new_status not in ALLOWED_TRANSITIONS:
        raise ValueError(f"unknown callback status: {new_status}")
    if new_status not in ALLOWED_TRANSITIONS[cb.status]:
        raise ValueError(f"cannot change callback from {cb.status} to {new_status}")
    cb.status = new_status
    session.flush()
    return cb
