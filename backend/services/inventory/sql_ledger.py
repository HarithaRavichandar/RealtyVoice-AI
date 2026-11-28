# -*- coding: utf-8 -*-
"""Durable, database-backed credit ledger for live inventory.

Written to the same SQLAlchemy session as the rest of the agent turn (the
agent loop never commits; the router does), so it follows the project's
transaction convention. Because the router rolls the whole turn back when a
later step fails - which would erase a record of credits already spent - every
entry is also remembered on ``session.info`` and :func:`restore_after_rollback`
writes them again, in their own transaction, once the rollback is done.

What is recorded
----------------
* ``credits_consumed``    - credits the provider *reported* as used.
* ``credits_unconfirmed`` - a conservative hold for a request that was sent but
  whose cost was not reported (or is still pending). It counts toward the
  per-conversation limit but is never called confirmed consumption.

Idempotency
-----------
A request is identified by ``(conversation, provider, request_id)``. Reserving
the same identifier twice reports ``duplicate`` and records nothing more, so a
retried or repeated tool call cannot be counted - or sent - twice.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.models._time import utc_now
from backend.models.live_credit import LiveCreditEntry

logger = logging.getLogger(__name__)

RESERVED = "reserved"
DUPLICATE = "duplicate"

PENDING = "pending"

_INFO_KEY = "live_credit_entries"

Key = Tuple[int, str, str]


def _conversation_id(conversation_key: str) -> int:
    try:
        return int(conversation_key)
    except (TypeError, ValueError):
        raise ValueError("conversation_key must be a numeric conversation id") from None


class SqlCreditLedger:
    """Credit ledger stored in the ``live_credit_ledger`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- reading ---------------------------------------------------------
    def used(self, conversation_key: str) -> int:
        """Credits counted toward the limit: confirmed plus unconfirmed holds."""
        total = (
            self._session.query(
                func.coalesce(
                    func.sum(
                        LiveCreditEntry.credits_consumed
                        + LiveCreditEntry.credits_unconfirmed
                    ),
                    0,
                )
            )
            .filter(LiveCreditEntry.conversation_id == _conversation_id(conversation_key))
            .scalar()
        )
        return int(total or 0)

    def confirmed(self, conversation_key: str) -> int:
        total = (
            self._session.query(func.coalesce(func.sum(LiveCreditEntry.credits_consumed), 0))
            .filter(LiveCreditEntry.conversation_id == _conversation_id(conversation_key))
            .scalar()
        )
        return int(total or 0)

    def entries(self, conversation_key: str) -> list[LiveCreditEntry]:
        return (
            self._session.query(LiveCreditEntry)
            .filter(LiveCreditEntry.conversation_id == _conversation_id(conversation_key))
            .order_by(LiveCreditEntry.entry_id)
            .all()
        )

    # -- writing ---------------------------------------------------------
    def _find(self, key: Key) -> Optional[LiveCreditEntry]:
        conversation_id, provider, request_id = key
        return (
            self._session.query(LiveCreditEntry)
            .filter(
                LiveCreditEntry.conversation_id == conversation_id,
                LiveCreditEntry.provider == provider,
                LiveCreditEntry.request_id == request_id,
            )
            .one_or_none()
        )

    def _remember(self, key: Key, consumed: int, unconfirmed: int, outcome: str,
                  failure_kind: Optional[str]) -> None:
        store: Dict[Key, dict] = self._session.info.setdefault(_INFO_KEY, {})
        store[key] = {
            "credits_consumed": consumed,
            "credits_unconfirmed": unconfirmed,
            "outcome": outcome,
            "failure_kind": failure_kind,
        }

    def reserve(
        self,
        conversation_key: str,
        provider: str,
        request_id: str,
        *,
        hold: int = 1,
    ) -> str:
        """Record a pending hold before the request is sent.

        Returns ``"reserved"``, or ``"duplicate"`` when this request id was
        already recorded (nothing is added and the caller must not send it).
        """
        if not request_id or not str(request_id).strip():
            raise ValueError("request_id cannot be empty")
        key: Key = (_conversation_id(conversation_key), provider, str(request_id))
        if self._find(key) is not None:
            return DUPLICATE
        values = {
            "conversation_id": key[0],
            "provider": provider,
            "request_id": key[2],
            "credits_consumed": 0,
            "credits_unconfirmed": hold,
            "outcome": PENDING,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        if not self._insert_if_absent(values):
            return DUPLICATE
        self._remember(key, 0, hold, PENDING, None)
        return RESERVED

    def _insert_if_absent(self, values: dict) -> bool:
        """Insert one row unless its request id exists. True if inserted.

        Uses the database's own conflict handling instead of a SAVEPOINT:
        SQLite's legacy transaction mode can commit an outermost SAVEPOINT
        early, which would make the ledger write behave differently from the
        rest of the turn. A conflict here leaves the transaction intact.
        """
        dialect = self._session.get_bind().dialect.name
        table = LiveCreditEntry.__table__
        if dialect in ("sqlite", "postgresql"):
            module = sqlite if dialect == "sqlite" else postgresql
            statement = module.insert(table).values(**values).on_conflict_do_nothing(
                index_elements=["conversation_id", "provider", "request_id"]
            )
            return bool(self._session.execute(statement).rowcount)
        self._session.add(LiveCreditEntry(**values))  # other databases: pre-checked
        self._session.flush()
        return True

    def settle(
        self,
        conversation_key: str,
        provider: str,
        request_id: str,
        *,
        consumed: int,
        unconfirmed: int,
        outcome: str,
        failure_kind: Optional[str] = None,
    ) -> None:
        """Replace the pending hold with the final figures. Idempotent."""
        if consumed < 0 or unconfirmed < 0:
            raise ValueError("credits cannot be negative")
        key: Key = (_conversation_id(conversation_key), provider, str(request_id))
        entry = self._find(key)
        if entry is None:
            raise KeyError("no reservation exists for this request")
        entry.credits_consumed = consumed
        entry.credits_unconfirmed = unconfirmed
        entry.outcome = outcome
        entry.failure_kind = failure_kind
        self._session.flush()
        self._remember(key, consumed, unconfirmed, outcome, failure_kind)


def restore_after_rollback(session: Session) -> int:
    """Write this session's ledger entries again after a rolled-back turn.

    Call it right after ``session.rollback()``. Credits that were spent on a
    live request stay on the record even though the rest of the turn was
    discarded. Entries that already exist are brought up to date rather than
    repeated. Returns how many entries were written or updated. Never raises: a failure here is logged and the
    original error is the one that matters.
    """
    store: Dict[Key, dict] = session.info.pop(_INFO_KEY, None) or {}
    written = 0
    if not store:
        return written
    try:
        for (conversation_id, provider, request_id), values in store.items():
            exists = (
                session.query(LiveCreditEntry.entry_id)
                .filter(
                    LiveCreditEntry.conversation_id == conversation_id,
                    LiveCreditEntry.provider == provider,
                    LiveCreditEntry.request_id == request_id,
                )
                .first()
            )
            if exists is not None:
                # Already stored (for example the early hold): bring it up
                # to the last state this turn recorded.
                session.query(LiveCreditEntry).filter(
                    LiveCreditEntry.entry_id == exists[0]
                ).update(values)
            else:
                session.add(
                    LiveCreditEntry(
                        conversation_id=conversation_id,
                        provider=provider,
                        request_id=request_id,
                        **values,
                    )
                )
            written += 1
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Could not restore live credit ledger entries after rollback")
        return 0
    return written


def forget_pending(session: Session) -> None:
    """Drop remembered entries once the turn has been committed."""
    session.info.pop(_INFO_KEY, None)
