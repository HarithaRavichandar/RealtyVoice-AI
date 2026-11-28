"""SQLAlchemy model for the live-search credit ledger.

One row per live provider request. The row is created *before* the request is
sent (a conservative hold) and settled when the outcome is known, so a crash
mid-request still counts against the conversation's limit.

Columns:
- entry_id: primary key
- conversation_id: FK to conversations (rows go with the conversation)
- provider: the inventory source name, e.g. ``apish_99acres``
- request_id: caller-supplied identifier of this one search; unique per
  conversation and provider, so a retried or duplicated request can never be
  counted twice
- credits_consumed: credits the provider itself reported as used (the
  ``credits_used`` field of its response). Confirmed figures only.
- credits_unconfirmed: credits counted toward the limit although the provider
  did not confirm them (a request was sent but its cost is not known). Never
  presented as confirmed consumption.
- outcome: pending, ok, no_results, failed, ...
- failure_kind: the provider failure category, when there was one
- created_at, updated_at: DateTime UTC
"""
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)

from ..database import Base
from ._time import utc_now


class LiveCreditEntry(Base):
    __tablename__ = "live_credit_ledger"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "provider", "request_id",
            name="uq_live_credit_request",
        ),
        Index("ix_live_credit_conversation_id", "conversation_id"),
    )

    entry_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    provider = Column(String, nullable=False)
    request_id = Column(String, nullable=False)
    credits_consumed = Column(Integer, nullable=False, default=0)
    credits_unconfirmed = Column(Integer, nullable=False, default=0)
    outcome = Column(String, nullable=False, default="pending")
    failure_kind = Column(String, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)
