"""SQLAlchemy model for ConsentRecord.

Fields:
- consent_id: primary key
- lead_id: FK to leads (on delete set null to preserve history)
- consent_type: Enum (contact_storage, callback_contact, marketing_communications)
- granted: Boolean
- recorded_at: DateTime UTC (default now)
- source: String (e.g., web, phone)
- conversation_id: FK to conversations (optional, set null on conversation delete)
- policy_version: String
- created_at, updated_at: DateTime UTC
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from ..database import Base
from ._time import utc_now

class ConsentRecord(Base):
    __tablename__ = "consent_records"
    __table_args__ = (
        Index("ix_consents_lead_id", "lead_id"),
        Index("ix_consents_conversation_id", "conversation_id"),
    )

    consent_id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer, ForeignKey("leads.lead_id", ondelete="SET NULL"), nullable=True)
    consent_type = Column(
        Enum(
            "contact_storage",
            "callback_contact",
            "marketing_communications",
            name="consent_type",
        ),
        nullable=False,
    )
    granted = Column(Boolean, nullable=False)
    recorded_at = Column(DateTime, default=utc_now, nullable=False)
    source = Column(String, nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="SET NULL"), nullable=True)
    policy_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships – preserve history even if lead or conversation is deleted
    lead = relationship("Lead", back_populates="consents", passive_deletes=True)
    conversation = relationship("Conversation", back_populates="consents", passive_deletes=True)

