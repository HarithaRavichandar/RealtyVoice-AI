"""SQLAlchemy model for CallbackRequest.

Fields:
- callback_id: primary key
- lead_id: FK to leads (cascade delete when lead removed)
- conversation_id: FK to conversations (optional, set null on conversation delete)
- preferred_contact: string
- preferred_date: string (ISO date)
- preferred_time: string (HH:MM)
- reason: text (optional)
- status: Enum (Pending, Scheduled, Completed, Cancelled)
- created_at, updated_at: DateTime UTC
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from ..database import Base

class CallbackRequest(Base):
    __tablename__ = "callback_requests"
    __table_args__ = (
        Index("ix_callbacks_lead_id", "lead_id"),
        Index("ix_callbacks_conversation_id", "conversation_id"),
    )

    callback_id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer, ForeignKey("leads.lead_id", ondelete="CASCADE"), nullable=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="SET NULL"), nullable=True)
    preferred_contact = Column(String, nullable=False)
    preferred_date = Column(String, nullable=False)  # ISO date string
    preferred_time = Column(String, nullable=False)  # HH:MM
    reason = Column(Text, nullable=True)
    status = Column(
        Enum("Pending", "Scheduled", "Completed", "Cancelled", name="callback_status"),
        nullable=False,
        default="Pending",
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships – lead required, conversation optional
    lead = relationship("Lead", back_populates="callbacks")
    conversation = relationship("Conversation", back_populates="callbacks")

