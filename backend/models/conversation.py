"""SQLAlchemy model for Conversation.

Fields:
- conversation_id: primary key
- lead_id: foreign key to leads.lead_id
- channel: Enum (browser_voice, browser_text, telephone)
- external_session_id: string
- created_at, updated_at: DateTime UTC
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum, Index
from sqlalchemy.orm import relationship
from ..database import Base
from ._time import utc_now

class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_lead_id", "lead_id"),)

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer, ForeignKey("leads.lead_id", ondelete="CASCADE"), nullable=True)
    # New column for provisional requirements when conversation is anonymous
    temp_requirements = Column(JSON, nullable=True)
    channel = Column(
        Enum("browser_voice", "browser_text", "telephone", name="channel_enum"),
        nullable=False,
    )
    external_session_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    lead = relationship("Lead", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    callbacks = relationship("CallbackRequest", back_populates="conversation", passive_deletes=True)
    consents = relationship("ConsentRecord", back_populates="conversation", passive_deletes=True)

