"""SQLAlchemy model for Message.

Fields:
- message_id: primary key
- conversation_id: foreign key to conversations
- turn_id: integer indicating message order
- role: Enum (customer, assistant, tool)
- content: text
- created_at: DateTime UTC
"""
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Enum, Index, String, JSON
from sqlalchemy.orm import relationship
from ..database import Base
from ._time import utc_now

class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
    )

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    turn_id = Column(Integer, nullable=False)
    role = Column(Enum("customer", "assistant", "tool", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    tool_name = Column(String, nullable=True)
    tool_call_id = Column(String, nullable=True)
    tool_arguments = Column(JSON, nullable=True)
    tool_status = Column(String, nullable=True)
    tool_result = Column(JSON, nullable=True)
    tool_error = Column(Text, nullable=True)
    conversation = relationship("Conversation", back_populates="messages")

