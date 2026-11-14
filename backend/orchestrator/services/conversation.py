"""Conversation service for the orchestrator."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ...models import Conversation

def create_conversation(session: Session, channel: str) -> Conversation:
    """Create an anonymous conversation without committing the transaction."""
    conv = Conversation(lead_id=None, channel=channel, external_session_id=uuid.uuid4().hex, temp_requirements=None)
    session.add(conv)
    session.flush()
    return conv

def get_conversation_by_session_id(session: Session, session_id: str) -> Optional[Conversation]:
    """Retrieve a conversation using its external session ID."""
    return session.query(Conversation).filter(Conversation.external_session_id == session_id).first()

def update_temp_requirements(session: Session, conversation_id: int, requirements: Dict[str, Any]) -> None:
    """Update temporary requirements without committing the transaction."""
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError("Conversation not found")
    conv.temp_requirements = requirements
