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
    """Update temporary requirements safely.

    The ``Conversation.temp_requirements`` column is a JSON column, which can only store
    JSON‑serialisable primitive types. ``Decimal`` and ``date``/``datetime`` values are
    not directly serialisable, so we coerce them to strings:
    * ``Decimal`` → ``str`` (preserves exact precision)
    * ``date`` / ``datetime`` → ISO‑8601 ``str``

    Nested structures are handled recursively.
    """
    from decimal import Decimal
    from datetime import date, datetime
    import json

    def _make_serialisable(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: _make_serialisable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_make_serialisable(v) for v in value]
        return value

    serialisable_req = _make_serialisable(requirements)
    # Validate that the result can be JSON‑encoded (will raise if not)
    json.dumps(serialisable_req)
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError("Conversation not found")
    conv.temp_requirements = serialisable_req