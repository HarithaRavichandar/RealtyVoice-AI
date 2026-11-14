"""Context builder utilities for the orchestrator.

Assembles a structured context for the central agent core from a conversation
record, its message history, and the latest incoming user message.
"""

from __future__ import annotations

import logging
from pydantic import ValidationError

from sqlalchemy.orm import Session

from ..services.message import get_conversation_messages
from .session_manager import resolve_session
from ..schemas import CustomerRequirements
from typing import List, Optional
from ...models import Message


def _validate_history_limit(limit: int) -> int:
    # Reject booleans explicitly because bool is a subclass of int
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("history_limit must be a non‑negative integer")
    return limit


def _validate_incoming_message(msg: str) -> str:
    if not isinstance(msg, str) or not msg.strip():
        raise ValueError("incoming_message must be a non‑empty string")
    return msg


def _safe_requirements(raw: Optional[dict]) -> CustomerRequirements:
    """Validate stored ``temp_requirements`` JSON.

    If ``raw`` is ``None`` or fails Pydantic validation, return an empty
    ``CustomerRequirements`` instance. Errors are swallowed (could be logged
    without exposing PII).
    """
    if not isinstance(raw, dict):
        return CustomerRequirements()
    try:
        return CustomerRequirements(**raw)
    except ValidationError as exc:
        # Log a warning without exposing the raw data or PII
        logging.getLogger(__name__).warning("Invalid stored requirements; falling back to empty.")
        return CustomerRequirements()


def build_context(
    session: Session,
    external_session_id: str,
    incoming_message: str,
    *,
    history_limit: int = 20,
) -> dict:
    """Build a full agent context for a given conversation.

    Args:
        session: SQLAlchemy Session (read‑only).
        external_session_id: Public session identifier.
        incoming_message: The new user message that triggered the request.
        history_limit: Maximum number of *customer/assistant* messages to
            include from the past conversation (most recent first). ``0``
            returns an empty list.

    Returns:
        dict with keys: ``session_id``, ``conversation_id``, ``channel``,
        ``requirements`` (validated ``CustomerRequirements``), ``history``
        (list of safe message dicts), ``incoming_message`` (str), and
        ``tool_audit`` (list of minimal tool‑audit dicts).
    """
    # Validate inputs early
    history_limit = _validate_history_limit(history_limit)
    incoming_message = _validate_incoming_message(incoming_message)

    # Resolve the conversation (raises ValueError if missing)
    conv = resolve_session(session, external_session_id)

    # Load all messages (ordered by turn_id, message_id)
    msgs: List[Message] = get_conversation_messages(session, conv.conversation_id)

    # Separate tool messages and normal messages
    tool_audit = []
    normal = []
    for m in msgs:
        if m.role == "tool":
            tool_audit.append(
                {
                    "turn_id": m.turn_id,
                    "role": m.role,
                    "content": m.content,
                    "tool_name": m.tool_name,
                    "tool_call_id": m.tool_call_id,
                    "tool_arguments": m.tool_arguments,
                    "tool_status": m.tool_status,
                    "tool_result": m.tool_result,
                    "tool_error": m.tool_error,
                    "created_at": m.created_at,
                }
            )
        else:
            normal.append(
                {
                    "turn_id": m.turn_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                }
            )

    # Apply history limit – keep most recent N normal messages (chronological order)
    if history_limit == 0:
        history = []
    else:
        # normal is already chronological; slice the tail
        history = normal[-history_limit:]

    # Validate stored temp_requirements
    requirements = _safe_requirements(conv.temp_requirements)

    return {
        "session_id": conv.external_session_id,
        "conversation_id": conv.conversation_id,
        "channel": conv.channel,
        "requirements": requirements,
        "history": history,
        "incoming_message": incoming_message,
        "tool_audit": tool_audit,
    }

"""Note: The ``NextAction`` schema is defined elsewhere and will be used by the
central agent after this context is processed.  It is not part of the context
builder output, but the builder keeps tool audit records separate for later
use."""

