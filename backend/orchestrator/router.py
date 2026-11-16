
"""Conversation, message, and agent API routes for RealtyVoice AI."""

from datetime import datetime
from typing import List, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas.property_search import PropertySearchItem
from .schemas import (
    AgentRequest,
    AgentResponse,
    ConversationStartRequest,
    ConversationStartResponse,
    NextAction,
)
from .services.conversation import (
    create_conversation,
    get_conversation_by_session_id,
)
from .services.message import (
    add_message,
    get_conversation_messages as load_conversation_messages,
)
from .services.agent_service import run_agent_loop

router = APIRouter()


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    turn_id: int
    role: str
    content: str
    created_at: datetime


def _collect_matched_properties(
    tool_results: list[dict[str, Any]],
) -> list[PropertySearchItem]:
    """Extract valid property records from registered tool results."""

    properties: dict[int, PropertySearchItem] = {}

    for tool_entry in tool_results:
        result = tool_entry.get("result") or {}

        candidates = []

        if isinstance(result.get("items"), list):
            candidates.extend(result["items"])

        if isinstance(result.get("property"), dict):
            candidates.append(result["property"])

        for candidate in candidates:
            try:
                item = PropertySearchItem.model_validate(candidate)
                properties[item.property_id] = item
            except (ValidationError, TypeError, ValueError):
                # Do not expose malformed tool data in the API response.
                continue

    return list(properties.values())


def _build_agent_response(
    result: Any,
    reply: str,
) -> AgentResponse:
    """Build the public response from the agent's execution results."""

    tool_results = result.tool_results
    tool_names = {
        entry.get("tool_name")
        for entry in tool_results
    }

    matched_properties = _collect_matched_properties(tool_results)
    property_ids = [
        item.property_id for item in matched_properties
    ]

    requirements = result.requirements

    if result.iteration_limit_reached:
        intent = "unknown"
        next_action = NextAction(
            action="error",
            message=reply,
        )

    elif "create_callback_request" in tool_names:
        intent = "callback_request"

        callback_succeeded = any(
            entry.get("tool_name") == "create_callback_request"
            and isinstance(entry.get("result"), dict)
            and isinstance(entry["result"].get("callback"), dict)
            for entry in tool_results
        )

        next_action = NextAction(
            action="await_user" if callback_succeeded else "ask_clarification",
            message=reply,
        )

    elif "get_property_details" in tool_names:
        intent = "property_details"

        next_action = NextAction(
            action="show_properties" if matched_properties else "ask_clarification",
            message=reply,
            property_ids=property_ids,
        )

    elif "search_properties" in tool_names:
        intent = "property_search"

        next_action = NextAction(
            action="show_properties" if matched_properties else "ask_clarification",
            message=reply,
            property_ids=property_ids,
        )

    elif any(
        entry.get("tool_name") not in {
            "update_customer_requirements",
            None,
        }
        for entry in tool_results
    ):
        intent = "unknown"
        next_action = NextAction(
            action="error",
            message=reply,
        )

    else:
        intent = "general_question"

        next_action = NextAction(
            action="ask_clarification" if "?" in reply else "await_user",
            message=reply,
        )

    return AgentResponse(
        session_id=result.session_id,
        turn_id=result.turn_id,
        reply=reply,
        intent=intent,
        requirements=requirements,
        matched_properties=matched_properties,
        next_action=next_action,
    )


@router.post(
    "/conversations",
    response_model=ConversationStartResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_conversation(
    request: ConversationStartRequest,
    db: Session = Depends(get_db),
):
    try:
        conv = create_conversation(db, request.channel)
        db.commit()
        db.refresh(conv)

        return ConversationStartResponse(
            session_id=conv.external_session_id,
            channel=conv.channel,
            created_at=conv.created_at,
        )

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to create conversation",
        ) from exc


@router.get(
    "/conversations/{session_id}",
    response_model=ConversationStartResponse,
)
def get_conversation(
    session_id: str,
    db: Session = Depends(get_db),
):
    conv = get_conversation_by_session_id(db, session_id)

    if conv is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return ConversationStartResponse(
        session_id=conv.external_session_id,
        channel=conv.channel,
        created_at=conv.created_at,
    )


@router.get(
    "/conversations/{session_id}/messages",
    response_model=List[MessageResponse],
)
def list_conversation_messages(
    session_id: str,
    db: Session = Depends(get_db),
):
    conv = get_conversation_by_session_id(db, session_id)

    if conv is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = load_conversation_messages(db, conv.conversation_id)

    return [
        MessageResponse(
            turn_id=m.turn_id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.post(
    "/agent/turn",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
)
async def agent_turn(
    request: AgentRequest,
    db: Session = Depends(get_db),
):
    """Process one customer message and persist the final assistant reply."""

    # Verify the session before invoking the agent.
    conversation = get_conversation_by_session_id(
        db,
        request.session_id,
    )

    if conversation is None:
        db.rollback()
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    try:
        result = await run_agent_loop(
            session=db,
            session_id=request.session_id,
            incoming_message=request.message,
        )

        # Persist the final assistant response in the same transaction
        # as the customer message and tool audit records.
        add_message(
            session=db,
            conversation_id=result.conversation_id,
            turn_id=result.turn_id,
            content=result.reply,
            role="assistant",
        )

        response = _build_agent_response(
            result=result,
            reply=result.reply,
        )

        db.commit()

        return response

    except ValueError as exc:
        db.rollback()

        if "conversation not found" in str(exc).lower():
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            ) from exc

        raise HTTPException(
            status_code=400,
            detail="Unable to process the supplied request",
        ) from exc

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=502,
            detail="The agent service is temporarily unavailable",
        ) from exc


__all__ = [
    "router",
    "MessageResponse",
    "start_conversation",
    "get_conversation",
    "list_conversation_messages",
    "agent_turn",
]
