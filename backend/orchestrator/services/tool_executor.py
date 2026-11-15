# -*- coding: utf-8 -*-
"""
Tool Executor – Phase 5C Increment 3C (partial implementation)

Implements execution for the two core tools:
* search_properties
* get_property_details

The executor validates incoming payloads against the request schemas, calls the
`PropertySearchService` using the caller‑supplied SQLAlchemy session, and
returns JSON‑serialisable dictionaries. No session commit/rollback/close is
performed – the caller remains responsible for session lifecycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
from ..tools.schemas import SearchPropertiesRequest, GetPropertyDetailsRequest

# ---------------------------------------------------------------------------
# Service and response schemas
# ---------------------------------------------------------------------------
from pydantic import ValidationError
from backend.services.property_search import PropertySearchService
from backend.schemas.property_search import (
    PropertySearchItem,
    PropertyDetailResponse,
)


def execute_tool(
    session: Session,
    tool_name: str,
    payload: Dict[str, Any],
    conversation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a tool with the supplied SQLAlchemy session.

    Parameters
    ----------
    session:
        Caller‑provided SQLAlchemy Session (must remain open after the call).
    tool_name:
        Name of the tool to execute – currently ``search_properties`` or
        ``get_property_details``.
    payload:
        Raw input dictionary supplied by the caller.
    conversation_id:
        Accepted for signature compatibility but not used for the two tools.

    Returns
    -------
    dict
        JSON‑serialisable result describing success, or a structured error
        payload (validation errors, unknown tool, or not‑found).
    """

    # ---------------------------------------------------------------
    # Resolve the appropriate request schema
    # ---------------------------------------------------------------
    if tool_name == "search_properties":
        Schema = SearchPropertiesRequest
    elif tool_name == "get_property_details":
        Schema = GetPropertyDetailsRequest
    else:
        return {"error": "unknown_tool", "tool": tool_name}

    # ---------------------------------------------------------------
    # Validate the payload (Pydantic v2)
    # ---------------------------------------------------------------
    try:
        request = Schema.model_validate(payload)
    except ValidationError as exc:
        return {
            "error": "validation",
            "tool": tool_name,
            "details": exc.errors(include_context=False),
        }

    # ---------------------------------------------------------------
    # Service – keep the caller's session alive (close_session=False)
    # ---------------------------------------------------------------
    service = PropertySearchService(
        session_factory=lambda: session, close_session=False
    )

    # ---------------------------------------------------------------
    # Dispatch based on tool_name
    # ---------------------------------------------------------------
    if tool_name == "search_properties":
        total, results = service.search(
            location=request.location,
            property_type=request.property_type,
            configuration=request.configuration,
            min_price=None,  # not part of the request schema
            max_price=request.max_budget,  # map max_budget → max_price
            possession_before=request.possession_before,
            availability=None,
        )
        items = [PropertySearchItem.model_validate(p) for p in results]
        return {
            "tool": tool_name,
            "total": total,
            "items": [i.model_dump(mode="json") for i in items],
        }

    # ---------------------------------------------------------------
    # get_property_details
    # ---------------------------------------------------------------
    prop = service.get_by_id(request.property_id)
    if prop is None:
        return {
            "error": "not_found",
            "tool": tool_name,
            "property_id": request.property_id,
        }
    detail = PropertyDetailResponse.model_validate(prop)
    return {"tool": tool_name, "property": detail.model_dump(mode="json")}
