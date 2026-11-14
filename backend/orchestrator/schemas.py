"""Pydantic schemas for the orchestrator API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class ConversationStartRequest(BaseModel):
    channel: Literal["browser_voice", "browser_text", "telephone"] = Field(..., description="Channel through which the user is interacting.")

class ConversationStartResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str = Field(..., description="Opaque session identifier.")
    channel: Literal["browser_voice", "browser_text", "telephone"]
    created_at: datetime
