"""Pydantic schemas for the RealtyVoice AI orchestrator."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ..schemas.property_search import PropertySearchItem

class ConversationStartRequest(BaseModel):
    channel: Literal["browser_voice", "browser_text", "telephone"] = Field(..., description="Channel through which the user is interacting.")

class ConversationStartResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: str = Field(..., description="Opaque session identifier.")
    channel: Literal["browser_voice", "browser_text", "telephone"]
    created_at: datetime

class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("session_id", "message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot contain only whitespace.")
        return value

class CustomerRequirements(BaseModel):
    budget_min: Optional[Decimal] = None
    budget_max: Optional[Decimal] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    configuration: Optional[str] = None
    possession_before: Optional[date] = None
    availability: Optional[str] = None

    @model_validator(mode="after")
    def validate_budget(self) -> CustomerRequirements:
        if self.budget_min is not None and self.budget_min < 0:
            raise ValueError("budget_min must be non-negative.")
        if self.budget_max is not None and self.budget_max < 0:
            raise ValueError("budget_max must be non-negative.")
        if self.budget_min is not None and self.budget_max is not None and self.budget_min > self.budget_max:
            raise ValueError("budget_min cannot exceed budget_max.")
        return self

class NextAction(BaseModel):
    action: Literal["ask_clarification", "show_properties", "offer_callback", "await_user", "error"]
    message: Optional[str] = None
    property_ids: List[int] = Field(default_factory=list)

class AgentResponse(BaseModel):
    session_id: str
    turn_id: int = Field(..., ge=0)
    reply: str
    spoken_reply: str = Field("", description="Speech-friendly version of the reply, stripped of markdown and tables.")
    intent: Literal["property_search", "property_details", "callback_request", "general_question", "unknown"]
    requirements: CustomerRequirements = Field(default_factory=CustomerRequirements)
    matched_properties: List[PropertySearchItem] = Field(default_factory=list)
    next_action: NextAction
