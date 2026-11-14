'''Pydantic request schemas for agent tools.

This module defines the input models for the four backend tools required by
Phase 5C Increment 3B.  All models forbid extra fields, use Pydantic v2
validation, and perform the same budget‑range and whitespace checks as the
existing ``CustomerRequirements`` model.
''' 

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------

def _ensure_decimal(value: Optional[object]) -> Optional[Decimal]:
    """Convert ``None`` or a numeric/str value to ``Decimal``.

    ``pydantic`` will call this in ``mode='before'`` so that string inputs are
    accepted.
    """
    if value is None:
        return None
    return Decimal(str(value))

def _reject_whitespace(value: Optional[str]) -> Optional[str]:
    """Return the string if it contains non‑whitespace characters, otherwise raise.
    """
    if value is None:
        return None
    if not value.strip():
        raise ValueError("value cannot be whitespace only")
    return value

# ---------------------------------------------------------------------------
# 1. SearchPropertiesRequest
# ---------------------------------------------------------------------------
class SearchPropertiesRequest(BaseModel):
    """Parameters accepted by the ``search_properties`` tool.

    The field names are intentionally aligned with the public API used by the
    ``PropertySearchService`` – they are a subset of the full service signature
    and therefore keep the contract small and stable.
    """

    model_config = ConfigDict(extra="forbid")

    location: Optional[str] = Field(default=None)
    property_type: Optional[str] = Field(default=None)
    configuration: Optional[str] = Field(default=None)
    # ``max_budget`` mirrors the ``max_price`` argument of the service.
    max_budget: Optional[Decimal] = Field(default=None, description="Upper price bound")
    possession_before: Optional[date] = Field(default=None, description="Desired possession date")

    # Convert strings to Decimal before validation.
    _decimal = field_validator("max_budget", mode="before")(_ensure_decimal)
    # Reject whitespace‑only strings for the textual criteria.
    _no_ws_location = field_validator("location", mode="before")(_reject_whitespace)
    _no_ws_property_type = field_validator("property_type", mode="before")(_reject_whitespace)
    _no_ws_configuration = field_validator("configuration", mode="before")(_reject_whitespace)

    @model_validator(mode="after")
    def _budget_non_negative(self) -> "SearchPropertiesRequest":
        if self.max_budget is not None and self.max_budget < 0:
            raise ValueError("max_budget must be non‑negative")
        return self
# ---------------------------------------------------------------------------
# 2. GetPropertyDetailsRequest
# ---------------------------------------------------------------------------
class GetPropertyDetailsRequest(BaseModel):
    """Request schema for ``get_property_details``.
    """

    model_config = ConfigDict(extra="forbid")

    property_id: int = Field(..., ge=1, description="Positive identifier of the property")

# ---------------------------------------------------------------------------
# 3. UpdateCustomerRequirementsRequest – partial update of CustomerRequirements
# ---------------------------------------------------------------------------
class UpdateCustomerRequirementsRequest(BaseModel):
    """Partial update model for a user's ``CustomerRequirements``.

    All fields are optional and share the same validation rules as the full
    ``CustomerRequirements`` schema defined in ``backend.orchestrator.schemas``.
    """

    model_config = ConfigDict(extra="forbid")

    # Re‑use the same field names; they are all optional in the base model.
    budget_min: Optional[Decimal] = Field(default=None)
    budget_max: Optional[Decimal] = Field(default=None)
    location: Optional[str] = Field(default=None)
    property_type: Optional[str] = Field(default=None)
    configuration: Optional[str] = Field(default=None)
    possession_before: Optional[date] = Field(default=None)
    availability: Optional[str] = Field(default=None)

    # Decimal conversion
    _budget_min_decimal = field_validator("budget_min", mode="before")(_ensure_decimal)
    _budget_max_decimal = field_validator("budget_max", mode="before")(_ensure_decimal)
    # Whitespace rejection for string fields
    _ws_location = field_validator("location", mode="before")(_reject_whitespace)
    _ws_property_type = field_validator("property_type", mode="before")(_reject_whitespace)
    _ws_configuration = field_validator("configuration", mode="before")(_reject_whitespace)
    _ws_availability = field_validator("availability", mode="before")(_reject_whitespace)

    @model_validator(mode="after")
    def _validate_budget(self) -> "UpdateCustomerRequirementsRequest":
        # Non‑negative checks
        if self.budget_min is not None and self.budget_min < 0:
            raise ValueError("budget_min must be non‑negative")
        if self.budget_max is not None and self.budget_max < 0:
            raise ValueError("budget_max must be non‑negative")
        # Ordering check when both supplied
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("budget_min cannot exceed budget_max")
        return self
        

# ---------------------------------------------------------------------------
# 4. CreateCallbackRequest
# ---------------------------------------------------------------------------
class CreateCallbackRequest(BaseModel):
    """Input for the ``create_callback_request`` tool.
    """

    model_config = ConfigDict(extra="forbid")

    preferred_date: Optional[date] = Field(default=None)
    preferred_time: Optional[str] = Field(default=None)
    reason: str = Field(..., description="User‑provided reason for the callback")

    # Whitespace‑only rejection for ``reason``
    _reason_ws = field_validator("reason", mode="before")(_reject_whitespace)
    _time_ws = field_validator("preferred_time", mode="before")(_reject_whitespace)

__all__ = [
    "SearchPropertiesRequest",
    "GetPropertyDetailsRequest",
    "UpdateCustomerRequirementsRequest",
    "CreateCallbackRequest",
]
