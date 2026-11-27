"""Pydantic request schemas for agent tools.

This module defines the input models for the four backend tools required by
Phase 5C Increment 3B. All models forbid extra fields, use Pydantic v2
validation, and perform the same budget-range and whitespace checks as the
existing CustomerRequirements model.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from re import fullmatch
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------

def _ensure_decimal(value: Optional[object]) -> Optional[Decimal]:
    """Convert None or a numeric/string value to Decimal."""
    if value is None:
        return None
    return Decimal(str(value))


def _reject_whitespace(value: Optional[str]) -> Optional[str]:
    """Reject strings containing only whitespace."""
    if value is None:
        return None
    if not value.strip():
        raise ValueError("value cannot be whitespace only")
    return value


# ---------------------------------------------------------------------------
# 1. SearchPropertiesRequest
# ---------------------------------------------------------------------------

class SearchPropertiesRequest(BaseModel):
    """Parameters accepted by the search_properties tool.

    The field names are intentionally aligned with the public API used by the
    PropertySearchService. They are a subset of the full service signature
    and therefore keep the contract small and stable.
    """

    model_config = ConfigDict(extra="forbid")

    location: Optional[str] = Field(default=None)
    property_type: Optional[str] = Field(default=None)
    configuration: Optional[str] = Field(default=None)

    # min_budget / max_budget mirror the min_price / max_price arguments
    # of the PropertySearchService.
    min_budget: Optional[Decimal] = Field(
        default=None,
        description="Lower price bound",
    )
    max_budget: Optional[Decimal] = Field(
        default=None,
        description="Upper price bound",
    )
    possession_before: Optional[date] = Field(
        default=None,
        description="Desired possession date",
    )

    # Convert strings to Decimal before validation.
    _decimal = field_validator(
        "min_budget",
        "max_budget",
        mode="before",
    )(_ensure_decimal)

    # Reject whitespace-only strings for textual criteria.
    _no_ws_location = field_validator(
        "location",
        mode="before",
    )(_reject_whitespace)

    _no_ws_property_type = field_validator(
        "property_type",
        mode="before",
    )(_reject_whitespace)

    _no_ws_configuration = field_validator(
        "configuration",
        mode="before",
    )(_reject_whitespace)

    @model_validator(mode="after")
    def _budget_non_negative(self) -> "SearchPropertiesRequest":
        if self.min_budget is not None and self.min_budget < 0:
            raise ValueError("min_budget must be non‑negative")
        if self.max_budget is not None and self.max_budget < 0:
            raise ValueError("max_budget must be non‑negative")
        if (
            self.min_budget is not None
            and self.max_budget is not None
            and self.min_budget > self.max_budget
        ):
            raise ValueError("min_budget must not exceed max_budget")
        return self


# ---------------------------------------------------------------------------
# 2. GetPropertyDetailsRequest
# ---------------------------------------------------------------------------

class GetPropertyDetailsRequest(BaseModel):
    """Request schema for get_property_details."""

    model_config = ConfigDict(extra="forbid")

    property_id: int = Field(
        ...,
        ge=1,
        description="Positive identifier of the property",
    )


# ---------------------------------------------------------------------------
# 3. UpdateCustomerRequirementsRequest
# ---------------------------------------------------------------------------

class UpdateCustomerRequirementsRequest(BaseModel):
    """Partial update model for a user's CustomerRequirements.

    All fields are optional and share the same validation rules as the full
    CustomerRequirements schema defined in backend.orchestrator.schemas.
    """

    model_config = ConfigDict(extra="forbid")

    # Re-use the same field names; they are all optional in the base model.
    budget_min: Optional[Decimal] = Field(default=None)
    budget_max: Optional[Decimal] = Field(default=None)
    location: Optional[str] = Field(default=None)
    property_type: Optional[str] = Field(default=None)
    configuration: Optional[str] = Field(default=None)
    possession_before: Optional[date] = Field(default=None)
    availability: Optional[str] = Field(default=None)

    # Decimal conversion.
    _budget_min_decimal = field_validator(
        "budget_min",
        mode="before",
    )(_ensure_decimal)

    _budget_max_decimal = field_validator(
        "budget_max",
        mode="before",
    )(_ensure_decimal)

    # Whitespace rejection for string fields.
    _ws_location = field_validator(
        "location",
        mode="before",
    )(_reject_whitespace)

    _ws_property_type = field_validator(
        "property_type",
        mode="before",
    )(_reject_whitespace)

    _ws_configuration = field_validator(
        "configuration",
        mode="before",
    )(_reject_whitespace)

    _ws_availability = field_validator(
        "availability",
        mode="before",
    )(_reject_whitespace)

    @model_validator(mode="after")
    def _validate_budget(self) -> "UpdateCustomerRequirementsRequest":
        # Non-negative checks.
        if self.budget_min is not None and self.budget_min < 0:
            raise ValueError("budget_min must be non‑negative")

        if self.budget_max is not None and self.budget_max < 0:
            raise ValueError("budget_max must be non‑negative")

        # Ordering check when both are supplied.
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
    """Input for the create_callback_request tool."""

    model_config = ConfigDict(extra="forbid")

    preferred_contact: str = Field(
        ...,
        description="Preferred contact method",
    )

    preferred_date: date = Field(
        ...,
        description="Preferred date for callback",
    )

    preferred_time: str = Field(
        ...,
        description="Preferred time for callback (HH:MM, 24-hour format)",
    )

    reason: str = Field(
        ...,
        description="User-provided reason for the callback",
    )

    # Reject whitespace-only values.
    _contact_ws = field_validator(
        "preferred_contact",
        mode="before",
    )(_reject_whitespace)

    _reason_ws = field_validator(
        "reason",
        mode="before",
    )(_reject_whitespace)

    _time_ws = field_validator(
        "preferred_time",
        mode="before",
    )(_reject_whitespace)

    # Validate callback time in strict HH:MM 24-hour format.
    @field_validator("preferred_time")
    @classmethod
    def validate_preferred_time(cls, value: str) -> str:
        if not fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError(
                "preferred_time must use HH:MM format"
            )

        return value


# ---------------------------------------------------------------------------
# 5. RegisterLeadRequest
# ---------------------------------------------------------------------------

class RegisterLeadRequest(BaseModel):
    """Input for the register_lead tool.

    Collects customer contact information so a Lead record can be created
    and linked to the current conversation.  At least one of ``customer_name``,
    ``phone``, or ``email`` must be provided.
    """

    model_config = ConfigDict(extra="forbid")

    customer_name: Optional[str] = Field(
        default=None,
        description="Customer's full name",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Customer's phone number",
    )
    email: Optional[str] = Field(
        default=None,
        description="Customer's email address",
    )

    # Reject whitespace-only values.
    _name_ws = field_validator(
        "customer_name",
        mode="before",
    )(_reject_whitespace)

    _phone_ws = field_validator(
        "phone",
        mode="before",
    )(_reject_whitespace)

    _email_ws = field_validator(
        "email",
        mode="before",
    )(_reject_whitespace)

    @model_validator(mode="after")
    def _at_least_one_contact(self) -> "RegisterLeadRequest":
        if not any([self.customer_name, self.phone, self.email]):
            raise ValueError(
                "At least one of customer_name, phone, or email is required"
            )
        return self


# ---------------------------------------------------------------------------
# 6. RecordConsentRequest
# ---------------------------------------------------------------------------

class RecordConsentRequest(BaseModel):
    """Input for the record_consent tool.

    Records the customer's explicit consent (or refusal) for a specific
    consent type.  The agent must ask for consent clearly before calling
    this tool.
    """

    model_config = ConfigDict(extra="forbid")

    consent_type: str = Field(
        ...,
        description=(
            "Type of consent: 'callback_contact' for callback requests, "
            "'contact_storage' for storing contact info, "
            "'marketing_communications' for marketing opt-in"
        ),
    )
    granted: bool = Field(
        ...,
        description="True if the customer granted consent, False if they refused",
    )

    _consent_type_ws = field_validator(
        "consent_type",
        mode="before",
    )(_reject_whitespace)

    @field_validator("consent_type")
    @classmethod
    def _validate_consent_type(cls, value: str) -> str:
        allowed = {
            "contact_storage",
            "callback_contact",
            "marketing_communications",
        }
        if value not in allowed:
            raise ValueError(
                f"consent_type must be one of: {', '.join(sorted(allowed))}"
            )
        return value


# ---------------------------------------------------------------------------
# 7. NearbyAmenitiesRequest
# ---------------------------------------------------------------------------

class NearbyAmenitiesRequest(BaseModel):
    """Input for the nearby_amenities tool.

    Finds nearby points-of-interest around a property location using
    the Mappls API.  The agent should supply the location name (which
    will be geocoded) and optionally a list of amenity categories.
    """

    model_config = ConfigDict(extra="forbid")

    location: str = Field(
        ...,
        min_length=1,
        description=(
            "Location or address to search around, "
            "e.g. 'Whitefield, Bengaluru'"
        ),
    )

    categories: Optional[list[str]] = Field(
        default=None,
        description=(
            "Amenity categories to search for. "
            "Options: hospital, school, metro, park, restaurant, "
            "supermarket, atm, pharmacy, gym, mall, bus_stop, bank. "
            "Defaults to a standard real-estate set if omitted."
        ),
    )

    radius: Optional[int] = Field(
        default=3000,
        ge=100,
        le=50000,
        description="Search radius in metres (100–50000, default 3000)",
    )

    # Reject whitespace-only location.
    _ws_location = field_validator(
        "location",
        mode="before",
    )(_reject_whitespace)


__all__ = [
    "SearchPropertiesRequest",
    "GetPropertyDetailsRequest",
    "UpdateCustomerRequirementsRequest",
    "CreateCallbackRequest",
    "RegisterLeadRequest",
    "RecordConsentRequest",
    "NearbyAmenitiesRequest",
]