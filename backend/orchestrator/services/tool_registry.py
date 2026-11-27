'''Static tool registry for the backend orchestrator.

The registry maps the four supported tool names to their input Pydantic
schemas.  Implementations are deliberately set to ``None`` – they will be
filled in when the actual business logic is added in a later increment.
''' 

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict

# Static imports of the request schemas – no dynamic loading.
from ..tools.schemas import (
    SearchPropertiesRequest,
    GetPropertyDetailsRequest,
    UpdateCustomerRequirementsRequest,
    CreateCallbackRequest,
    RegisterLeadRequest,
    RecordConsentRequest,
    NearbyAmenitiesRequest,
    RecommendPropertiesRequest,
)

# ---------------------------------------------------------------------------
# Data structure representing a registered tool.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Immutable description of a tool.

    Attributes:
        name: The tool name as used by the orchestrator.
        input_schema: The Pydantic model class defining the request payload.
        implementation: Callable that executes the tool – ``None`` for now.
    """

    name: str
    input_schema: type
    implementation: Optional[Callable] = None

# ---------------------------------------------------------------------------
# Registry definition – exactly four entries as required.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, ToolInfo] = {
    "search_properties": ToolInfo(
        name="search_properties",
        input_schema=SearchPropertiesRequest,
        implementation=None,
    ),
    "get_property_details": ToolInfo(
        name="get_property_details",
        input_schema=GetPropertyDetailsRequest,
        implementation=None,
    ),
    "update_customer_requirements": ToolInfo(
        name="update_customer_requirements",
        input_schema=UpdateCustomerRequirementsRequest,
        implementation=None,
    ),
    "create_callback_request": ToolInfo(
        name="create_callback_request",
        input_schema=CreateCallbackRequest,
        implementation=None,
    ),
    "register_lead": ToolInfo(
        name="register_lead",
        input_schema=RegisterLeadRequest,
        implementation=None,
    ),
    "record_consent": ToolInfo(
        name="record_consent",
        input_schema=RecordConsentRequest,
        implementation=None,
    ),
    "nearby_amenities": ToolInfo(
        name="nearby_amenities",
        input_schema=NearbyAmenitiesRequest,
        implementation=None,
    ),
    "recommend_properties": ToolInfo(
        name="recommend_properties",
        input_schema=RecommendPropertiesRequest,
        implementation=None,
    ),
}

# ---------------------------------------------------------------------------
# Helper lookup function.
# ---------------------------------------------------------------------------
def lookup_tool(name: str) -> ToolInfo:
    """Return the ``ToolInfo`` for *name*.

    Raises ``KeyError`` if the tool name is not present in ``TOOL_REGISTRY``.
    """
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Tool '{name}' not found in registry") from exc

__all__ = ["ToolInfo", "TOOL_REGISTRY", "lookup_tool"]
