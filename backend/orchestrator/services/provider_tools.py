
"""Convert registered backend tools into provider-independent definitions."""

from backend.orchestrator.providers.base import ToolDefinition
from .tool_registry import TOOL_REGISTRY


TOOL_DESCRIPTIONS = {
    "search_properties": (
        "Search available property listings using the customer's "
        "location, property type, configuration, budget, and possession "
        "requirements. Use this when the customer wants property options."
    ),
    "get_property_details": (
        "Retrieve the full details of a specific property using its "
        "property ID. Use this when the customer asks about a listed property."
    ),
    "update_customer_requirements": (
        "Save or update the customer's explicitly stated property "
        "requirements, such as budget, location, property type, "
        "configuration, possession date, or availability."
    ),
    "create_callback_request": (
        "Create a callback request using the customer's preferred "
        "contact method, date, time, and reason. Only use this when "
        "the customer has explicitly requested a callback. Backend "
        "consent checks must still pass."
    ),
}


def get_provider_tool_definitions() -> list[ToolDefinition]:
    """Build provider tool definitions from the static tool registry."""

    definitions = []

    for name, tool_info in TOOL_REGISTRY.items():
        definitions.append(
            ToolDefinition(
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                parameters=tool_info.input_schema.model_json_schema(),
            )
        )

    return definitions


__all__ = [
    "TOOL_DESCRIPTIONS",
    "get_provider_tool_definitions",
]
