
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
        "the customer has explicitly requested a callback. A lead "
        "must be registered and callback_contact consent must be "
        "recorded before calling this tool."
    ),
    "register_lead": (
        "Register the customer as a lead by saving their contact "
        "information (name, phone, or email). Call this when the "
        "customer provides their contact details, typically during "
        "a callback request flow. At least one contact field is required. "
        "If a lead is already registered for this conversation, "
        "this tool updates the existing lead."
    ),
    "record_consent": (
        "Record the customer's explicit consent or refusal for a "
        "specific purpose. The agent must clearly ask the customer "
        "for permission before calling this tool. Required consent "
        "types: 'callback_contact' before creating a callback request, "
        "'contact_storage' before storing contact information."
    ),
    "nearby_amenities": (
        "Find nearby amenities and points of interest around a "
        "location. Use this when the customer asks about hospitals, "
        "schools, metro stations, parks, restaurants, or other "
        "facilities near a property or area. Provide the location "
        "name (e.g. 'Whitefield, Bengaluru') and optionally specify "
        "which categories to search for."
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
