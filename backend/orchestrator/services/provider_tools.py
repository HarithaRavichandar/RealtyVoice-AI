
"""Convert registered backend tools into provider-independent definitions."""

from backend.orchestrator.providers.base import ToolDefinition
from .tool_registry import TOOL_REGISTRY


TOOL_DESCRIPTIONS = {
    "search_properties": (
        "Search available property listings using the customer's "
        "location, property type, configuration, budget range "
        "(min_budget and/or max_budget), possession date, and "
        "availability ('Ready to Move' or 'Under Construction'). "
        "Use this when the customer wants property options."
    ),
    "search_live_properties": (
        "Search LIVE 99acres listings (fetched on request through the "
        "API.SH data service). Use this ONLY when the customer explicitly "
        "asks for live, current or real 99acres listings or real "
        "inventory; for every ordinary property search use "
        "search_properties instead. A location is required. Each call "
        "uses one credit from a small per-conversation allowance, so never "
        "call it twice for the same request and never call it "
        "speculatively. Results are separate from the demo dataset and "
        "must always be labelled with their source."
    ),
    "get_property_details": (
        "Retrieve the full details of a specific property using its "
        "property ID. Use this when the customer asks about a listed property "
        "from the demo dataset. It does not work for live listings."
    ),
    "update_customer_requirements": (
        "Save or update the customer's explicitly stated property "
        "requirements, such as budget, location, property type, "
        "configuration, possession date, or availability, and the "
        "customer's qualification details: purchase_timeline "
        "(immediate, within_3_months, within_6_months, within_12_months, "
        "later, exploring), visit_readiness (ready_to_visit, considering, "
        "not_ready) and contact_preference (phone_call, whatsapp, sms, "
        "email). Save these only when the customer has said them. To remove a "
        "requirement the customer has withdrawn, pass that field as null. "
        "The budget minimum cannot exceed the maximum."
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
    "recommend_properties": (
        "Rank properties against ALL of the customer's saved requirements "
        "(location, budget range, configuration, property type, possession, "
        "availability) and return the best matches with a 0-100 match score "
        "and a per-criterion explanation. Use this when the customer asks "
        "what you recommend or which properties suit them best; use "
        "search_properties for a plain filtered listing."
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
