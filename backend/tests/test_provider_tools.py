
from backend.orchestrator.services.provider_tools import (
    get_provider_tool_definitions,
)
from backend.orchestrator.services.tool_registry import TOOL_REGISTRY
from backend.orchestrator.providers.base import ToolDefinition


def test_provider_definitions_cover_all_registered_tools():
    definitions = get_provider_tool_definitions()

    assert len(definitions) == len(TOOL_REGISTRY)

    assert {item.name for item in definitions} == set(TOOL_REGISTRY)


def test_provider_definitions_have_valid_structure():
    definitions = get_provider_tool_definitions()

    for definition in definitions:
        assert isinstance(definition, ToolDefinition)
        assert definition.description
        assert isinstance(definition.parameters, dict)
        assert definition.parameters.get("type") == "object"


def test_callback_tool_schema_forbids_extra_fields():
    definitions = get_provider_tool_definitions()

    callback = next(
        item
        for item in definitions
        if item.name == "create_callback_request"
    )

    assert callback.parameters["additionalProperties"] is False
