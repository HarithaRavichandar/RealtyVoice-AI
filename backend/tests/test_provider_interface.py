
import pytest

from backend.orchestrator.providers import (
    BaseLLMProvider,
    Message,
    MessageRole,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)


def test_provider_response_with_tool_call():
    call = ToolCall(
        id="call_123",
        function_name="search_properties",
        arguments={"location": "Bengaluru"},
    )

    response = ProviderResponse(
        content=None,
        tool_calls=[call],
        finish_reason="tool_calls",
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function_name == "search_properties"
    assert response.finish_reason == "tool_calls"


def test_provider_message_structure():
    message = Message(
        role=MessageRole.USER,
        content="Find a 2 BHK apartment",
    )

    assert message.role == MessageRole.USER
    assert message.content == "Find a 2 BHK apartment"


def test_tool_definition_structure():
    tool = ToolDefinition(
        name="search_properties",
        description="Search available properties",
        parameters={"type": "object", "properties": {}},
    )

    assert tool.name == "search_properties"
    assert tool.parameters["type"] == "object"


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        BaseLLMProvider()
