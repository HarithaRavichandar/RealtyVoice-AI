
"""Tests for the core agent orchestration loop."""

import pytest

from backend.orchestrator.providers.base import (
    MessageRole,
    ProviderResponse,
    ToolCall,
)
from backend.orchestrator.services.agent_service import run_agent_loop
from backend.orchestrator.services.conversation import create_conversation
from backend.orchestrator.services.message import get_conversation_messages


class FakeProvider:
    """Deterministic provider for orchestration tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        if not self.responses:
            raise AssertionError("No fake provider response remains")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response

    async def aclose(self):
        self.closed = True


def make_tool_call(name, arguments, call_id="call-1"):
    return ToolCall(
        id=call_id,
        function_name=name,
        arguments=arguments,
    )


@pytest.mark.asyncio
async def test_plain_text_response_persists_customer_message(dbsession):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                content="Hello! How can I help you find a property?",
            )
        ]
    )

    result = await run_agent_loop(
        dbsession,
        conversation.external_session_id,
        "Hello",
        provider=provider,
    )

    assert result.reply == "Hello! How can I help you find a property?"
    assert result.turn_id == 1
    assert result.tool_iterations == 0
    assert result.tool_results == []

    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    assert len(messages) == 1
    assert messages[0].role == "customer"
    assert messages[0].content == "Hello"

    assert provider.closed is False


@pytest.mark.asyncio
async def test_update_requirements_tool_then_final_response(dbsession):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                content=None,
                tool_calls=[
                    make_tool_call(
                        "update_customer_requirements",
                        {"location": "Bengaluru"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                content="I'll look for properties in Bengaluru.",
            ),
        ]
    )

    result = await run_agent_loop(
        dbsession,
        conversation.external_session_id,
        "I am looking in Bengaluru",
        provider=provider,
    )

    assert result.reply == "I'll look for properties in Bengaluru."
    assert result.tool_iterations == 1
    assert len(result.tool_results) == 1

    tool_result = result.tool_results[0]["result"]

    assert tool_result["tool"] == "update_customer_requirements"
    assert tool_result["requirements"]["location"] == "Bengaluru"

    # Verify that AgentRunResult contains the latest persisted requirements.
    assert result.requirements.location == "Bengaluru"

    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    assert len(messages) == 2
    assert messages[0].role == "customer"
    assert messages[1].role == "tool"
    assert messages[1].tool_name == "update_customer_requirements"
    assert messages[1].tool_call_id == "call-1"
    assert messages[1].tool_status == "success"

    assert len(provider.calls) == 2

    tool_messages = [
        message
        for message in provider.calls[1]["messages"]
        if message.role == MessageRole.TOOL
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_sequentially(dbsession):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                content=None,
                tool_calls=[
                    make_tool_call(
                        "update_customer_requirements",
                        {"location": "Chennai"},
                        "call-update",
                    ),
                    make_tool_call(
                        "get_property_details",
                        {"property_id": 9999},
                        "call-details",
                    ),
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(content="I couldn't find that property."),
        ]
    )

    result = await run_agent_loop(
        dbsession,
        conversation.external_session_id,
        "Search in Chennai and check property 9999",
        provider=provider,
    )

    assert result.tool_iterations == 1
    assert len(result.tool_results) == 2

    assert [
        item["tool_name"] for item in result.tool_results
    ] == [
        "update_customer_requirements",
        "get_property_details",
    ]

    assert result.tool_results[0]["result"]["requirements"]["location"] == "Chennai"
    assert result.tool_results[1]["result"]["error"] == "not_found"

    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    tool_messages = [message for message in messages if message.role == "tool"]

    assert len(tool_messages) == 2
    assert tool_messages[0].tool_call_id == "call-update"
    assert tool_messages[1].tool_call_id == "call-details"

    assert len(provider.calls) == 2

    returned_tool_messages = [
        message
        for message in provider.calls[1]["messages"]
        if message.role == MessageRole.TOOL
    ]

    assert len(returned_tool_messages) == 2


@pytest.mark.asyncio
async def test_unregistered_tool_is_rejected_and_audited(dbsession):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                content=None,
                tool_calls=[
                    make_tool_call(
                        "delete_database",
                        {},
                        "call-invalid",
                    )
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(content="I can help with property enquiries."),
        ]
    )

    result = await run_agent_loop(
        dbsession,
        conversation.external_session_id,
        "Hello",
        provider=provider,
    )

    assert len(result.tool_results) == 1
    assert result.tool_results[0]["result"]["error"] == "unknown_tool"

    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    audit = [message for message in messages if message.role == "tool"]

    assert len(audit) == 1
    assert audit[0].tool_name == "delete_database"
    assert audit[0].tool_status == "error"
    assert audit[0].tool_error == "unknown_tool"


@pytest.mark.asyncio
async def test_tool_iteration_limit_stops_additional_execution(
    dbsession,
    monkeypatch,
):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    monkeypatch.setattr(
        "backend.orchestrator.services.agent_service.AGENT_MAX_TOOL_ITERATIONS",
        1,
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                content=None,
                tool_calls=[
                    make_tool_call(
                        "update_customer_requirements",
                        {"location": "Bengaluru"},
                        "call-first",
                    )
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                content=None,
                tool_calls=[
                    make_tool_call(
                        "update_customer_requirements",
                        {"property_type": "villa"},
                        "call-second",
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    result = await run_agent_loop(
        dbsession,
        conversation.external_session_id,
        "Find a property",
        provider=provider,
    )

    assert result.iteration_limit_reached is True
    assert result.tool_iterations == 1
    assert len(result.tool_results) == 1
    assert len(provider.calls) == 2

    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    tool_messages = [message for message in messages if message.role == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-first"


@pytest.mark.asyncio
async def test_provider_failure_propagates_without_commit_or_rollback(dbsession):
    conversation = create_conversation(
        dbsession,
        channel="browser_text",
    )

    dbsession.commit()

    provider = FakeProvider(
        [RuntimeError("Simulated provider failure")]
    )

    with pytest.raises(RuntimeError, match="Simulated provider failure"):
        await run_agent_loop(
            dbsession,
            conversation.external_session_id,
            "Hello",
            provider=provider,
        )

    # The service has flushed the customer message but has not committed
    # or rolled back. The API layer remains responsible for rollback.
    messages = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    assert len(messages) == 1
    assert messages[0].role == "customer"

    assert provider.closed is False

    dbsession.rollback()

    messages_after_rollback = get_conversation_messages(
        dbsession,
        conversation.conversation_id,
    )

    assert messages_after_rollback == []
