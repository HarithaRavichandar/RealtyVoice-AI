
"""Tests for the Groq HTTP provider using mocked HTTP responses."""

import asyncio
import json

import httpx
import pytest

from backend.orchestrator.providers import (
    GroqAPIError,
    GroqAuthenticationError,
    GroqProvider,
    GroqRateLimitError,
    GroqResponseError,
    GroqTimeoutError,
    Message,
    MessageRole,
    ToolDefinition,
)


def make_provider(handler):
    """Create a provider with a mocked HTTP transport."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    provider = GroqProvider(
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
        client=client,
    )
    return provider, client


def test_successful_text_completion():
    def handler(request):
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-api-key"

        payload = json.loads(request.content)
        assert payload["model"] == "openai/gpt-oss-20b"
        assert payload["messages"][0]["role"] == "user"

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Here are some properties.",
                    },
                    "finish_reason": "stop",
                }]
            },
        )

    async def run():
        provider, client = make_provider(handler)
        async with client:
            response = await provider.chat_completion([
                Message(
                    role=MessageRole.USER,
                    content="Find properties in Bengaluru",
                )
            ])

        assert response.content == "Here are some properties."
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    asyncio.run(run())


def test_successful_tool_call():
    def handler(request):
        payload = json.loads(request.content)

        assert payload["parallel_tool_calls"] is False
        assert payload["tools"][0]["function"]["name"] == "search_properties"

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search_properties",
                                "arguments": '{"location": "Bengaluru"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        )

    async def run():
        provider, client = make_provider(handler)
        async with client:
            response = await provider.chat_completion(
                messages=[
                    Message(
                        role=MessageRole.USER,
                        content="Find properties in Bengaluru",
                    )
                ],
                tools=[
                    ToolDefinition(
                        name="search_properties",
                        description="Search available properties",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                        },
                    )
                ],
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].function_name == "search_properties"
        assert response.tool_calls[0].arguments == {
            "location": "Bengaluru"
        }
        assert response.finish_reason == "tool_calls"

    asyncio.run(run())


def test_assistant_message_serializes_previous_tool_calls():
    def handler(request):
        payload = json.loads(request.content)
        assistant_message = payload["messages"][1]

        assert assistant_message["role"] == "assistant"
        assert assistant_message["tool_calls"][0]["id"] == "call_123"
        assert json.loads(
            assistant_message["tool_calls"][0]["function"]["arguments"]
        ) == {"location": "Bengaluru"}

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "I found some properties.",
                    },
                    "finish_reason": "stop",
                }]
            },
        )

    async def run():
        from backend.orchestrator.providers import ToolCall

        provider, client = make_provider(handler)
        async with client:
            await provider.chat_completion([
                Message(role=MessageRole.USER, content="Search"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            function_name="search_properties",
                            arguments={"location": "Bengaluru"},
                        )
                    ],
                ),
            ])

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments",
    [
        "{invalid-json}",
        "[]",
        "null",
    ],
)
def test_invalid_tool_arguments_raise_response_error(arguments):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search_properties",
                                "arguments": arguments,
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        )

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqResponseError):
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Search")
                ])

    asyncio.run(run())


def test_invalid_response_json_raises_response_error():
    def handler(request):
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqResponseError):
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

    asyncio.run(run())


def test_missing_choices_raise_response_error():
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqResponseError):
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

    asyncio.run(run())


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_errors(status_code):
    def handler(request):
        return httpx.Response(status_code, json={"error": "Unauthorized"})

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqAuthenticationError):
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

    asyncio.run(run())


def test_rate_limit_error_includes_retry_after():
    def handler(request):
        return httpx.Response(
            429,
            headers={"retry-after": "10"},
            json={"error": "Rate limit exceeded"},
        )

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqRateLimitError) as exc_info:
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

        assert exc_info.value.retry_after == "10"

    asyncio.run(run())


def test_server_error():
    def handler(request):
        return httpx.Response(500, json={"error": "Internal error"})

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqAPIError) as exc_info:
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

        assert exc_info.value.status_code == 500

    asyncio.run(run())


def test_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("Request timed out")

    async def run():
        provider, client = make_provider(handler)
        async with client:
            with pytest.raises(GroqTimeoutError):
                await provider.chat_completion([
                    Message(role=MessageRole.USER, content="Hello")
                ])

    asyncio.run(run())


def test_missing_api_key_is_rejected():
    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        GroqProvider(api_key="")


def test_empty_messages_are_rejected():
    async def run():
        provider, client = make_provider(
            lambda request: httpx.Response(200, json={})
        )
        async with client:
            with pytest.raises(ValueError, match="At least one message"):
                await provider.chat_completion([])

    asyncio.run(run())


def test_unsupported_request_option_is_rejected():
    async def run():
        provider, client = make_provider(
            lambda request: httpx.Response(200, json={})
        )
        async with client:
            with pytest.raises(ValueError, match="Unsupported Groq request options"):
                await provider.chat_completion(
                    [Message(role=MessageRole.USER, content="Hello")],
                    unsupported_option=True,
                )

    asyncio.run(run())


def test_invalid_message_role_is_rejected():
    async def run():
        provider, client = make_provider(
            lambda request: httpx.Response(200, json={})
        )
        async with client:
            with pytest.raises(ValueError, match="Unsupported message role"):
                await provider.chat_completion([
                    Message(role="customer", content="Hello")
                ])

    asyncio.run(run())
