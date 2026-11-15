
"""Groq Chat Completions provider using asynchronous HTTP requests."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import httpx

from ...config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
)
from .base import (
    BaseLLMProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)

class GroqProviderError(RuntimeError):
    """Base exception for Groq provider failures."""

class GroqAuthenticationError(GroqProviderError):
    """Raised when Groq rejects the API credentials."""

class GroqRateLimitError(GroqProviderError):
    """Raised when Groq rate limits the request."""

    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after

class GroqTimeoutError(GroqProviderError):
    """Raised when a Groq request times out."""

class GroqAPIError(GroqProviderError):
    """Raised when Groq returns an unsuccessful HTTP response."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code

class GroqResponseError(GroqProviderError):
    """Raised when Groq returns an unexpected response structure."""

class GroqProvider(BaseLLMProvider):
    """Asynchronous provider for Groq's OpenAI-compatible Chat API."""

    _SUPPORTED_OPTIONS = {
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "reasoning_effort",
        "tool_choice",
        "response_format",
        "user",
        "frequency_penalty",
        "presence_penalty",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else GROQ_API_KEY

        if not resolved_key or not resolved_key.strip():
            raise ValueError("GROQ_API_KEY is required to initialize GroqProvider")

        self.api_key = resolved_key.strip()
        self.model = model or GROQ_MODEL
        self.base_url = (base_url or GROQ_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else GROQ_TIMEOUT_SECONDS

        if self.timeout <= 0:
            raise ValueError("Groq timeout must be greater than zero")

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GroqProvider":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        """Convert a provider-independent message to API format."""
        role = getattr(message.role, "value", message.role)

        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role!r}")

        result: dict[str, Any] = {
            "role": role,
            "content": message.content,
        }

        if message.name is not None:
            result["name"] = message.name

        if message.tool_call_id is not None:
            result["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            if role != "assistant":
                raise ValueError("Tool calls can only be attached to assistant messages")

            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function_name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]

        return result

    @staticmethod
    def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
        """Convert a tool definition to Groq function-tool format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        """Parse and validate tool calls returned by Groq."""
        if raw_tool_calls is None:
            return []

        if not isinstance(raw_tool_calls, list):
            raise GroqResponseError("Groq tool_calls must be a list")

        parsed_calls = []

        for item in raw_tool_calls:
            try:
                call_id = item["id"]
                function = item["function"]
                function_name = function["name"]
                raw_arguments = function["arguments"]

                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Tool call ID is missing")

                if not isinstance(function_name, str) or not function_name:
                    raise ValueError("Tool function name is missing")

                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments)
                else:
                    arguments = raw_arguments

                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                parsed_calls.append(
                    ToolCall(
                        id=call_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise GroqResponseError(
                    "Groq returned an invalid tool call"
                ) from exc

        return parsed_calls

    @classmethod
    def _parse_response(cls, data: Any) -> ProviderResponse:
        """Convert a Groq completion response into ProviderResponse."""
        if not isinstance(data, dict):
            raise GroqResponseError("Groq response must be a JSON object")

        try:
            choices = data["choices"]

            if not isinstance(choices, list) or not choices:
                raise ValueError("Response has no choices")

            choice = choices[0]
            message = choice["message"]

            if not isinstance(message, dict):
                raise ValueError("Response message must be an object")

            content = message.get("content")

            if content is not None and not isinstance(content, str):
                raise ValueError("Assistant content must be a string or null")

            tool_calls = cls._parse_tool_calls(message.get("tool_calls"))

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason"),
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise GroqResponseError(
                "Groq returned an invalid chat completion response"
            ) from exc

    async def chat_completion(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Request a completion and normalize the response."""
        if not messages:
            raise ValueError("At least one message is required")

        unsupported = set(kwargs) - self._SUPPORTED_OPTIONS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported Groq request options: {names}")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                self._serialize_message(message) for message in messages
            ],
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                self._serialize_tool(tool) for tool in tools
            ]
            # GPT-OSS-20B does not support parallel tool calls.
            payload["parallel_tool_calls"] = False

        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        except httpx.TimeoutException as exc:
            raise GroqTimeoutError("Groq request timed out") from exc

        except httpx.RequestError as exc:
            raise GroqProviderError("Unable to connect to Groq API") from exc

        if response.status_code in (401, 403):
            raise GroqAuthenticationError(
                "Groq rejected the API credentials"
            )

        if response.status_code == 429:
            raise GroqRateLimitError(
                "Groq API rate limit exceeded",
                retry_after=response.headers.get("retry-after"),
            )

        if response.is_error:
            raise GroqAPIError(
                response.status_code,
                f"Groq API returned HTTP {response.status_code}",
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GroqResponseError(
                "Groq returned invalid JSON"
            ) from exc

        return self._parse_response(data)