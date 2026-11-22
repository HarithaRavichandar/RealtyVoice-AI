# -*- coding: utf-8 -*-
"""
Tests for Phase 3 agent improvements:
- Tool error hint generation
- Provider retry logic
- Iteration limit fallback messages
"""

import pytest
from unittest.mock import AsyncMock, patch

from backend.orchestrator.services.agent_service import (
    _build_tool_error_hint,
    _chat_with_retry,
    _TOOL_ERROR_HINTS,
)
from backend.orchestrator.providers.base import ProviderResponse


# ---------------------------------------------------------------
# _build_tool_error_hint tests
# ---------------------------------------------------------------

class TestBuildToolErrorHint:

    def test_known_error_code_lead_required(self):
        result = {"error": "lead_required", "tool": "create_callback_request"}
        hint = _build_tool_error_hint(result)
        assert "lead_required" in hint
        assert "register_lead" in hint

    def test_known_error_code_consent_required(self):
        result = {"error": "callback_consent_required"}
        hint = _build_tool_error_hint(result)
        assert "callback_consent_required" in hint
        assert "record_consent" in hint

    def test_validation_error_with_details(self):
        result = {
            "error": "validation",
            "details": [
                {"loc": ["customer_name"], "msg": "value cannot be whitespace only"},
            ],
        }
        hint = _build_tool_error_hint(result)
        assert "validation" in hint
        assert "customer_name" in hint

    def test_unknown_error_code(self):
        result = {"error": "some_new_error"}
        hint = _build_tool_error_hint(result)
        assert "some_new_error" in hint

    def test_all_known_codes_have_hints(self):
        for code in _TOOL_ERROR_HINTS:
            hint = _build_tool_error_hint({"error": code})
            assert code in hint
            assert len(hint) > len(code) + 20  # Has actual guidance


# ---------------------------------------------------------------
# _chat_with_retry tests
# ---------------------------------------------------------------

class TestChatWithRetry:

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        provider = AsyncMock()
        expected = ProviderResponse(content="Hello", tool_calls=[], finish_reason="stop")
        provider.chat_completion.return_value = expected

        result = await _chat_with_retry(provider, [], [])
        assert result == expected
        assert provider.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        from backend.orchestrator.providers.groq_provider import GroqTimeoutError

        provider = AsyncMock()
        expected = ProviderResponse(content="OK", tool_calls=[], finish_reason="stop")
        provider.chat_completion.side_effect = [
            GroqTimeoutError("timeout"),
            expected,
        ]

        with patch("backend.orchestrator.services.agent_service._PROVIDER_RETRY_DELAY", 0):
            result = await _chat_with_retry(provider, [], [])

        assert result == expected
        assert provider.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        from backend.orchestrator.providers.groq_provider import GroqRateLimitError

        provider = AsyncMock()
        expected = ProviderResponse(content="OK", tool_calls=[], finish_reason="stop")
        provider.chat_completion.side_effect = [
            GroqRateLimitError("rate limited"),
            expected,
        ]

        with patch("backend.orchestrator.services.agent_service._PROVIDER_RETRY_DELAY", 0):
            result = await _chat_with_retry(provider, [], [])

        assert result == expected
        assert provider.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_auth_error(self):
        from backend.orchestrator.providers.groq_provider import GroqAuthenticationError

        provider = AsyncMock()
        provider.chat_completion.side_effect = GroqAuthenticationError("bad key")

        with pytest.raises(GroqAuthenticationError):
            await _chat_with_retry(provider, [], [])

        assert provider.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        from backend.orchestrator.providers.groq_provider import GroqTimeoutError

        provider = AsyncMock()
        provider.chat_completion.side_effect = GroqTimeoutError("timeout")

        with patch("backend.orchestrator.services.agent_service._PROVIDER_RETRY_DELAY", 0):
            with pytest.raises(GroqTimeoutError):
                await _chat_with_retry(provider, [], [], max_retries=1)

        assert provider.chat_completion.call_count == 2
