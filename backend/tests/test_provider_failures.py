# -*- coding: utf-8 -*-
"""Every external call tells the failure kinds apart (task 4.9)."""

import httpx
import pytest

from backend.orchestrator.providers.failures import (
    CUSTOMER_MESSAGES,
    HTTP_STATUS,
    FailureKind,
    classify,
    is_quota_response,
)
from backend.orchestrator.providers.base import Message, MessageRole
from backend.orchestrator.providers.groq_provider import GroqProvider
from backend.orchestrator.providers.mappls_provider import MappslsProvider
from backend.orchestrator.providers.sarvam_provider import SarvamProvider


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_error(status, body="", headers=None):
    return lambda request: httpx.Response(status, text=body, headers=headers or {})


def raising(exc):
    def handler(request):
        raise exc
    return handler


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def call_groq(handler):
    provider = GroqProvider(api_key="k", client=client_for(handler))
    return await provider.chat_completion([Message(role=MessageRole.USER, content="hi")])


async def call_sarvam_stt(handler):
    provider = SarvamProvider(api_key="k", client=client_for(handler))
    return await provider.speech_to_text(b"RIFF", "audio/wav")


async def call_sarvam_tts(handler):
    provider = SarvamProvider(api_key="k", client=client_for(handler))
    return await provider.text_to_speech("hello")


async def call_mappls_geocode(handler):
    provider = MappslsProvider(api_key="k", client=client_for(handler))
    return await provider.geocode("Whitefield")


async def call_mappls_nearby(handler):
    provider = MappslsProvider(api_key="k", client=client_for(handler))
    return await provider.nearby(latitude=12.9, longitude=77.7, keywords="hospital")


CALLS = {
    "groq": call_groq,
    "sarvam_stt": call_sarvam_stt,
    "sarvam_tts": call_sarvam_tts,
    "mappls_geocode": call_mappls_geocode,
    "mappls_nearby": call_mappls_nearby,
}

SCENARIOS = [
    ("auth_401", http_error(401), FailureKind.AUTH_FAILURE),
    ("rate_limited", http_error(429, "slow down", {"retry-after": "7"}), FailureKind.RATE_LIMITED),
    ("quota_402", http_error(402, "payment required"), FailureKind.QUOTA_EXCEEDED),
    ("quota_429", http_error(429, '{"error": "You exceeded your current quota"}'), FailureKind.QUOTA_EXCEEDED),
    ("server_500", http_error(500, "boom"), FailureKind.PROVIDER_ERROR),
    ("timeout", raising(httpx.ReadTimeout("slow")), FailureKind.TIMEOUT),
    ("connect_error", raising(httpx.ConnectError("no route")), FailureKind.NETWORK_ERROR),
    ("read_error", raising(httpx.ReadError("reset")), FailureKind.NETWORK_ERROR),
    ("bad_json", http_error(200, "<html>not json</html>"), FailureKind.INVALID_RESPONSE),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(CALLS))
@pytest.mark.parametrize("name, handler, expected", SCENARIOS, ids=[s[0] for s in SCENARIOS])
async def test_each_provider_classifies_each_failure(provider, name, handler, expected):
    if provider == "groq" and name == "bad_json":
        pass  # same behaviour; all providers must report invalid_response
    with pytest.raises(Exception) as caught:
        await CALLS[provider](handler)
    assert classify(caught.value) is expected, f"{provider}/{name}: {caught.value!r}"


@pytest.mark.asyncio
async def test_quota_is_not_treated_as_a_retryable_rate_limit_by_name():
    # The agent retries only errors named in _RETRYABLE_PROVIDER_ERRORS.
    from backend.orchestrator.services.agent_service import _RETRYABLE_PROVIDER_ERRORS

    with pytest.raises(Exception) as caught:
        await call_groq(http_error(402))
    assert type(caught.value).__name__ not in _RETRYABLE_PROVIDER_ERRORS
    assert classify(caught.value) is FailureKind.QUOTA_EXCEEDED


@pytest.mark.asyncio
async def test_plain_rate_limit_keeps_its_retry_after_and_stays_retryable():
    from backend.orchestrator.services.agent_service import _RETRYABLE_PROVIDER_ERRORS

    with pytest.raises(Exception) as caught:
        await call_groq(http_error(429, "slow down", {"retry-after": "7"}))
    assert caught.value.retry_after == "7"
    assert type(caught.value).__name__ in _RETRYABLE_PROVIDER_ERRORS


# ---------------------------------------------------------------------------
# "No results" is not a failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_geocode_result_is_none_not_an_error():
    assert await call_mappls_geocode(lambda r: httpx.Response(200, json={"copResults": []})) is None


@pytest.mark.asyncio
async def test_nearby_204_is_an_empty_result_not_an_error():
    result = await call_mappls_nearby(lambda r: httpx.Response(204))
    assert result.places == [] and result.error is None


@pytest.mark.asyncio
async def test_no_results_from_the_tool_is_labelled_distinctly(monkeypatch):
    from backend.orchestrator.services import tool_executor

    class NoResults:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def geocode(self, location): return None

    monkeypatch.setattr("backend.orchestrator.providers.mappls_provider.MappslsProvider", NoResults)
    monkeypatch.setattr("backend.config.MAPPLS_API_KEY", "k")
    result = await tool_executor.execute_nearby_amenities("Nowhereville")
    assert result["error"] == "geocode_failed"
    assert result["failure_kind"] == FailureKind.NO_RESULTS.value


@pytest.mark.asyncio
@pytest.mark.parametrize("name, handler, expected", [s for s in SCENARIOS if s[0] not in ("bad_json",)], ids=[s[0] for s in SCENARIOS if s[0] != "bad_json"])
async def test_amenity_tool_reports_the_failure_kind_without_provider_text(monkeypatch, name, handler, expected):
    from backend.orchestrator.services import tool_executor

    real = MappslsProvider

    def factory(**kw):
        return real(api_key="SECRET-TOKEN", client=client_for(handler))

    monkeypatch.setattr("backend.orchestrator.providers.mappls_provider.MappslsProvider", factory)
    monkeypatch.setattr("backend.config.MAPPLS_API_KEY", "SECRET-TOKEN")

    result = await tool_executor.execute_nearby_amenities("Whitefield")

    assert result["error"] == "mappls_api_error"
    assert result["failure_kind"] == expected.value
    assert result["message"] == CUSTOMER_MESSAGES[expected]
    assert "SECRET-TOKEN" not in str(result)
    assert "boom" not in str(result)


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status, body, expected",
    [
        (402, "", True),
        (429, "Quota exceeded for this plan", True),
        (429, "insufficient credits", True),
        (429, "Too many requests, slow down", False),
        (429, "", False),
        (500, "quota", False),
        (200, "quota", False),
    ],
)
def test_quota_detection(status, body, expected):
    assert is_quota_response(status, body) is expected


def test_every_customer_visible_kind_has_wording_and_a_status():
    visible = set(FailureKind) - {FailureKind.NO_RESULTS}
    assert set(CUSTOMER_MESSAGES) == visible == set(HTTP_STATUS)
    for text in CUSTOMER_MESSAGES.values():
        assert not any(w in text.lower() for w in ("groq", "sarvam", "mappls", "token", "key"))


def test_unclassified_exceptions_have_no_kind():
    assert classify(RuntimeError("x")) is None
