"""Tests for the Sarvam AI voice provider (STT + TTS)."""

from __future__ import annotations

import base64
import json
import pytest
import httpx

from backend.orchestrator.providers.sarvam_provider import (
    SarvamProvider,
    SarvamAuthenticationError,
    SarvamRateLimitError,
    SarvamTimeoutError,
    SarvamAPIError,
    SarvamResponseError,
    SarvamProviderError,
    STTResult,
    TTSResult,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_SPEAKER,
    SUPPORTED_STT_LANGUAGES,
    SUPPORTED_TTS_LANGUAGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_API_KEY = "test-sarvam-key-12345"
FAKE_BASE_URL = "https://fake.sarvam.ai"

SAMPLE_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100


def _make_provider(
    client: httpx.AsyncClient | None = None,
    api_key: str = FAKE_API_KEY,
    **kwargs,
) -> SarvamProvider:
    """Create a SarvamProvider with a fake key and optional mock client."""
    return SarvamProvider(
        api_key=api_key,
        base_url=FAKE_BASE_URL,
        client=client,
        **kwargs,
    )


def _mock_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text: str = "",
    headers: dict | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    if json_body is not None:
        content = json.dumps(json_body).encode()
        content_type = "application/json"
    else:
        content = text.encode()
        content_type = "text/plain"

    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={
            "content-type": content_type,
            **(headers or {}),
        },
        request=httpx.Request("POST", FAKE_BASE_URL),
    )


class MockTransport(httpx.AsyncBaseTransport):
    """Transport that returns a pre-configured response."""

    def __init__(self, response: httpx.Response | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.last_request: httpx.Request | None = None
        self.last_request_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        # Read the streaming body so we can inspect it in tests.
        self.last_request_body = await request.aread()
        if self._exc:
            raise self._exc
        assert self._response is not None
        return self._response


# ===========================================================================
# Initialization
# ===========================================================================


class TestSarvamProviderInit:
    """Test provider initialization and validation."""

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="SARVAM_API_KEY is required"):
            SarvamProvider(api_key="", base_url=FAKE_BASE_URL)

    def test_requires_non_whitespace_key(self):
        with pytest.raises(ValueError, match="SARVAM_API_KEY is required"):
            SarvamProvider(api_key="   ", base_url=FAKE_BASE_URL)

    def test_rejects_zero_timeout(self):
        with pytest.raises(ValueError, match="timeout must be greater than zero"):
            SarvamProvider(api_key=FAKE_API_KEY, base_url=FAKE_BASE_URL, timeout=0)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout must be greater than zero"):
            SarvamProvider(api_key=FAKE_API_KEY, base_url=FAKE_BASE_URL, timeout=-5)

    def test_strips_trailing_slash_from_base_url(self):
        p = SarvamProvider(api_key=FAKE_API_KEY, base_url="https://api.sarvam.ai/")
        assert p.base_url == "https://api.sarvam.ai"

    def test_default_timeout(self):
        p = _make_provider()
        assert p.timeout == 30.0

    def test_custom_timeout(self):
        p = _make_provider(timeout=10.0)
        assert p.timeout == 10.0

    def test_strips_api_key_whitespace(self):
        p = SarvamProvider(api_key="  key123  ", base_url=FAKE_BASE_URL)
        assert p.api_key == "key123"


# ===========================================================================
# STT — speech_to_text
# ===========================================================================


class TestSpeechToText:
    """Test the speech_to_text method."""

    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        response = _mock_response(json_body={
            "transcript": "Hello, I am looking for a 2BHK in Whitefield",
            "language_code": "en-IN",
            "language_probability": 0.95,
            "request_id": "req-123",
        })
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        result = await provider.speech_to_text(SAMPLE_WAV_BYTES)

        assert isinstance(result, STTResult)
        assert result.transcript == "Hello, I am looking for a 2BHK in Whitefield"
        assert result.language_code == "en-IN"
        assert result.language_probability == 0.95
        assert result.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_sends_correct_multipart_fields(self):
        response = _mock_response(json_body={
            "transcript": "test",
            "language_code": "kn-IN",
        })
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        await provider.speech_to_text(
            SAMPLE_WAV_BYTES,
            "audio/webm",
            language_code="kn-IN",
            mode="translate",
        )

        req = transport.last_request
        assert req is not None
        assert req.url.path == "/speech-to-text"
        assert req.headers.get("api-subscription-key") == FAKE_API_KEY
        # multipart body should contain model, language_code, mode fields
        body = transport.last_request_body.decode("utf-8", errors="replace")
        assert "saaras:v4" in body
        assert "kn-IN" in body
        assert "translate" in body

    @pytest.mark.asyncio
    async def test_auto_detect_language(self):
        response = _mock_response(json_body={
            "transcript": "namaskara",
            "language_code": "kn-IN",
            "language_probability": 0.88,
        })
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        result = await provider.speech_to_text(SAMPLE_WAV_BYTES)
        # Default language_code is "unknown" (auto-detect)
        assert result.language_code == "kn-IN"

    @pytest.mark.asyncio
    async def test_auth_error_raises(self):
        response = _mock_response(status_code=403)
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamAuthenticationError):
            await provider.speech_to_text(SAMPLE_WAV_BYTES)

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self):
        response = _mock_response(
            status_code=429,
            headers={"retry-after": "5"},
        )
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamRateLimitError) as exc_info:
            await provider.speech_to_text(SAMPLE_WAV_BYTES)
        assert exc_info.value.retry_after == "5"

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        transport = MockTransport(exc=httpx.ReadTimeout("timeout"))
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamTimeoutError, match="timed out"):
            await provider.speech_to_text(SAMPLE_WAV_BYTES)

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        transport = MockTransport(exc=httpx.ConnectError("refused"))
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamProviderError, match="Unable to connect"):
            await provider.speech_to_text(SAMPLE_WAV_BYTES)

    @pytest.mark.asyncio
    async def test_missing_transcript_raises(self):
        response = _mock_response(json_body={"language_code": "en-IN"})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="missing 'transcript'"):
            await provider.speech_to_text(SAMPLE_WAV_BYTES)

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        response = _mock_response(status_code=200, text="not json")
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="invalid JSON"):
            await provider.speech_to_text(SAMPLE_WAV_BYTES)

    @pytest.mark.asyncio
    async def test_server_error_raises(self):
        response = _mock_response(
            status_code=500,
            json_body={"error": {"message": "internal", "code": 500}},
        )
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamAPIError) as exc_info:
            await provider.speech_to_text(SAMPLE_WAV_BYTES)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_mime_type_mapping_mp3(self):
        response = _mock_response(json_body={
            "transcript": "test",
        })
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        await provider.speech_to_text(SAMPLE_WAV_BYTES, "audio/mpeg")
        body = transport.last_request_body.decode("utf-8", errors="replace")
        assert "audio.mp3" in body


# ===========================================================================
# TTS — text_to_speech
# ===========================================================================


class TestTextToSpeech:
    """Test the text_to_speech method."""

    @pytest.mark.asyncio
    async def test_successful_synthesis(self):
        fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 100  # fake MP3 header
        b64_audio = base64.b64encode(fake_audio).decode()

        response = _mock_response(json_body={
            "audios": [b64_audio],
            "request_id": "tts-456",
        })
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        result = await provider.text_to_speech("Welcome to RealtyVoice!", "en-IN")

        assert isinstance(result, TTSResult)
        assert result.audio_bytes == fake_audio
        assert result.request_id == "tts-456"

    @pytest.mark.asyncio
    async def test_sends_correct_json_payload(self):
        b64_audio = base64.b64encode(b"audio").decode()
        response = _mock_response(json_body={"audios": [b64_audio]})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        await provider.text_to_speech(
            "Namaskara!",
            "kn-IN",
            speaker="meera",
            pace=1.2,
            pitch=0.1,
        )

        req = transport.last_request
        assert req is not None
        assert req.url.path == "/text-to-speech"
        assert req.headers.get("api-subscription-key") == FAKE_API_KEY
        assert req.headers.get("content-type") == "application/json"

        payload = json.loads(req.content)
        assert payload["text"] == "Namaskara!"
        assert payload["language_code"] == "kn-IN"
        assert payload["model"] == DEFAULT_TTS_MODEL
        assert payload["speaker"] == "meera"
        assert payload["pace"] == 1.2
        assert payload["pitch"] == 0.1

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        provider = _make_provider()
        with pytest.raises(ValueError, match="must not be empty"):
            await provider.text_to_speech("", "en-IN")

    @pytest.mark.asyncio
    async def test_whitespace_text_raises(self):
        provider = _make_provider()
        with pytest.raises(ValueError, match="must not be empty"):
            await provider.text_to_speech("   ", "en-IN")

    @pytest.mark.asyncio
    async def test_long_text_truncated(self):
        long_text = "A" * 3000
        b64_audio = base64.b64encode(b"audio").decode()
        response = _mock_response(json_body={"audios": [b64_audio]})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        await provider.text_to_speech(long_text, "en-IN")

        payload = json.loads(transport.last_request.content)
        assert len(payload["text"]) == 2500

    @pytest.mark.asyncio
    async def test_auth_error_raises(self):
        response = _mock_response(status_code=401)
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamAuthenticationError):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self):
        response = _mock_response(status_code=429)
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamRateLimitError):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        transport = MockTransport(exc=httpx.ReadTimeout("timeout"))
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamTimeoutError):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_missing_audios_field_raises(self):
        response = _mock_response(json_body={"request_id": "x"})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="missing 'audios'"):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_empty_audios_list_raises(self):
        response = _mock_response(json_body={"audios": []})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="missing 'audios'"):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_invalid_base64_raises(self):
        response = _mock_response(json_body={"audios": ["not!valid!base64!!!"]})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="decode base64"):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises(self):
        response = _mock_response(status_code=200, text="<html>error</html>")
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        with pytest.raises(SarvamResponseError, match="invalid JSON"):
            await provider.text_to_speech("Hello", "en-IN")

    @pytest.mark.asyncio
    async def test_enable_preprocessing_sent(self):
        b64_audio = base64.b64encode(b"audio").decode()
        response = _mock_response(json_body={"audios": [b64_audio]})
        transport = MockTransport(response=response)
        client = httpx.AsyncClient(transport=transport)
        provider = _make_provider(client=client)

        await provider.text_to_speech("₹50 lakhs", "en-IN", enable_preprocessing=True)

        payload = json.loads(transport.last_request.content)
        assert payload["enable_preprocessing"] is True


# ===========================================================================
# Context manager / cleanup
# ===========================================================================


class TestContextManager:
    """Test async context manager and cleanup."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with _make_provider() as provider:
            assert isinstance(provider, SarvamProvider)

    @pytest.mark.asyncio
    async def test_external_client_not_closed(self):
        client = httpx.AsyncClient()
        provider = _make_provider(client=client)
        await provider.aclose()
        # External client should still be open
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.asyncio
    async def test_owned_client_closed(self):
        provider = _make_provider()
        assert provider._owns_client is True
        await provider.aclose()
        assert provider._client.is_closed


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify exported language sets and defaults."""

    def test_stt_languages_include_key_indian_languages(self):
        for lang in ("en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN"):
            assert lang in SUPPORTED_STT_LANGUAGES

    def test_stt_languages_include_auto_detect(self):
        assert "unknown" in SUPPORTED_STT_LANGUAGES

    def test_tts_languages_include_key_indian_languages(self):
        for lang in ("en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN"):
            assert lang in SUPPORTED_TTS_LANGUAGES

    def test_tts_languages_no_auto_detect(self):
        # TTS requires explicit language
        assert "unknown" not in SUPPORTED_TTS_LANGUAGES

    def test_default_models(self):
        assert DEFAULT_STT_MODEL == "saaras:v4"
        assert DEFAULT_TTS_MODEL == "bulbul:v4-flash"
