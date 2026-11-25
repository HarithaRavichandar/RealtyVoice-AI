"""Tests for voice API endpoints (/api/voice/stt and /api/voice/tts)."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.orchestrator.providers.sarvam_provider import (
    STTResult,
    TTSResult,
    SarvamAuthenticationError,
    SarvamRateLimitError,
    SarvamTimeoutError,
    SarvamAPIError,
    SarvamResponseError,
    SarvamProviderError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100
FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 100


@pytest.fixture()
def client():
    """TestClient with SARVAM_API_KEY always set."""
    with patch("backend.orchestrator.voice_router.SARVAM_API_KEY", "test-key"):
        from backend.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_no_key():
    """TestClient with SARVAM_API_KEY empty (voice not configured)."""
    with patch("backend.orchestrator.voice_router.SARVAM_API_KEY", ""):
        from backend.main import app
        with TestClient(app) as c:
            yield c


def _mock_sarvam_provider(stt_result=None, tts_result=None, exc=None):
    """Return a mock that replaces SarvamProvider as an async context manager."""
    mock_instance = AsyncMock()

    if exc:
        mock_instance.speech_to_text.side_effect = exc
        mock_instance.text_to_speech.side_effect = exc
    else:
        if stt_result:
            mock_instance.speech_to_text.return_value = stt_result
        if tts_result:
            mock_instance.text_to_speech.return_value = tts_result

    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance


# ===========================================================================
# STT endpoint — POST /api/voice/stt
# ===========================================================================


class TestSTTEndpoint:
    """Tests for the speech-to-text endpoint."""

    def test_successful_transcription(self, client):
        stt_result = STTResult(
            transcript="I want a 3BHK in Whitefield",
            language_code="en-IN",
            language_probability=0.95,
            request_id="req-1",
        )
        mock_cls, _ = _mock_sarvam_provider(stt_result=stt_result)

        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
                data={"language": "en-IN"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["transcript"] == "I want a 3BHK in Whitefield"
        assert body["language_code"] == "en-IN"
        assert body["language_probability"] == 0.95

    def test_auto_detect_language(self, client):
        stt_result = STTResult(
            transcript="namaskara",
            language_code="kn-IN",
            language_probability=0.88,
        )
        mock_cls, _ = _mock_sarvam_provider(stt_result=stt_result)

        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.webm", FAKE_WAV, "audio/webm")},
                # No language param — defaults to "unknown"
            )

        assert resp.status_code == 200
        assert resp.json()["language_code"] == "kn-IN"

    def test_missing_file_returns_422(self, client):
        resp = client.post("/api/voice/stt")
        assert resp.status_code == 422

    def test_empty_file_returns_400(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            stt_result=STTResult(transcript="x")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", b"", "audio/wav")},
            )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_oversized_file_returns_413(self, client):
        big_audio = b"\x00" * (10 * 1024 * 1024 + 1)
        mock_cls, _ = _mock_sarvam_provider(
            stt_result=STTResult(transcript="x")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", big_audio, "audio/wav")},
            )
        assert resp.status_code == 413

    def test_unsupported_content_type_returns_415(self, client):
        resp = client.post(
            "/api/voice/stt",
            files={"file": ("doc.pdf", b"fake", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_no_api_key_returns_503(self, client_no_key):
        resp = client_no_key.post(
            "/api/voice/stt",
            files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
        )
        assert resp.status_code == 503
        assert "SARVAM_API_KEY" in resp.json()["detail"]

    def test_sarvam_auth_error_returns_502(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            exc=SarvamAuthenticationError("bad creds")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            )
        assert resp.status_code == 502

    def test_sarvam_rate_limit_returns_429(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            exc=SarvamRateLimitError("rate limited", retry_after="10")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            )
        assert resp.status_code == 429

    def test_sarvam_timeout_returns_504(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            exc=SarvamTimeoutError("timed out")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/stt",
                files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            )
        assert resp.status_code == 504


# ===========================================================================
# TTS endpoint — POST /api/voice/tts
# ===========================================================================


class TestTTSEndpoint:
    """Tests for the text-to-speech endpoint."""

    def test_successful_synthesis(self, client):
        tts_result = TTSResult(audio_bytes=FAKE_MP3, request_id="tts-1")
        mock_cls, _ = _mock_sarvam_provider(tts_result=tts_result)

        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/tts",
                json={"text": "Welcome to RealtyVoice!", "language_code": "en-IN"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert resp.content == FAKE_MP3

    def test_custom_speaker_and_pace(self, client):
        tts_result = TTSResult(audio_bytes=FAKE_MP3)
        mock_cls, mock_instance = _mock_sarvam_provider(tts_result=tts_result)

        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/tts",
                json={
                    "text": "Namaskara!",
                    "language_code": "kn-IN",
                    "speaker": "meera",
                    "pace": 1.3,
                },
            )

        assert resp.status_code == 200
        # Verify the provider was called with the custom speaker
        call_kwargs = mock_instance.text_to_speech.call_args
        assert call_kwargs.kwargs.get("speaker") == "meera" or \
               (call_kwargs[1].get("speaker") == "meera" if call_kwargs[1] else False)

    def test_empty_text_returns_422(self, client):
        resp = client.post(
            "/api/voice/tts",
            json={"text": "", "language_code": "en-IN"},
        )
        assert resp.status_code == 422

    def test_missing_text_returns_422(self, client):
        resp = client.post(
            "/api/voice/tts",
            json={"language_code": "en-IN"},
        )
        assert resp.status_code == 422

    def test_text_too_long_returns_422(self, client):
        resp = client.post(
            "/api/voice/tts",
            json={"text": "A" * 2501, "language_code": "en-IN"},
        )
        assert resp.status_code == 422

    def test_pace_too_low_returns_422(self, client):
        resp = client.post(
            "/api/voice/tts",
            json={"text": "Hello", "language_code": "en-IN", "pace": 0.1},
        )
        assert resp.status_code == 422

    def test_pace_too_high_returns_422(self, client):
        resp = client.post(
            "/api/voice/tts",
            json={"text": "Hello", "language_code": "en-IN", "pace": 3.0},
        )
        assert resp.status_code == 422

    def test_no_api_key_returns_503(self, client_no_key):
        resp = client_no_key.post(
            "/api/voice/tts",
            json={"text": "Hello", "language_code": "en-IN"},
        )
        assert resp.status_code == 503

    def test_sarvam_auth_error_returns_502(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            exc=SarvamAuthenticationError("bad creds")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/tts",
                json={"text": "Hello", "language_code": "en-IN"},
            )
        assert resp.status_code == 502

    def test_sarvam_timeout_returns_504(self, client):
        mock_cls, _ = _mock_sarvam_provider(
            exc=SarvamTimeoutError("timed out")
        )
        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/tts",
                json={"text": "Hello", "language_code": "en-IN"},
            )
        assert resp.status_code == 504

    def test_default_language_is_en_in(self, client):
        tts_result = TTSResult(audio_bytes=FAKE_MP3)
        mock_cls, mock_instance = _mock_sarvam_provider(tts_result=tts_result)

        with patch("backend.orchestrator.voice_router.SarvamProvider", mock_cls):
            resp = client.post(
                "/api/voice/tts",
                json={"text": "Hello"},
            )

        assert resp.status_code == 200
        # language_code should default to en-IN
        call_kwargs = mock_instance.text_to_speech.call_args
        assert "en-IN" in str(call_kwargs)
