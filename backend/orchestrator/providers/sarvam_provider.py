
"""Sarvam AI voice provider — STT (Saaras) and TTS (Bulbul).

Wraps the Sarvam REST API for speech-to-text and text-to-speech
using httpx async client.  This provider is independent of the
LLM provider abstraction (BaseLLMProvider) since it handles
audio I/O rather than chat completions.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from ...config import SARVAM_API_KEY, SARVAM_BASE_URL
from .failures import FailureKind, is_quota_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SarvamProviderError(RuntimeError):
    """Base exception for Sarvam provider failures."""

    failure_kind = FailureKind.PROVIDER_ERROR


class SarvamAuthenticationError(SarvamProviderError):
    """Raised when Sarvam rejects the API credentials."""

    failure_kind = FailureKind.AUTH_FAILURE


class SarvamNetworkError(SarvamProviderError):
    """Raised when the Sarvam API cannot be reached."""

    failure_kind = FailureKind.NETWORK_ERROR


class SarvamRateLimitError(SarvamProviderError):
    """Raised when Sarvam rate-limits the request."""

    failure_kind = FailureKind.RATE_LIMITED

    def __init__(self, message: str, retry_after: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after


class SarvamQuotaExceededError(SarvamRateLimitError):
    """Raised when the Sarvam plan or credit limit is used up (do not retry)."""

    failure_kind = FailureKind.QUOTA_EXCEEDED


class SarvamTimeoutError(SarvamProviderError):
    """Raised when a Sarvam request times out."""

    failure_kind = FailureKind.TIMEOUT


class SarvamAPIError(SarvamProviderError):
    """Raised for non-success HTTP status codes."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class SarvamResponseError(SarvamProviderError):
    """Raised when the Sarvam response is malformed."""

    failure_kind = FailureKind.INVALID_RESPONSE


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

SUPPORTED_STT_LANGUAGES = frozenset(
    {
        "unknown",
        "hi-IN",
        "bn-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
        "en-IN",
        "gu-IN",
    }
)

SUPPORTED_TTS_LANGUAGES = frozenset(
    {
        "bn-IN",
        "en-IN",
        "gu-IN",
        "hi-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
    }
)


@dataclass(frozen=True)
class STTResult:
    """Result of a speech-to-text transcription."""

    transcript: str
    language_code: Optional[str] = None
    language_probability: Optional[float] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class TTSResult:
    """Result of a text-to-speech conversion."""

    audio_bytes: bytes
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

# Default voice for TTS — a clear English-Indian female voice from v4-flash.
DEFAULT_TTS_SPEAKER = "simran_en_sales"
DEFAULT_TTS_MODEL = "bulbul:v4-flash"
DEFAULT_STT_MODEL = "saaras:v4"
DEFAULT_TIMEOUT = 30.0


class SarvamProvider:
    """Async provider for Sarvam AI STT + TTS REST APIs.

    Usage::

        async with SarvamProvider() as sarvam:
            stt_result = await sarvam.speech_to_text(audio_bytes, "audio/wav")
            tts_result = await sarvam.text_to_speech("Hello!", "en-IN")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else SARVAM_API_KEY

        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "SARVAM_API_KEY is required to initialize SarvamProvider"
            )

        self.api_key = resolved_key.strip()
        self.base_url = (base_url or SARVAM_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        if self.timeout <= 0:
            raise ValueError("Sarvam timeout must be greater than zero")

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    # -- context-manager support -----------------------------------------------

    async def aclose(self) -> None:
        """Close the HTTP client if this provider owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "SarvamProvider":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    # -- helpers ---------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Headers required by every Sarvam API call."""
        return {"api-subscription-key": self.api_key}

    def _handle_error_status(self, response: httpx.Response) -> None:
        """Raise a typed exception for non-success status codes."""
        if response.status_code in (401, 403):
            raise SarvamAuthenticationError(
                "Sarvam rejected the API credentials"
            )
        if is_quota_response(response.status_code, response.text):
            raise SarvamQuotaExceededError(
                "Sarvam usage quota exceeded",
                retry_after=response.headers.get("retry-after"),
            )
        if response.status_code == 429:
            raise SarvamRateLimitError(
                "Sarvam API rate limit exceeded",
                retry_after=response.headers.get("retry-after"),
            )
        if response.is_error:
            # Try to extract an error message from the response body.
            try:
                body = response.json()
                detail = body.get("error", {}).get("message", "")
            except Exception:
                detail = response.text[:200]
            raise SarvamAPIError(
                response.status_code,
                f"Sarvam API returned HTTP {response.status_code}: {detail}",
            )

    # -- STT -------------------------------------------------------------------

    async def speech_to_text(
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        *,
        language_code: str = "unknown",
        model: str = DEFAULT_STT_MODEL,
        mode: str = "transcribe",
    ) -> STTResult:
        """Transcribe speech audio to text.

        Parameters
        ----------
        audio_data:
            Raw audio bytes (WAV, MP3, WebM, OGG, etc.).
        content_type:
            MIME type of the audio (used to pick file extension).
        language_code:
            BCP-47 code or ``"unknown"`` for auto-detect.
        model:
            Sarvam STT model identifier.
        mode:
            One of ``transcribe``, ``translate``, ``verbatim``,
            ``translit``, ``codemix``.

        Returns
        -------
        STTResult with the transcription text and detected language.
        """
        # Map MIME type to a sensible file extension for the multipart upload.
        ext_map = {
            "audio/wav": "audio.wav",
            "audio/x-wav": "audio.wav",
            "audio/mpeg": "audio.mp3",
            "audio/mp3": "audio.mp3",
            "audio/webm": "audio.webm",
            "audio/ogg": "audio.ogg",
            "audio/flac": "audio.flac",
            "audio/aac": "audio.aac",
            "audio/mp4": "audio.m4a",
        }
        filename = ext_map.get(content_type, "audio.wav")

        files = {"file": (filename, audio_data, content_type)}
        data = {
            "model": model,
            "language_code": language_code,
            "mode": mode,
        }

        logger.debug(
            "Sarvam STT request: model=%s language=%s mode=%s size=%d bytes",
            model,
            language_code,
            mode,
            len(audio_data),
        )

        try:
            response = await self._client.post(
                f"{self.base_url}/speech-to-text",
                headers=self._auth_headers(),
                files=files,
                data=data,
            )
        except httpx.TimeoutException as exc:
            raise SarvamTimeoutError("Sarvam STT request timed out") from exc
        except httpx.RequestError as exc:
            raise SarvamNetworkError(
                "Unable to connect to Sarvam API"
            ) from exc

        self._handle_error_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise SarvamResponseError(
                "Sarvam STT returned invalid JSON"
            ) from exc

        transcript = body.get("transcript")
        if transcript is None:
            raise SarvamResponseError(
                "Sarvam STT response missing 'transcript' field"
            )

        result = STTResult(
            transcript=transcript,
            language_code=body.get("language_code"),
            language_probability=body.get("language_probability"),
            request_id=body.get("request_id"),
        )

        logger.info(
            "Sarvam STT success: lang=%s prob=%.2f len=%d",
            result.language_code or "?",
            result.language_probability or 0.0,
            len(result.transcript),
        )
        return result

    # -- TTS -------------------------------------------------------------------

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "en-IN",
        *,
        model: str = DEFAULT_TTS_MODEL,
        speaker: str = DEFAULT_TTS_SPEAKER,
        pace: float = 1.0,
        pitch: float = 0.0,
        loudness: float = 1.0,
        sample_rate: int = 22050,
        output_codec: str = "mp3",
        enable_preprocessing: bool = True,
    ) -> TTSResult:
        """Convert text to speech audio.

        Parameters
        ----------
        text:
            Text to speak (max 2500 characters).
        language_code:
            BCP-47 code (e.g. ``en-IN``, ``kn-IN``, ``hi-IN``).
        model:
            Sarvam TTS model identifier.
        speaker:
            Voice name (must match the model catalogue).
        pace:
            Speaking speed, 0.5–2.0.
        pitch:
            Pitch shift, -0.5–0.5.
        loudness:
            Volume multiplier, 0.1–2.5.
        sample_rate:
            Output sample rate in Hz.
        output_codec:
            Audio codec — ``mp3``, ``wav``, ``opus``, ``flac``, etc.
        enable_preprocessing:
            Normalize English words and numeric entities.

        Returns
        -------
        TTSResult with decoded audio bytes ready to stream to a client.
        """
        if not text or not text.strip():
            raise ValueError("TTS text must not be empty")

        if len(text) > 2500:
            logger.warning(
                "TTS text exceeds 2500 chars (%d), truncating", len(text)
            )
            text = text[:2500]

        payload = {
            "text": text,
            "language_code": language_code,
            "model": model,
            "speaker": speaker,
            "pace": pace,
            "pitch": pitch,
            "loudness": loudness,
            "speech_sample_rate": sample_rate,
            "output_audio_codec": output_codec,
            "enable_preprocessing": enable_preprocessing,
        }

        logger.debug(
            "Sarvam TTS request: model=%s lang=%s speaker=%s len=%d",
            model,
            language_code,
            speaker,
            len(text),
        )

        try:
            response = await self._client.post(
                f"{self.base_url}/text-to-speech",
                headers={
                    **self._auth_headers(),
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise SarvamTimeoutError("Sarvam TTS request timed out") from exc
        except httpx.RequestError as exc:
            raise SarvamNetworkError(
                "Unable to connect to Sarvam API"
            ) from exc

        self._handle_error_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise SarvamResponseError(
                "Sarvam TTS returned invalid JSON"
            ) from exc

        audios = body.get("audios")
        if not audios or not isinstance(audios, list) or not audios[0]:
            raise SarvamResponseError(
                "Sarvam TTS response missing 'audios' field"
            )

        try:
            audio_bytes = base64.b64decode(audios[0])
        except Exception as exc:
            raise SarvamResponseError(
                "Failed to decode base64 audio from Sarvam TTS"
            ) from exc

        result = TTSResult(
            audio_bytes=audio_bytes,
            request_id=body.get("request_id"),
        )

        logger.info(
            "Sarvam TTS success: %d bytes of %s audio",
            len(result.audio_bytes),
            output_codec,
        )
        return result
