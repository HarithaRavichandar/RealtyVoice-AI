
"""Voice API endpoints for RealtyVoice AI.

Provides:
- POST /api/voice/stt  — Upload audio, get transcribed text
- POST /api/voice/tts  — Send text + language, get audio back
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import SARVAM_API_KEY
from .providers.sarvam_provider import (
    SarvamProvider,
    SarvamAuthenticationError,
    SarvamRateLimitError,
    SarvamTimeoutError,
    SarvamAPIError,
    SarvamResponseError,
    SarvamProviderError,
    SUPPORTED_TTS_LANGUAGES,
)


logger = logging.getLogger(__name__)

voice_router = APIRouter(tags=["voice"])

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# MIME types the browser's MediaRecorder typically produces.
_ACCEPTED_AUDIO_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/webm",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
        "audio/mp4",
        "audio/m4a",
        "application/octet-stream",  # fallback when browser doesn't set type
    }
)

# Max upload: 10 MB (Sarvam REST API accepts ≤30 s of audio).
_MAX_AUDIO_BYTES = 10 * 1024 * 1024


def _require_sarvam_key() -> None:
    """Raise 503 early if the Sarvam API key is not configured."""
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Voice services are not configured (missing SARVAM_API_KEY)",
        )


def _handle_sarvam_error(exc: SarvamProviderError) -> None:
    """Map Sarvam provider exceptions to appropriate HTTP errors."""
    if isinstance(exc, SarvamAuthenticationError):
        logger.error("Sarvam authentication failed")
        raise HTTPException(status_code=502, detail="Voice service authentication failed")
    if isinstance(exc, SarvamRateLimitError):
        logger.warning("Sarvam rate limited: retry_after=%s", exc.retry_after)
        raise HTTPException(status_code=429, detail="Voice service rate limit exceeded")
    if isinstance(exc, SarvamTimeoutError):
        logger.warning("Sarvam request timed out")
        raise HTTPException(status_code=504, detail="Voice service timed out")
    if isinstance(exc, SarvamAPIError):
        logger.error("Sarvam API error: %d", exc.status_code)
        raise HTTPException(status_code=502, detail="Voice service returned an error")
    if isinstance(exc, SarvamResponseError):
        logger.error("Sarvam response malformed: %s", exc)
        raise HTTPException(status_code=502, detail="Voice service returned an unexpected response")
    # Generic fallback
    logger.exception("Unexpected Sarvam error: %s", exc)
    raise HTTPException(status_code=502, detail="Voice service is temporarily unavailable")


# ---------------------------------------------------------------------------
# POST /api/voice/stt — Speech-to-Text
# ---------------------------------------------------------------------------

@voice_router.post("/voice/stt")
async def speech_to_text(
    file: UploadFile = File(..., description="Audio file (WAV, WebM, MP3, OGG, etc.)"),
    language: str = Form("unknown", description="BCP-47 language code or 'unknown' for auto-detect"),
):
    """Transcribe uploaded audio to text using Sarvam Saaras v4.

    Returns JSON with the transcript and detected language.
    """
    _require_sarvam_key()

    # Validate content type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in _ACCEPTED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {content_type}. Send WAV, WebM, MP3, OGG, or similar.",
        )

    # Read and validate size
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(audio_data) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_data)} bytes). Maximum is {_MAX_AUDIO_BYTES} bytes.",
        )

    logger.info(
        "STT request: content_type=%s size=%d language=%s",
        content_type,
        len(audio_data),
        language,
    )

    try:
        async with SarvamProvider() as sarvam:
            result = await sarvam.speech_to_text(
                audio_data=audio_data,
                content_type=content_type,
                language_code=language,
            )
    except SarvamProviderError as exc:
        _handle_sarvam_error(exc)

    return {
        "transcript": result.transcript,
        "language_code": result.language_code,
        "language_probability": result.language_probability,
    }


# ---------------------------------------------------------------------------
# POST /api/voice/tts — Text-to-Speech
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """Request body for the TTS endpoint."""

    text: str = Field(..., min_length=1, max_length=2500, description="Text to synthesize")
    language_code: str = Field("en-IN", description="BCP-47 language code")
    speaker: Optional[str] = Field(None, description="Voice name (omit for default)")
    pace: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed")


# Content-type for the TTS audio response.
_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


@voice_router.post("/voice/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech audio using Sarvam Bulbul v4 Flash.

    Returns the audio as an MP3 binary response.
    """
    _require_sarvam_key()

    logger.info(
        "TTS request: lang=%s speaker=%s len=%d",
        request.language_code,
        request.speaker or "default",
        len(request.text),
    )

    tts_kwargs = {
        "text": request.text,
        "language_code": request.language_code,
        "pace": request.pace,
        "output_codec": "mp3",
        "enable_preprocessing": True,
    }

    if request.speaker:
        tts_kwargs["speaker"] = request.speaker

    try:
        async with SarvamProvider() as sarvam:
            result = await sarvam.text_to_speech(**tts_kwargs)
    except SarvamProviderError as exc:
        _handle_sarvam_error(exc)

    return Response(
        content=result.audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=speech.mp3",
        },
    )
