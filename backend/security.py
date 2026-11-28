# -*- coding: utf-8 -*-
"""Request protection: request ids and access log, body size limits, rate limits.

Pure ASGI middleware (no extra dependencies). Order, outermost first:
request context -> body size limit -> rate limit -> application.
"""

import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from .logging_config import request_id_var

access_logger = logging.getLogger("backend.access")
logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


async def _send_json(send, status: int, body: dict, headers: Optional[Dict[str, str]] = None) -> None:
    payload = json.dumps(body).encode()
    raw_headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode(), str(v).encode()))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": payload})


# ---------------------------------------------------------------------------
# Request id + access log
# ---------------------------------------------------------------------------

def _route_template(scope) -> str:
    """The request path with path parameters (e.g. session ids) masked."""
    path = scope.get("path", "")
    for name, value in (scope.get("path_params") or {}).items():
        if value:
            path = path.replace(str(value), "{" + name + "}")
    return path


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log the route template, not the raw path: paths can contain
            # session ids.
            access_logger.info(
                "request",
                extra={
                    "method": scope.get("method"),
                    "route": _route_template(scope),
                    "status": status["code"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------

class _BodyTooLarge(Exception):
    pass


def _is_voice_upload(path: str) -> bool:
    return path.startswith("/api/voice/stt")


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies with 413 before they are read in full."""

    # Multipart framing around the audio itself.
    MULTIPART_OVERHEAD = 64 * 1024

    def __init__(self, app, max_json_bytes: int, max_audio_bytes: int):
        self.app = app
        self.max_json_bytes = max_json_bytes
        self.max_audio_bytes = max_audio_bytes

    def limit_for(self, path: str) -> int:
        if _is_voice_upload(path):
            return self.max_audio_bytes + self.MULTIPART_OVERHEAD
        return self.max_json_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)

        limit = self.limit_for(scope["path"])
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    return await _send_json(send, 413, {"detail": "Request body too large"})
            except ValueError:
                return await _send_json(send, 400, {"detail": "Invalid Content-Length"})

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await _send_json(send, 413, {"detail": "Request body too large"})


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@dataclass
class RateLimitConfig:
    enabled: bool = True
    agent_per_minute: int = 30
    voice_per_minute: int = 20
    default_per_minute: int = 120
    trust_proxy_headers: bool = False
    window_seconds: float = 60.0
    # Paths that are never limited (health checks, the page itself).
    exempt_prefixes: Tuple[str, ...] = ("/health", "/db-health", "/static", "/docs", "/openapi.json")
    exempt_exact: Tuple[str, ...] = ("/",)
    max_tracked_clients: int = 10_000


class SlidingWindowLimiter:
    """In-memory sliding-window counter keyed by (client, bucket).

    State lives in this process only: with several worker processes each has
    its own counters, so the effective limit is per worker.
    """

    def __init__(self, config: RateLimitConfig, clock=time.monotonic):
        self.config = config
        self.clock = clock
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def check(self, client: str, bucket: str, limit: int) -> Tuple[bool, int, float]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        now = self.clock()
        cutoff = now - self.config.window_seconds
        key = (client, bucket)
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.config.max_tracked_clients:
                self._evict(cutoff)
            hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return False, 0, max(0.0, hits[0] + self.config.window_seconds - now)
        hits.append(now)
        return True, limit - len(hits), 0.0

    def _evict(self, cutoff: float) -> None:
        for key in [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]:
            del self._hits[key]
        if len(self._hits) >= self.config.max_tracked_clients:
            # Still full of active clients: drop the oldest quarter.
            for key in sorted(self._hits, key=lambda k: self._hits[k][-1])[: len(self._hits) // 4 + 1]:
                del self._hits[key]


class RateLimitMiddleware:
    def __init__(self, app, config: RateLimitConfig, limiter: Optional[SlidingWindowLimiter] = None):
        self.app = app
        self.config = config
        self.limiter = limiter or SlidingWindowLimiter(config)

    def bucket_for(self, path: str) -> Tuple[str, int]:
        if path.startswith("/api/agent"):
            return "agent", self.config.agent_per_minute
        if path.startswith("/api/voice"):
            return "voice", self.config.voice_per_minute
        return "default", self.config.default_per_minute

    def client_id(self, scope) -> str:
        if self.config.trust_proxy_headers:
            forwarded = dict(scope.get("headers") or []).get(b"x-forwarded-for", b"").decode("latin-1")
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.config.enabled or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path in self.config.exempt_exact or path.startswith(self.config.exempt_prefixes):
            return await self.app(scope, receive, send)

        bucket, limit = self.bucket_for(path)
        allowed, remaining, retry_after = self.limiter.check(self.client_id(scope), bucket, limit)
        if not allowed:
            wait = max(1, int(retry_after + 0.999))
            logger.warning("Rate limit exceeded", extra={"bucket": bucket})
            return await _send_json(
                send, 429,
                {"detail": "Too many requests. Please wait a moment and try again."},
                {"Retry-After": wait, "X-RateLimit-Limit": limit, "X-RateLimit-Remaining": 0},
            )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    [(b"x-ratelimit-limit", str(limit).encode()), (b"x-ratelimit-remaining", str(remaining).encode())]
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)
