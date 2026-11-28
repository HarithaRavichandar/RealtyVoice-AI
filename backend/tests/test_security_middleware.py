# -*- coding: utf-8 -*-
"""Rate limiting (4.5), request size limits (4.6) and request ids (4.8)."""

import json
import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.security import (
    RateLimitConfig,
    RateLimitMiddleware,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    SlidingWindowLimiter,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def build_app(config=None, *, json_limit=100, audio_limit=1000, clock=None):
    config = config or RateLimitConfig(agent_per_minute=3, voice_per_minute=2, default_per_minute=5)
    limiter = SlidingWindowLimiter(config, clock=clock or Clock())
    app = FastAPI()

    @app.get("/")
    def root():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/thing")
    def thing():
        return {"ok": True}

    @app.post("/api/agent/turn")
    async def agent(request: Request):
        return {"size": len(await request.body())}

    @app.post("/api/voice/stt")
    async def stt(request: Request):
        return {"size": len(await request.body())}

    @app.post("/other")
    async def other(request: Request):
        return {"size": len(await request.body())}

    app.add_middleware(RateLimitMiddleware, config=config, limiter=limiter)
    app.add_middleware(RequestSizeLimitMiddleware, max_json_bytes=json_limit, max_audio_bytes=audio_limit)
    app.add_middleware(RequestContextMiddleware)
    app.state.limiter, app.state.config = limiter, config
    return app


@pytest.fixture
def app():
    return build_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_requests_under_the_limit_pass_with_remaining_count(client):
    r = client.post("/api/agent/turn", content=b"{}")
    assert r.status_code == 200
    assert r.headers["x-ratelimit-limit"] == "3"
    assert r.headers["x-ratelimit-remaining"] == "2"


def test_request_over_the_limit_gets_429_with_retry_after(client):
    for _ in range(3):
        assert client.post("/api/agent/turn", content=b"{}").status_code == 200
    r = client.post("/api/agent/turn", content=b"{}")
    assert r.status_code == 429
    assert r.json()["detail"].startswith("Too many requests")
    assert 1 <= int(r.headers["retry-after"]) <= 60
    assert r.headers["x-ratelimit-remaining"] == "0"


def test_window_slides_so_requests_are_allowed_again(app, client):
    clock = app.state.limiter.clock
    for _ in range(3):
        client.post("/api/agent/turn", content=b"{}")
    assert client.post("/api/agent/turn", content=b"{}").status_code == 429
    clock.now += 61
    assert client.post("/api/agent/turn", content=b"{}").status_code == 200


def test_buckets_are_independent(client):
    for _ in range(3):
        client.post("/api/agent/turn", content=b"{}")
    assert client.post("/api/agent/turn", content=b"{}").status_code == 429
    assert client.post("/api/voice/stt", content=b"x").status_code == 200  # own bucket
    assert client.get("/thing").status_code == 200


def test_voice_has_its_own_lower_limit(client):
    assert [client.post("/api/voice/stt", content=b"x").status_code for _ in range(3)] == [200, 200, 429]


def test_health_page_and_preflight_are_never_limited(client):
    for _ in range(20):
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        assert client.options("/api/agent/turn").status_code != 429


def test_limits_are_per_client(app):
    limiter = app.state.limiter
    assert limiter.check("1.1.1.1", "agent", 1)[0] is True
    assert limiter.check("1.1.1.1", "agent", 1)[0] is False
    assert limiter.check("2.2.2.2", "agent", 1)[0] is True


def test_disabled_limiter_lets_everything_through():
    app = build_app(RateLimitConfig(enabled=False, agent_per_minute=1))
    c = TestClient(app)
    assert all(c.post("/api/agent/turn", content=b"{}").status_code == 200 for _ in range(5))


def test_forwarded_for_is_ignored_unless_proxy_is_trusted():
    untrusted = TestClient(build_app(RateLimitConfig(agent_per_minute=1)))
    assert untrusted.post("/api/agent/turn", content=b"{}", headers={"X-Forwarded-For": "9.9.9.1"}).status_code == 200
    # A spoofed header must not give a fresh allowance.
    assert untrusted.post("/api/agent/turn", content=b"{}", headers={"X-Forwarded-For": "9.9.9.2"}).status_code == 429

    trusted = TestClient(build_app(RateLimitConfig(agent_per_minute=1, trust_proxy_headers=True)))
    assert trusted.post("/api/agent/turn", content=b"{}", headers={"X-Forwarded-For": "9.9.9.1"}).status_code == 200
    assert trusted.post("/api/agent/turn", content=b"{}", headers={"X-Forwarded-For": "9.9.9.2"}).status_code == 200
    assert trusted.post("/api/agent/turn", content=b"{}", headers={"X-Forwarded-For": "9.9.9.1, 10.0.0.1"}).status_code == 429


def test_memory_stays_bounded_with_many_clients():
    config = RateLimitConfig(max_tracked_clients=50)
    limiter = SlidingWindowLimiter(config, clock=Clock())
    for i in range(500):
        limiter.check(f"10.0.{i // 250}.{i % 250}", "agent", 5)
    assert len(limiter._hits) <= 50


def test_the_real_app_is_wired_with_the_limiter(monkeypatch):
    from backend import main

    monkeypatch.setattr(main.rate_limit_config, "enabled", True)
    monkeypatch.setattr(main.rate_limit_config, "default_per_minute", 2)
    main.rate_limiter.reset()
    try:
        with TestClient(main.app) as c:
            codes = [c.get("/no-such-page").status_code for _ in range(4)]
        assert codes == [404, 404, 429, 429]
    finally:
        main.rate_limiter.reset()


# ---------------------------------------------------------------------------
# Body size
# ---------------------------------------------------------------------------

def test_small_body_is_accepted(client):
    assert client.post("/other", content=b"x" * 100).json() == {"size": 100}


def test_oversize_content_length_is_rejected_before_reading(client):
    r = client.post("/other", content=b"x" * 101)
    assert r.status_code == 413
    assert r.json() == {"detail": "Request body too large"}


def test_chunked_body_without_content_length_is_still_limited(client):
    def chunks():
        for _ in range(10):
            yield b"x" * 20  # 200 bytes total, no Content-Length

    r = client.post("/other", content=chunks())
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_non_numeric_content_length_is_a_400():
    sent = []

    async def downstream(scope, receive, send):  # must never be reached
        raise AssertionError("request should have been rejected")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = RequestSizeLimitMiddleware(downstream, max_json_bytes=100, max_audio_bytes=1000)
    scope = {"type": "http", "method": "POST", "path": "/other", "headers": [(b"content-length", b"abc")]}
    await middleware(scope, receive, send)
    assert sent[0]["status"] == 400


def test_voice_upload_gets_the_larger_audio_limit(client):
    assert client.post("/api/voice/stt", content=b"x" * 500).status_code == 200
    assert client.post("/api/voice/stt", content=b"x" * (1000 + 64 * 1024 + 1)).status_code == 413


def test_agent_endpoint_uses_the_small_json_limit(client):
    assert client.post("/api/agent/turn", content=b"x" * 500).status_code == 413


def test_get_requests_are_not_size_checked(client):
    assert client.get("/thing").status_code == 200


def test_the_real_app_rejects_an_oversized_agent_request():
    from backend import main

    with TestClient(main.app) as c:
        r = c.post("/api/agent/turn", content=b"x" * (main.MAX_JSON_BODY_BYTES + 1),
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Request id and access log
# ---------------------------------------------------------------------------

def test_every_response_has_a_request_id(client):
    r = client.get("/thing")
    assert len(r.headers["x-request-id"]) >= 8


def test_a_sane_incoming_request_id_is_kept_and_a_bad_one_replaced(client):
    assert client.get("/thing", headers={"X-Request-ID": "trace-abc-12345"}).headers["x-request-id"] == "trace-abc-12345"
    bad = client.get("/thing", headers={"X-Request-ID": "bad id\twith spaces"})
    assert bad.headers["x-request-id"] != "bad id\twith spaces"
    assert len(bad.headers["x-request-id"]) == 32


def test_rejected_requests_also_carry_a_request_id(client):
    assert "x-request-id" in client.post("/other", content=b"x" * 500).headers


def test_access_log_records_route_status_and_timing_but_no_body(client, caplog):
    with caplog.at_level(logging.INFO, logger="backend.access"):
        client.post("/other", content=b'{"phone": "9876543210"}')
    record = next(r for r in caplog.records if r.name == "backend.access")
    assert (record.method, record.route, record.status) == ("POST", "/other", 200)
    assert record.duration_ms >= 0
    assert "9876543210" not in json.dumps(record.__dict__, default=str)


def test_access_log_masks_path_parameters_such_as_session_ids(caplog):
    app = build_app()

    @app.get("/api/conversations/{session_id}")
    def conv(session_id: str):
        return {}

    with caplog.at_level(logging.INFO, logger="backend.access"):
        TestClient(app).get("/api/conversations/abcdef1234567890abcdef1234567890")
    record = next(r for r in caplog.records if r.name == "backend.access")
    assert record.route == "/api/conversations/{session_id}"
    assert "abcdef1234" not in json.dumps(record.__dict__, default=str)
