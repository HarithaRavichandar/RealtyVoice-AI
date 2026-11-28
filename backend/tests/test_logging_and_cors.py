# -*- coding: utf-8 -*-
"""Structured JSON logging (4.8) and CORS configuration (4.7)."""

import json
import logging
import logging.config

import pytest
from fastapi.testclient import TestClient

from backend.config import parse_cors_origins
from backend.logging_config import JsonFormatter, build_logging_config, request_id_var


def record(msg="hello", level=logging.INFO, exc_info=None, **extra):
    r = logging.LogRecord("backend.test", level, __file__, 10, msg, (), exc_info)
    for k, v in extra.items():
        setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------

def test_each_record_is_one_json_object_with_the_core_fields():
    line = JsonFormatter().format(record("hello"))
    data = json.loads(line)
    assert "\n" not in line
    assert (data["level"], data["logger"], data["message"]) == ("INFO", "backend.test", "hello")
    assert data["timestamp"].endswith("+00:00") and data["request_id"] == "-"


def test_message_arguments_are_interpolated():
    r = logging.LogRecord("x", logging.WARNING, __file__, 1, "limit %s hit by %d", ("agent", 3), None)
    assert json.loads(JsonFormatter().format(r))["message"] == "limit agent hit by 3"


def test_extra_fields_become_json_keys():
    data = json.loads(JsonFormatter().format(record(route="/api/x", status=200)))
    assert data["route"] == "/api/x" and data["status"] == 200


def test_current_request_id_is_included():
    token = request_id_var.set("req-123456")
    try:
        assert json.loads(JsonFormatter().format(record()))["request_id"] == "req-123456"
    finally:
        request_id_var.reset(token)


def test_exceptions_are_included_as_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        data = json.loads(JsonFormatter().format(record("failed", logging.ERROR, exc_info=sys.exc_info())))
    assert "ValueError: boom" in data["exception"]


def test_unserialisable_values_do_not_break_logging():
    data = json.loads(JsonFormatter().format(record(thing=object())))
    assert "object" in data["thing"]


def test_unicode_is_kept_readable():
    assert "₹50 lakh" in JsonFormatter().format(record("budget ₹50 lakh"))


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_logging_config_is_valid_for_both_formats(fmt, capsys):
    logging.config.dictConfig(build_logging_config("INFO", fmt))
    logging.getLogger("backend.check").info("configured")
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "configured" in out
    if fmt == "json":
        assert json.loads(out)["message"] == "configured"


def test_invalid_log_format_is_refused_at_startup(monkeypatch):
    import importlib
    import backend.config as config

    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValueError):
        importlib.reload(config)
    monkeypatch.delenv("LOG_FORMAT")
    importlib.reload(config)


# ---------------------------------------------------------------------------
# CORS origin parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://localhost:8000", ["http://localhost:8000"]),
        ("https://app.example.com, https://www.example.com/", ["https://app.example.com", "https://www.example.com"]),
        ("https://a.com,https://a.com", ["https://a.com"]),
        ("", []),
        (" , ", []),
        ("http://127.0.0.1:5173", ["http://127.0.0.1:5173"]),
    ],
)
def test_valid_origins_are_normalised(raw, expected):
    assert parse_cors_origins(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["*", "https://a.com,*", "example.com", "ftp://a.com", "https://a.com/app", "https://a.com?x=1",
     "https://", "javascript:alert(1)", "https://a.com:99999999"],
)
def test_unsafe_or_malformed_origins_are_refused(raw):
    with pytest.raises(ValueError):
        parse_cors_origins(raw)


# ---------------------------------------------------------------------------
# CORS behaviour of the real app
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


def preflight(client, origin, method="POST"):
    return client.options(
        "/api/agent/turn",
        headers={"Origin": origin, "Access-Control-Request-Method": method,
                 "Access-Control-Request-Headers": "content-type"},
    )


def test_allowed_origin_passes_preflight(client):
    r = preflight(client, "http://localhost:8000")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:8000"


def test_other_origins_are_refused(client):
    r = preflight(client, "https://evil.example.com")
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_credentials_are_not_allowed_by_default(client):
    r = preflight(client, "http://localhost:8000")
    assert "access-control-allow-credentials" not in r.headers


def test_dangerous_methods_are_not_allowed(client):
    for method in ("DELETE", "PUT", "PATCH"):
        assert preflight(client, "http://localhost:8000", method).status_code == 400


def test_useful_response_headers_are_exposed_to_the_browser(client):
    r = client.get("/health", headers={"Origin": "http://localhost:8000"})
    exposed = r.headers["access-control-expose-headers"].lower()
    assert "x-request-id" in exposed and "retry-after" in exposed and "x-failure-kind" in exposed


def test_rejected_requests_still_carry_cors_headers_so_the_browser_can_read_them(client):
    r = client.post("/api/agent/turn", content=b"x" * (70 * 1024),
                    headers={"Origin": "http://localhost:8000", "Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.headers["access-control-allow-origin"] == "http://localhost:8000"
