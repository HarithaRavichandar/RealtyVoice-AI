# -*- coding: utf-8 -*-
"""Structured (JSON) logging with a per-request correlation id.

One JSON object per line, so logs can be searched and aggregated:

    {"timestamp": "...", "level": "INFO", "logger": "backend.access",
     "message": "request", "request_id": "ab12...", "method": "POST", ...}

Message bodies, phone numbers and other customer data must not be logged;
the access log records only method, route, status and timing.
"""

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Attributes every LogRecord has; anything else was passed via ``extra=``.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class RequestIdFilter(logging.Filter):
    """Adds the current request id to text-format records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def build_logging_config(level: str, fmt: str) -> Dict[str, Any]:
    formatter = (
        {"()": "backend.logging_config.JsonFormatter"}
        if fmt == "json"
        else {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": "backend.logging_config.RequestIdFilter"}},
        "formatters": {"default": formatter},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "filters": ["request_id"],
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": level, "handlers": ["console"]},
        "loggers": {
            "uvicorn": {"level": level, "handlers": ["console"], "propagate": False},
            "uvicorn.access": {"level": "WARNING", "handlers": ["console"], "propagate": False},
            "sqlalchemy.engine": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        },
    }
