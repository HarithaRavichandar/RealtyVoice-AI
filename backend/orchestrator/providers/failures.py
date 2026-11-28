# -*- coding: utf-8 -*-
"""One vocabulary for external-service failures (Groq, Sarvam, Mappls).

Every provider exception carries a ``failure_kind`` so callers can tell the
cases apart instead of treating all of them as "service unavailable":

    no_results       the call worked and found nothing (not an error)
    auth_failure     credentials rejected: our configuration problem
    quota_exceeded   plan/credit limit reached: retrying will not help
    rate_limited     too many requests right now: retry later
    timeout          no answer in time
    network_error    could not reach the service
    invalid_response the service answered with something unusable
    provider_error   any other error status from the service
"""

from enum import Enum
from typing import Optional


class FailureKind(str, Enum):
    NO_RESULTS = "no_results"
    AUTH_FAILURE = "auth_failure"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_ERROR = "provider_error"


_QUOTA_WORDS = ("quota", "billing", "credit", "insufficient", "exceeded your", "plan limit")


def is_quota_response(status_code: int, body_text: Optional[str]) -> bool:
    """True for HTTP 402, or a 429 whose body says a quota/credit ran out."""
    if status_code == 402:
        return True
    if status_code == 429 and body_text:
        text = body_text.lower()
        return any(word in text for word in _QUOTA_WORDS)
    return False


def classify(exc: BaseException) -> Optional[FailureKind]:
    """The failure kind of a provider exception, or ``None`` if it has none."""
    kind = getattr(exc, "failure_kind", None)
    return kind if isinstance(kind, FailureKind) else None


# What may be said to a customer. Never includes provider names, keys or
# raw error bodies, and never claims more than is known.
CUSTOMER_MESSAGES = {
    FailureKind.AUTH_FAILURE: "A service we depend on is not available right now. Please try again later.",
    FailureKind.QUOTA_EXCEEDED: "A service we depend on has reached its usage limit. Please try again later.",
    FailureKind.RATE_LIMITED: "We are receiving a lot of requests. Please wait a moment and try again.",
    FailureKind.TIMEOUT: "The service took too long to respond. Please try again.",
    FailureKind.NETWORK_ERROR: "We could not reach a service we depend on. Please try again shortly.",
    FailureKind.INVALID_RESPONSE: "A service we depend on returned something we could not use. Please try again.",
    FailureKind.PROVIDER_ERROR: "A service we depend on reported a problem. Please try again later.",
}

# HTTP status used when a failure ends a request.
HTTP_STATUS = {
    FailureKind.AUTH_FAILURE: 502,
    FailureKind.QUOTA_EXCEEDED: 503,
    FailureKind.RATE_LIMITED: 429,
    FailureKind.TIMEOUT: 504,
    FailureKind.NETWORK_ERROR: 502,
    FailureKind.INVALID_RESPONSE: 502,
    FailureKind.PROVIDER_ERROR: 502,
}
