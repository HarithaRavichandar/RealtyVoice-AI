
"""Factory for configured LLM providers."""

from backend.config import LLM_PROVIDER
from .base import BaseLLMProvider
from .groq_provider import GroqProvider


class ProviderConfigurationError(RuntimeError):
    """Raised when the configured LLM provider is unsupported."""


def create_llm_provider() -> BaseLLMProvider:
    """Create the provider selected by LLM_PROVIDER."""

    if LLM_PROVIDER == "groq":
        return GroqProvider()

    raise ProviderConfigurationError(
        f"Unsupported LLM provider: {LLM_PROVIDER!r}"
    )


__all__ = [
    "ProviderConfigurationError",
    "create_llm_provider",
]
