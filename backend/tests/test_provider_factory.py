
import pytest

from backend.orchestrator.providers.factory import (
    ProviderConfigurationError,
    create_llm_provider,
)



def test_create_groq_provider(monkeypatch):
    monkeypatch.setattr(
        "backend.orchestrator.providers.factory.LLM_PROVIDER",
        "groq",
    )
    monkeypatch.setattr(
        "backend.orchestrator.providers.factory.GroqProvider",
        lambda: "groq-provider",
    )

    provider = create_llm_provider()

    assert provider == "groq-provider"


def test_unsupported_provider_raises_error(monkeypatch):
    monkeypatch.setattr(
        "backend.orchestrator.providers.factory.LLM_PROVIDER",
        "unsupported",
    )

    with pytest.raises(ProviderConfigurationError):
        create_llm_provider()
