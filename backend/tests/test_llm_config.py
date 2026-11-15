
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_config_import(extra_env=None):
    env = os.environ.copy()

    for key in (
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "GROQ_BASE_URL",
        "GROQ_TIMEOUT_SECONDS",
        "AGENT_MAX_TOOL_ITERATIONS",
    ):
        env.pop(key, None)

    env["PYTHON_DOTENV_DISABLED"] = "1"

    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; import backend.config as c; "
                "print(json.dumps({"
                "'provider': c.LLM_PROVIDER, "
                "'key': c.GROQ_API_KEY, "
                "'model': c.GROQ_MODEL, "
                "'url': c.GROQ_BASE_URL, "
                "'timeout': c.GROQ_TIMEOUT_SECONDS, "
                "'iterations': c.AGENT_MAX_TOOL_ITERATIONS"
                "}))"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_llm_config_defaults():
    result = run_config_import()

    assert result.returncode == 0, result.stderr
    assert '"provider": "groq"' in result.stdout
    assert '"key": ""' in result.stdout
    assert '"model": "openai/gpt-oss-20b"' in result.stdout
    assert '"timeout": 30' in result.stdout
    assert '"iterations": 4' in result.stdout


def test_llm_config_environment_overrides():
    result = run_config_import({
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "test-key",
        "GROQ_MODEL": "test-model",
        "GROQ_BASE_URL": "https://example.com/v1/",
        "GROQ_TIMEOUT_SECONDS": "45",
        "AGENT_MAX_TOOL_ITERATIONS": "6",
    })

    assert result.returncode == 0, result.stderr
    assert '"key": "test-key"' in result.stdout
    assert '"model": "test-model"' in result.stdout
    assert '"url": "https://example.com/v1"' in result.stdout
    assert '"timeout": 45' in result.stdout
    assert '"iterations": 6' in result.stdout


@pytest.mark.parametrize(
    "variable,value",
    [
        ("GROQ_TIMEOUT_SECONDS", "invalid"),
        ("AGENT_MAX_TOOL_ITERATIONS", "invalid"),
        ("GROQ_TIMEOUT_SECONDS", "0"),
        ("AGENT_MAX_TOOL_ITERATIONS", "-1"),
    ],
)
def test_invalid_numeric_config_is_rejected(variable, value):
    result = run_config_import({variable: value})

    assert result.returncode != 0
    assert "ValueError" in result.stderr
