"""End-to-end against a real Hermes checkout. Skipped when Hermes is not importable.

Run with the checkout's interpreter::

    PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/.venv/bin/python -m pytest tests

No real network: each turn talks to one of Hermes' own recording fake servers
(Anthropic Messages, OpenAI Responses, Chat Completions) and every other
outbound HTTP(S) request goes to a dead proxy. The base URL in every turn has
the same shape as the hosted one (``<host>/v1``), so the tests also prove that
one base URL serves all three wires.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PLUGIN_DIR

try:
    import hermes_cli

    HERMES_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
except ImportError:  # unit-only environments
    HERMES_ROOT = None

pytestmark = pytest.mark.skipif(HERMES_ROOT is None, reason="Hermes is not importable")

API_KEY = "sk-llmapi-e2e-not-a-real-key"
ATTRIBUTION = {"x-source": "hermes-agent", "x-llmapi-app-id": "hermes-provider-llmapi"}


def _fake(relative: str):
    """Import one of Hermes' fake providers from its checkout by path."""
    path = HERMES_ROOT / relative
    if not path.is_file():
        pytest.skip(f"this Hermes checkout has no {relative}")
    name = "hermes_fake_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "hermes-home"
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "llmapi", ignore=shutil.ignore_patterns("__pycache__"))
    (home / ".env").write_text(f"LLMAPI_API_KEY={API_KEY}\n", encoding="utf-8")
    yield home
    _stop_background_hermes(home)


def _env(home: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OPENAI", "ANTHROPIC", "LLMAPI", "OPENROUTER", "HERMES_"))
        and k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    }
    dead_proxy = "http://127.0.0.1:9"  # discard port: any stray outbound request fails fast
    env.update(
        HERMES_HOME=str(home),
        PYTHONPATH=str(HERMES_ROOT),
        NO_COLOR="1",
        HTTP_PROXY=dead_proxy,
        HTTPS_PROXY=dead_proxy,
        ALL_PROXY=dead_proxy,
        NO_PROXY="127.0.0.1,localhost",
    )
    return env


def _python(home: Path, code: str):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_env(home),
        cwd=str(home.parent),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _turn(home: Path, model: str, base_url: str) -> str:
    """One ``hermes chat`` turn; ``LLMAPI_BASE_URL`` points the provider at the fake server."""
    (home / "config.yaml").write_text(
        f"model:\n  provider: llmapi\n  default: {model}\n  context_length: 1000000\nagent:\n  api_max_retries: 1\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Reply with: pong", "--oneshot", "-Q", "-t", "todo"],
        env={**_env(home), "LLMAPI_BASE_URL": base_url},
        cwd=str(home.parent),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout[-1500:], proc.stderr[-3000:])
    return proc.stdout


def _stop_background_hermes(home: Path) -> None:
    """Kill processes the turn left behind (Hermes starts an idle-exit gateway) for this home only."""
    marker = f"HERMES_HOME={home}".encode()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            if marker in (proc_dir / "environ").read_bytes().split(b"\0"):
                os.kill(int(proc_dir.name), signal.SIGTERM)
        except (OSError, ValueError):
            continue


def test_hermes_discovers_the_provider(home):
    found = _python(
        home,
        """
import json
from providers import get_provider_profile
from hermes_cli.models import list_available_providers
p = get_provider_profile("llmapi")
listed = {row["id"]: row["authenticated"] for row in list_available_providers()}
print(json.dumps({
    "cls": type(p).__name__, "base_url": p.base_url, "api_mode": p.api_mode,
    "alias": get_provider_profile("llm-api") is p,
    "authenticated": listed.get("llmapi"),
    "wires": {m: p.api_mode_for_model(m) for m in ("claude-sonnet-5-5", "gpt-5.5", "kimi-k3")},
}))
""",
    )
    assert found == {
        "cls": "LLMAPIProfile",
        "base_url": "https://api.llmapi.ai/v1",
        "api_mode": "chat_completions",
        "alias": True,
        "authenticated": True,  # LLMAPI_API_KEY from $HERMES_HOME/.env resolves
        "wires": {
            "claude-sonnet-5-5": "anthropic_messages",
            "gpt-5.5": "codex_responses",
            "kimi-k3": "chat_completions",
        },
    }


def test_claude_turn_uses_messages_with_prompt_caching(home):
    fake = _fake("tests/fakes/providers/anthropic_messages.py")
    with fake.AnthropicMessagesServer() as server:
        # The hosted shape, <host>/v1: Hermes strips /v1 and the Anthropic SDK appends /v1/messages.
        port = server.base_url.split(":")[2].split("/")[0]
        _turn(home, "claude-sonnet-5-5", f"http://127.0.0.1:{port}/v1")
        mains = server.main_requests()
    assert mains, "the turn made no Messages request"
    first = mains[0]
    # Hermes strips /v1 from the base URL and the Anthropic SDK appends /v1/messages.
    assert first["path"].split("?", 1)[0] == "/v1/messages"
    assert first["body"]["model"] == "claude-sonnet-5-5"
    assert first["x_api_key"] == API_KEY
    assert "authorization" not in first["headers"]
    # Hermes marks cache breakpoints for Claude on a third-party Anthropic gateway.
    assert '"cache_control"' in json.dumps(first["body"])


def test_gpt5_turn_uses_responses(home):
    fake = _fake("tests/fakes/providers/openai_responses.py")
    with fake.FakeResponsesServer(api_key=API_KEY) as server:
        _turn(home, "gpt-5.5", server.base_url)
        records = [r for r in server.requests if r["kind"] == "main"]
        invalid = server.invalid_requests()
    assert records, "the turn made no Responses request"
    first = records[0]
    assert first["path"].split("?", 1)[0] == "/v1/responses"
    assert first["body"]["model"] == "gpt-5.5"
    assert first["body"].get("tools"), "Hermes sent no tool schemas"
    assert first["headers"]["authorization"] == f"Bearer {API_KEY}"
    assert invalid == []


def test_other_models_use_chat_completions_with_tools(home):
    fake = _fake("tests/fakes/fake_llm_provider.py")
    with fake.FakeLLMServer([fake.Text("pong")], api_key=API_KEY) as server:
        out = _turn(home, "kimi-k3", server.base_url)
        mains = [r for r in server.requests if r["kind"] == "main"]
    assert "pong" in out
    assert mains, "the turn made no chat request"
    first = mains[0]
    assert first["body"]["model"] == "kimi-k3"
    assert first["body"].get("tools"), "Hermes sent no tool schemas"
    assert first["auth"] == f"Bearer {API_KEY}"


def test_attribution_headers_on_the_openai_wires(home):
    found = _python(
        home,
        """
import json
from providers import get_provider_profile
p = get_provider_profile("llmapi")
print(json.dumps({u: p.request_headers(base_url=u, model="kimi-k3")
                  for u in ("https://api.llmapi.ai/v1", "http://127.0.0.1:1/v1")}))
""",
    )
    assert {k.lower(): v for k, v in found["https://api.llmapi.ai/v1"].items()} == ATTRIBUTION
    assert found["http://127.0.0.1:1/v1"] == {}


@pytest.mark.parametrize(
    ("status", "code", "reason", "fallback"),
    [
        (402, "insufficient_credits", "billing", True),
        (403, "provider_blocked", "model_entitlement", True),
        (403, "rejected_not_in_plan", "model_entitlement", True),
        (429, "rate_limit_exceeded", "rate_limit", False),
        (429, "usage_limit_exceeded", "billing", True),
    ],
)
@pytest.mark.parametrize("sdk", ["openai", "anthropic"])
def test_hermes_classifies_gateway_errors(home, sdk, status, code, reason, fallback):
    found = _python(
        home,
        f"""
import json, httpx, {sdk}
from agent.error_classifier import classify_api_error
body = {{"error": {{"message": "gateway says no", "type": "x", "param": None, "code": "{code}"}}}}
req = httpx.Request("POST", "https://api.llmapi.ai/v1/chat/completions")
resp = httpx.Response({status}, json=body, request=req)
err = {sdk}.APIStatusError("gateway says no", response=resp, body=body)
c = classify_api_error(err, provider="llmapi", model="kimi-k3", base_url="https://api.llmapi.ai/v1")
print(json.dumps({{"reason": c.reason.value, "fallback": c.should_fallback, "rotate": c.should_rotate_credential}}))
""",
    )
    assert found == {"reason": reason, "fallback": fallback, "rotate": False}
