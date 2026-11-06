"""Load the plugin against stubbed Hermes modules, so unit tests run without Hermes.

The plugin imports ``providers`` / ``providers.base`` at import time and
``agent.*`` lazily inside hooks. When Hermes itself is importable (the
end-to-end job), the unit tests still use these stubs; the real Hermes is
exercised only by ``test_hermes_integration.py`` in a subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "llmapi"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Trimmed snapshot of the public https://api.llmapi.ai/v1/models?exclude_deprecated=true
# response (2026-11-06): real entries chosen for their edge cases (o3 has no tools,
# qwen3-235b-a22b-fp8 has a 40K window, embedding/image/TTS models, the auto/custom
# pseudo models without a kind).
CATALOG = json.loads((FIXTURES / "models.json").read_text(encoding="utf-8"))

EXPECTED_PICKER = [
    "claude-haiku-4-5",
    "claude-sonnet-5-5",
    "deepseek/deepseek-v4.1-flash",
    "gemini-3.5-flash-lite",
    "gpt-4.1",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-6.1-sol",
    "gpt-oss-120b",
    "kimi-k3",
]


@dataclass
class StubProviderProfile:
    """The ``providers.base.ProviderProfile`` fields this plugin sets."""

    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    hostname: str = ""
    auth_type: str = "api_key"
    supports_vision: bool = False
    fallback_models: tuple = ()
    default_aux_model: str = ""
    classify_api_error: Any = None
    default_headers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: float | None = None
    reset_at: datetime | None = None
    detail: str | None = None
    scope: str = "account"


@dataclass(frozen=True)
class AccountUsageSnapshot:
    provider: str
    source: str
    fetched_at: datetime
    title: str = "Account limits"
    plan: str | None = None
    windows: tuple = ()
    details: tuple = ()
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class CostResult:
    amount_usd: Decimal | None
    status: str
    source: str
    label: str
    notes: tuple = ()


@dataclass(frozen=True)
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1
    raw_usage: dict | None = None


class FakeResponse:
    def __init__(self, payload: Any):
        self._data = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeNetwork:
    """Stands in for ``urllib.request.urlopen``; records every request."""

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.routes: dict[str, Any] = {}

    def __call__(self, request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        self.requests.append({"url": request.full_url, "headers": headers, "timeout": timeout})
        for prefix, answer in self.routes.items():
            if request.full_url.startswith(prefix):
                if isinstance(answer, BaseException):
                    raise answer
                return FakeResponse(answer)
        raise OSError(f"unexpected request to {request.full_url}")


def _install_stubs(monkeypatch, secrets: dict[str, str]) -> list:
    registered: list = []
    modules = {
        "providers": types.ModuleType("providers"),
        "providers.base": types.ModuleType("providers.base"),
        "agent": types.ModuleType("agent"),
        "agent.account_usage": types.ModuleType("agent.account_usage"),
        "agent.usage_pricing": types.ModuleType("agent.usage_pricing"),
        "agent.secret_scope": types.ModuleType("agent.secret_scope"),
    }
    modules["providers"].register_provider = registered.append
    modules["providers.base"].ProviderProfile = StubProviderProfile
    modules["agent.account_usage"].AccountUsageSnapshot = AccountUsageSnapshot
    modules["agent.account_usage"].AccountUsageWindow = AccountUsageWindow
    modules["agent.usage_pricing"].CostResult = CostResult
    modules["agent.secret_scope"].get_secret = lambda name, default=None: secrets.get(name, default)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return registered


@pytest.fixture
def secrets() -> dict[str, str]:
    return {}


@pytest.fixture
def network(monkeypatch) -> FakeNetwork:
    fake = FakeNetwork()
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return fake


@pytest.fixture
def plugin(monkeypatch, secrets, network):
    """The plugin module, freshly imported against the stubs, with an empty catalog."""
    registered = _install_stubs(monkeypatch, secrets)
    spec = importlib.util.spec_from_file_location("llmapi_plugin_under_test", PLUGIN_DIR / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    module.registered = registered
    # A cold catalog warms in a background thread; tests drive the cache explicitly.
    monkeypatch.setattr(module.catalog, "warm", lambda: None)
    return module


@pytest.fixture
def profile(plugin):
    assert len(plugin.registered) == 1
    return plugin.registered[0]


@pytest.fixture
def warm(plugin):
    """Catalog loaded from the fixture."""
    plugin.catalog.store(CATALOG["data"])
    return plugin
