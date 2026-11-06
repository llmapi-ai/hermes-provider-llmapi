"""Profile fields, picker, wire routing, context windows and reasoning."""

from __future__ import annotations

import pytest
from conftest import CATALOG, EXPECTED_PICKER


def test_registers_one_profile(profile, plugin):
    assert type(profile).__name__ == "LLMAPIProfile"
    assert profile.name == "llmapi"
    assert profile.aliases == ("llm-api",)
    assert profile.env_vars == ("LLMAPI_API_KEY", "LLMAPI_BASE_URL")
    assert profile.base_url == "https://api.llmapi.ai/v1"
    # No models_url: a non-default one makes Hermes accept only fallback_models in `/model`.
    assert profile.models_url == ""
    assert profile.api_mode == "chat_completions"
    assert profile.classify_api_error is plugin.classify_gateway_error
    assert profile.fallback_models and all(isinstance(m, str) for m in profile.fallback_models)


def test_unknown_profile_fields_are_dropped_for_older_hermes(plugin):
    kwargs = plugin._supported_kwargs(plugin.LLMAPIProfile, {"name": "x", "not_a_field": 1})
    assert kwargs == {"name": "x"}


def test_picker_lists_tool_capable_chat_models_with_enough_context(profile, network):
    network.routes["https://api.llmapi.ai/v1/models"] = CATALOG
    assert profile.fetch_models(api_key="sk-secret", base_url=profile.base_url) == EXPECTED_PICKER
    (request,) = network.requests
    assert request["url"] == "https://api.llmapi.ai/v1/models?exclude_deprecated=true"
    # The catalog is public: the key is never sent with it.
    assert "authorization" not in request["headers"]
    assert request["headers"]["user-agent"].startswith("hermes-provider-llmapi/")


def test_picker_is_cached(profile, network):
    network.routes["https://api.llmapi.ai/v1/models"] = CATALOG
    profile.fetch_models()
    profile.fetch_models()
    assert len(network.requests) == 1


def test_picker_follows_the_base_url_override(profile, network):
    network.routes["https://api.dev.llmapi.ai/v1/models"] = CATALOG
    assert profile.fetch_models(base_url="https://api.dev.llmapi.ai/v1") == EXPECTED_PICKER
    assert network.requests[0]["url"].startswith("https://api.dev.llmapi.ai/v1/models")


@pytest.mark.parametrize("answer", [OSError("down"), {"data": []}, {"oops": 1}, []])
def test_catalog_failure_falls_back(profile, network, answer):
    network.routes["https://api.llmapi.ai/v1/models"] = answer
    assert profile.fetch_models() is None


@pytest.mark.parametrize(
    ("model", "wire"),
    [
        ("claude-sonnet-5-5", "anthropic_messages"),
        ("claude-haiku-4-5", "anthropic_messages"),
        ("CLAUDE-OPUS-5-5", "anthropic_messages"),
        ("gpt-5", "codex_responses"),
        ("gpt-5.5", "codex_responses"),
        ("gpt-5.4-mini", "codex_responses"),
        ("gpt-6.1-sol", "codex_responses"),
        ("gpt-10", "codex_responses"),
        ("gpt-4.1", "chat_completions"),
        ("gpt-4o-mini", "chat_completions"),
        ("gpt-oss-120b", "chat_completions"),
        ("gpt-50x", "chat_completions"),
        ("kimi-k3", "chat_completions"),
        ("deepseek/deepseek-v4.1-flash", "chat_completions"),
        ("gemini-3.5-flash-lite", "chat_completions"),
        ("", "chat_completions"),
    ],
)
def test_wire_per_model_without_catalog(profile, model, wire):
    # The hot path answers from the id alone (api_key=None, cold cache).
    assert profile.api_mode_for_model(model) == wire


def test_wire_does_not_depend_on_the_catalog(plugin, profile):
    models = ["claude-sonnet-5-5", "gpt-5.5", "kimi-k3", "gpt-4.1"]
    cold = [profile.api_mode_for_model(m) for m in models]
    plugin.catalog.store(CATALOG["data"])
    assert [profile.api_mode_for_model(m) for m in models] == cold


def test_stale_catalog_beats_fallback_when_refresh_fails(plugin, profile, network, monkeypatch):
    plugin.catalog.store(CATALOG["data"])
    monkeypatch.setattr(plugin.catalog, "fresh", lambda: False)
    network.routes["https://api.llmapi.ai/v1/models"] = OSError("down")
    assert profile.fetch_models() == EXPECTED_PICKER
    # Within the back-off window the next open does not wait on the network again.
    assert profile.fetch_models() == EXPECTED_PICKER
    assert len(network.requests) == 1


def test_warm_up_uses_the_configured_gateway(plugin, network, secrets, monkeypatch):
    secrets["LLMAPI_BASE_URL"] = "https://eu.api.llmapi.ai/v1"
    started = []
    monkeypatch.setattr(plugin.threading, "Thread", lambda target, **kw: started.append(target) or _NoThread())
    network.routes["https://eu.api.llmapi.ai/v1/models"] = CATALOG
    monkeypatch.setattr(plugin.catalog, "warm", plugin._Catalog.warm.__get__(plugin.catalog))  # undo the no-op
    plugin.catalog.get("kimi-k3")
    (run,) = started
    run()
    assert network.requests[0]["url"].startswith("https://eu.api.llmapi.ai/v1/models")
    assert plugin.catalog.get("kimi-k3") is not None


class _NoThread:
    def start(self):
        pass


def test_live_context_length_loads_a_cold_catalog(profile, network):
    network.routes["https://api.dev.llmapi.ai/v1/models"] = CATALOG
    # Hermes passes the runtime base URL, which has no /v1 on the Anthropic wire.
    assert profile.live_context_length("claude-sonnet-5-5", base_url="https://api.dev.llmapi.ai") == 1_000_000
    assert "authorization" not in network.requests[0]["headers"]


def test_context_window_from_catalog(warm, profile):
    assert profile.get_model_context_length("claude-sonnet-5-5") == 1_000_000
    assert profile.get_model_context_length("qwen3-235b-a22b-fp8") == 40_960
    assert profile.get_model_context_length("not-a-model") is None


def test_context_window_is_none_while_cold(profile, network):
    assert profile.get_model_context_length("claude-sonnet-5-5") is None
    assert network.requests == []


def test_reasoning_efforts(warm, profile):
    assert profile.supported_reasoning_efforts("gpt-5.5") == ("none", "low", "medium", "high", "xhigh")
    assert profile.supported_reasoning_efforts("gpt-4.1") == ()
    assert profile.supported_reasoning_efforts("not-a-model") is None


@pytest.mark.parametrize(
    ("model", "config", "expected"),
    [
        ("kimi-k3", {"enabled": True, "effort": "high"}, {"reasoning_effort": "max"}),  # only level: the floor
        ("gpt-oss-120b", {"enabled": True, "effort": "xhigh"}, {"reasoning_effort": "high"}),
        ("gpt-oss-120b", {"enabled": True, "effort": "medium"}, {"reasoning_effort": "medium"}),
        ("gpt-oss-120b", {"enabled": True}, {"reasoning_effort": "medium"}),
        ("gpt-oss-120b", {"enabled": False}, {"reasoning_effort": "none"}),
        ("kimi-k3", {"enabled": False}, {}),  # cannot switch off: send nothing
        ("gpt-4.1", {"enabled": True, "effort": "high"}, {}),  # no reasoning
        ("gpt-oss-120b", None, {}),
        ("not-a-model", {"enabled": True, "effort": "high"}, {}),
    ],
)
def test_reasoning_effort_on_chat(warm, profile, model, config, expected):
    assert profile.build_api_kwargs_extras(reasoning_config=config, model=model) == ({}, expected)


@pytest.mark.parametrize(
    ("requested", "supported", "expected"),
    [
        ("high", ("low", "medium", "high"), "high"),
        ("xhigh", ("low", "medium", "high"), "high"),
        ("minimal", ("none", "low", "high"), "low"),  # never clamps an enabled ask to "none"
        ("low", ("high", "max"), "high"),
        ("turbo", ("low",), "turbo"),  # outside the ladder: pass through
    ],
)
def test_clamp_matches_hermes(plugin, requested, supported, expected):
    assert plugin.clamp_effort(requested, supported) == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.llmapi.ai/v1", True),
        ("https://eu.api.llmapi.ai/v1", True),
        ("https://api.dev.llmapi.ai", True),
        ("https://llmapi.ai.evil.com/v1", False),
        ("https://api.openai.com/v1", False),
        ("", False),
    ],
)
def test_attribution_headers_only_for_llmapi_hosts(profile, base_url, expected):
    headers = profile.request_headers(base_url=base_url, model="kimi-k3")
    assert (headers == {"X-Source": "hermes-agent", "X-LLMAPI-App-Id": "hermes-provider-llmapi"}) is expected
    assert expected or headers == {}


def test_capabilities_mirror_the_catalog(warm, profile):
    caps = profile.model_capabilities
    assert caps["claude-sonnet-5-5"] == {
        "supports_tools": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "context_window": 1_000_000,
        "model_family": "anthropic",
    }
    # Text-only and no-reasoning models: Hermes summarises images and skips the effort prompt.
    assert caps["gpt-oss-120b"]["supports_vision"] is False
    assert caps["gpt-4.1"]["supports_reasoning"] is False
    # Aliases resolve to the same entry; non-chat models are left to Hermes.
    assert caps["claude-haiku-4.5"] == caps["claude-haiku-4-5"]
    assert "azure/text-embedding-3-small" not in caps


def test_capabilities_survive_a_failed_refresh(warm, profile, network):
    warm.catalog.mark_stale()
    network.routes["https://api.llmapi.ai/v1/models"] = OSError("down")
    profile.fetch_models()
    assert "claude-sonnet-5-5" in profile.model_capabilities


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("claude-haiku-4.5", "claude-haiku-4-5"),  # catalog alias
        ("Claude-Haiku-4-5", "claude-haiku-4-5"),  # case
        ("deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
        ("claude-haiku-4-5", None),  # already canonical
        ("not-a-model", None),
    ],
)
def test_normalize_model_id(warm, profile, typed, expected):
    assert profile.normalize_model_id(typed) == expected


def test_alias_lookups_reach_the_catalog_entry(warm, profile):
    assert profile.get_model_context_length("claude-haiku-4.5") == 200_000


def test_refresh_keeps_the_list_and_refetches(warm, profile, network):
    network.routes["https://api.llmapi.ai/v1/models"] = CATALOG
    profile.invalidate_catalog_cache()
    assert profile.fetch_models() == EXPECTED_PICKER
    assert len(network.requests) == 1


def test_default_vision_model_reads_images_and_is_listed(warm, profile):
    model = profile.default_vision_model()
    assert model == profile.default_aux_model
    assert model in EXPECTED_PICKER
    assert profile.model_capabilities[model]["supports_vision"] is True


def test_setup_reasoning_efforts(warm, profile):
    assert profile.setup_reasoning_efforts("gpt-5.5") == ("none", "low", "medium", "high", "xhigh")
    assert profile.setup_reasoning_efforts("gpt-4.1") == ()
    assert profile.setup_reasoning_efforts("not-a-model") is None
