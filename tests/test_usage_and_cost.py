"""``/usage`` from ``GET /v1/credits`` and per-request cost."""

from __future__ import annotations

import urllib.error
from decimal import Decimal

import pytest
from conftest import CanonicalUsage

CREDITS_URL = "https://api.llmapi.ai/v1/credits"


def credits(**overrides):
    payload = {
        "organization": {"id": "org_1", "name": "Acme"},
        "mode": "prepaid",
        "currency": "USD",
        "credits": 12.5,
        "credits_bonus": 0,
        "total_credits": 12.5,
        "gated_on_balance": True,
    }
    payload.update(overrides)
    return payload


def test_usage_shows_the_balance(profile, network, secrets):
    secrets["LLMAPI_API_KEY"] = "sk-test"
    network.routes[CREDITS_URL] = credits()
    snapshot = profile.fetch_account_usage()
    assert snapshot.provider == "llmapi"
    assert snapshot.source == "credits_api"
    assert snapshot.plan == "Acme"
    assert snapshot.details == ("Credits balance: $12.50",)
    assert snapshot.windows == ()
    (request,) = network.requests
    assert request["url"] == CREDITS_URL
    assert request["headers"]["authorization"] == "Bearer sk-test"
    assert request["timeout"] == 5.0


def test_usage_bonus_mode_and_monthly_cap(profile, network, secrets):
    secrets["LLMAPI_API_KEY"] = "sk-test"
    network.routes[CREDITS_URL] = credits(
        mode="payg",
        credits=10,
        credits_bonus=2.5,
        total_credits=12.5,
        monthly_cap={"limit": 200, "used": 50, "remaining": 150},
    )
    snapshot = profile.fetch_account_usage()
    assert snapshot.details == (
        "Credits balance: $12.50 (incl. $2.50 bonus)",
        "Billing: pay as you go, invoiced after use",
    )
    (window,) = snapshot.windows
    assert window.label == "Monthly cap"
    assert window.used_percent == 25.0
    assert window.detail == "$50.00 of $200.00 used"


def test_usage_prefers_the_key_hermes_passes(profile, network, secrets):
    secrets["LLMAPI_API_KEY"] = "sk-env"
    network.routes[CREDITS_URL] = credits()
    profile.fetch_account_usage(api_key="sk-passed")
    assert network.requests[0]["headers"]["authorization"] == "Bearer sk-passed"


@pytest.mark.parametrize(
    ("passed", "env", "expected"),
    [
        # The credential's own gateway; on the Anthropic wire Hermes passes it without /v1.
        ("https://eu.api.llmapi.ai", None, "https://eu.api.llmapi.ai/v1/credits"),
        ("https://eu.api.llmapi.ai/v1/", "https://api.dev.llmapi.ai/v1", "https://eu.api.llmapi.ai/v1/credits"),
        # `hermes usage` passes nothing: LLMAPI_BASE_URL, else the hosted gateway.
        (None, "https://api.dev.llmapi.ai/v1/", "https://api.dev.llmapi.ai/v1/credits"),
        (None, None, CREDITS_URL),
    ],
)
def test_usage_asks_the_credentials_gateway(profile, network, secrets, passed, env, expected):
    secrets["LLMAPI_API_KEY"] = "sk-test"
    if env:
        secrets["LLMAPI_BASE_URL"] = env
    network.routes[expected] = credits()
    assert profile.fetch_account_usage(base_url=passed) is not None
    assert network.requests[0]["url"] == expected


def test_usage_shows_a_negative_balance(profile, network, secrets):
    secrets["LLMAPI_API_KEY"] = "sk-test"
    network.routes[CREDITS_URL] = credits(mode="payg", credits=-42.1, total_credits=-42.1)
    snapshot = profile.fetch_account_usage()
    assert snapshot.details[0] == "Credits balance: -$42.10"


def test_usage_without_a_key_makes_no_request(profile, network):
    assert profile.fetch_account_usage() is None
    assert network.requests == []


@pytest.mark.parametrize(
    "answer",
    [urllib.error.HTTPError(CREDITS_URL, 401, "Unauthorized", {}, None), OSError("down"), ["not", "a", "dict"]],
)
def test_usage_fails_open(profile, network, secrets, answer):
    secrets["LLMAPI_API_KEY"] = "sk-test"
    network.routes[CREDITS_URL] = answer
    assert profile.fetch_account_usage() is None


def test_cost_reported_by_the_gateway_is_actual(warm, profile):
    usage = CanonicalUsage(
        input_tokens=1000,
        output_tokens=100,
        raw_usage={
            "prompt_tokens": 1000,
            "cost_usd_input": "0.002",
            "cost_usd_output": 0.001,
            "cost_usd_cached_input": 0,
        },
    )
    cost = profile.get_usage_cost("claude-sonnet-5-5", usage)
    assert cost.status == "actual"
    assert cost.source == "provider_cost_api"
    assert cost.amount_usd == Decimal("0.003")


def test_cost_estimated_from_catalog_prices(warm, profile):
    # claude-sonnet-5-5: prompt 0.000002, completion 0.00001, cache read 0.0000001, cache write 0.0000025.
    usage = CanonicalUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=10_000, cache_write_tokens=2000)
    cost = profile.get_usage_cost("claude-sonnet-5-5", usage)
    assert cost.status == "estimated"
    assert cost.source == "provider_models_api"
    assert cost.amount_usd == Decimal("0.002") + Decimal("0.002") + Decimal("0.001") + Decimal("0.005")


def test_cache_write_falls_back_to_the_prompt_price(warm, profile):
    # gpt-5.5 publishes no cache-write price.
    usage = CanonicalUsage(input_tokens=0, output_tokens=0, cache_write_tokens=1000)
    assert profile.get_usage_cost("gpt-5.5", usage).amount_usd == Decimal("0.005")


def test_cost_unknown_model_defers(warm, profile):
    assert profile.get_usage_cost("not-a-model", CanonicalUsage(input_tokens=10)) is None


def test_the_key_never_follows_a_redirect(plugin):
    """A redirect must not carry Authorization to another host: the plugin's opener refuses it."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = []

    class Elsewhere(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    other = ThreadingHTTPServer(("127.0.0.1", 0), Elsewhere)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{other.server_address[1]}/steal")
            self.end_headers()

        def log_message(self, *args):
            pass

    first = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    for server in (first, other):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    opener = plugin.urllib.request.build_opener(plugin._NoRedirect)
    plugin._OPENER = opener
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            plugin._get_json(f"http://127.0.0.1:{first.server_address[1]}/v1/credits", timeout=5, api_key="sk-x")
        assert caught.value.code == 302
        assert hits == []
    finally:
        for server in (first, other):
            server.shutdown()
            server.server_close()
