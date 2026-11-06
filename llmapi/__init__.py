"""LLM API model provider for Hermes Agent.

LLM API (https://llmapi.ai) is one gateway with one key for Claude, GPT, Gemini,
Kimi, DeepSeek, GLM, Qwen and more. This profile routes each model to the wire
it works best on, all against ``https://api.llmapi.ai/v1``:

- Claude models use the Anthropic Messages API (``/v1/messages``). On that wire
  Hermes adds Anthropic prompt-cache breakpoints, so a tool loop hits the cache
  from its second turn.
- GPT-5 and later OpenAI models use the Responses API (``/v1/responses``).
- Everything else uses Chat Completions (``/v1/chat/completions``).

The picker, context windows, reasoning levels and list prices come from the
public catalog at ``GET /v1/models``, fetched without the API key and cached in
memory. ``/usage`` reads ``GET /v1/credits`` with the key. The plugin reads no
files, writes nothing, starts no processes and reports nothing anywhere.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

PROVIDER_ID = "llmapi"
API_KEY_ENV = "LLMAPI_API_KEY"
BASE_URL_ENV = "LLMAPI_BASE_URL"
BASE_URL = "https://api.llmapi.ai/v1"
SIGNUP_URL = "https://app.llmapi.ai/dashboard/api-keys"
USER_AGENT = f"hermes-provider-llmapi/{__version__}"
# Cheap, 1M-context model that reads images: auxiliary tasks and image description.
AUX_MODEL = "gemini-3.5-flash-lite"

# Request attribution. Hermes adds ``request_headers`` to the first request of
# each user turn, on all three wires.
ATTRIBUTION_HEADERS = {"X-Source": "hermes-agent", "X-LLMAPI-App-Id": "hermes-provider-llmapi"}

MIN_CONTEXT = 64_000  # Hermes refuses to start a session below this
CATALOG_TTL_SECONDS = 3600.0
FAILED_FETCH_RETRY_SECONDS = 300.0
USAGE_TIMEOUT_SECONDS = 5.0

ANTHROPIC_MESSAGES = "anthropic_messages"
RESPONSES = "codex_responses"
CHAT = "chat_completions"

# GPT-5 and later (gpt-5, gpt-5.5, gpt-6.1-sol, ...); not gpt-4.x or gpt-oss.
_RESPONSES_MODEL = re.compile(r"^gpt-([5-9]|[1-9]\d)(\b|[.-])")

# Hermes' effort ladder, weakest first. Unknown levels are passed through.
EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def _base_name(model: str | None) -> str:
    """``deepseek/deepseek-v4.1-flash`` -> ``deepseek-v4.1-flash``, lower-cased."""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _decimal(value: Any, *, signed: bool = False) -> Decimal | None:
    """A finite decimal from a JSON number or string; negatives only when *signed*."""
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and (signed or result >= 0) else None


# ── Catalog ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """The ``/v1/models`` fields the plugin uses, for one model."""

    id: str
    kind: str
    family: str
    aliases: tuple[str, ...]
    context_length: int
    tools: bool
    vision: bool
    reasoning_levels: tuple[str, ...]
    prices: dict[str, Decimal]  # USD per token: prompt, completion, input_cache_read, input_cache_write


def parse_model(entry: Any) -> ModelInfo | None:
    """One catalog entry, or None when it is not a usable object."""
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
        return None
    providers = [p for p in entry.get("providers") or () if isinstance(p, dict)]
    context = entry.get("context_length")
    architecture = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    prices = {
        key: price
        for key in ("prompt", "completion", "input_cache_read", "input_cache_write")
        if (price := _decimal(pricing.get(key))) is not None
    }
    return ModelInfo(
        id=entry["id"],
        kind=str(entry.get("kind") or ""),
        family=str(entry.get("family") or ""),
        aliases=tuple(a for a in entry.get("aliases") or () if isinstance(a, str) and a),
        context_length=context if type(context) is int and context > 0 else 0,
        tools=any(p.get("tools") is True for p in providers),
        vision="image" in (architecture.get("input_modalities") or ()),
        reasoning_levels=tuple(level for level in entry.get("reasoning_levels") or () if isinstance(level, str)),
        prices=prices,
    )


def listed_in_picker(info: ModelInfo) -> bool:
    """A chat model that calls tools and has a context window Hermes can run."""
    return info.kind == "chat" and info.tools and info.context_length >= MIN_CONTEXT


def capabilities(info: ModelInfo) -> dict[str, Any]:
    """Hermes' canonical ``model_capabilities`` entry: drives image routing, badges, effort prompts."""
    entry: dict[str, Any] = {
        "supports_tools": info.tools,
        "supports_vision": info.vision,
        "supports_reasoning": bool(info.reasoning_levels),
    }
    if info.context_length:
        entry["context_window"] = info.context_length
    if info.family:
        entry["model_family"] = info.family
    return entry


class _Catalog:
    """Last good ``/v1/models`` answer, shared by every hook in the process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelInfo] = {}
        self._index: dict[str, str] = {}  # lower-cased id or alias -> id
        self._order: tuple[str, ...] = ()
        self._listeners: list[Any] = []
        self._fetched_at = 0.0
        self._failed_at: float | None = None
        self._warming = False

    def store(self, entries: list) -> None:
        models: dict[str, ModelInfo] = {}
        for entry in entries:
            info = parse_model(entry)
            if info is not None and info.id not in models:
                models[info.id] = info
        index: dict[str, str] = {}
        for info in models.values():
            for name in info.aliases:
                index.setdefault(name.lower(), info.id)
        index.update({model_id.lower(): model_id for model_id in models})  # ids win over aliases
        with self._lock:
            self._models = models
            self._index = index
            self._order = tuple(models)
            self._fetched_at = time.monotonic()
            self._failed_at = None
            listeners = list(self._listeners)
        for listener in listeners:
            listener(models, index)

    def subscribe(self, listener: Any) -> None:
        """Call ``listener(models, index)`` after every successful store."""
        with self._lock:
            self._listeners.append(listener)

    def mark_stale(self) -> None:
        """Refetch on next use, keeping the last good list meanwhile (``/model --refresh``)."""
        with self._lock:
            self._fetched_at = 0.0
            self._failed_at = None

    def mark_failed(self) -> None:
        with self._lock:
            self._failed_at = time.monotonic()

    def fresh(self) -> bool:
        with self._lock:
            return bool(self._models) and time.monotonic() - self._fetched_at < CATALOG_TTL_SECONDS

    def recently_failed(self) -> bool:
        with self._lock:
            return self._failed_at is not None and time.monotonic() - self._failed_at < FAILED_FETCH_RETRY_SECONDS

    def has_models(self) -> bool:
        with self._lock:
            return bool(self._models)

    def picker(self) -> list[str]:
        with self._lock:
            return [m for m in self._order if listed_in_picker(self._models[m])]

    def get(self, model: str | None) -> ModelInfo | None:
        """Cached entry for *model* (id or alias, any case). Never blocks: a cold cache warms in the background."""
        canonical = self.canonical(model)
        with self._lock:
            return self._models.get(canonical) if canonical else None

    def canonical(self, model: str | None) -> str | None:
        """Catalog id for an id or alias in any case, or None."""
        with self._lock:
            models, index = self._models, self._index
        if not models:
            self.warm()
            return None
        name = (model or "").strip()
        return name if name in models else index.get(name.lower())

    def warm(self) -> None:
        with self._lock:
            if (
                self._warming
                or self._models
                or (self._failed_at is not None and time.monotonic() - self._failed_at < FAILED_FETCH_RETRY_SECONDS)
            ):
                return
            self._warming = True
        # Resolved here: the secret scope that holds LLMAPI_BASE_URL is bound to this thread.
        base_url = configured_base_url()

        def run() -> None:
            try:
                refresh_catalog(base_url, timeout=8.0)
            finally:
                with self._lock:
                    self._warming = False

        threading.Thread(target=run, name="llmapi-catalog-warm", daemon=True).start()

    def clear(self) -> None:
        with self._lock:
            self._models = {}
            self._index = {}
            self._order = ()
            self._fetched_at = 0.0
            self._failed_at = None


catalog = _Catalog()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies ``Authorization`` to a redirect target on any host; refuse to follow."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 - urllib signature
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _get_json(url: str, *, timeout: float, api_key: str | None = None) -> Any:
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", USER_AGENT)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def configured_base_url() -> str:
    """``LLMAPI_BASE_URL`` when set, else the hosted gateway."""
    try:
        from agent.secret_scope import get_secret

        return (get_secret(BASE_URL_ENV) or "").strip() or BASE_URL
    except Exception:  # noqa: BLE001 - no secret scope outside an agent session
        return BASE_URL


def api_base(base_url: str | None) -> str:
    """*base_url* with the ``/v1`` that Hermes strips on the Anthropic wire put back."""
    base = (base_url or "").strip().rstrip("/") or configured_base_url().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def ensure_catalog(base_url: str | None, *, timeout: float) -> bool:
    """A fresh catalog, refreshing at most once per back-off window. True when one is cached."""
    if catalog.fresh():
        return True
    if not catalog.recently_failed():
        refresh_catalog(api_base(base_url), timeout=timeout)
    return catalog.has_models()


def refresh_catalog(base_url: str, *, timeout: float) -> bool:
    """GET the public catalog (no API key) and cache it. False on any failure."""
    url = f"{base_url.rstrip('/')}/models?exclude_deprecated=true"
    try:
        payload = _get_json(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a catalog failure must never break Hermes
        logger.debug("llmapi: GET %s failed: %s", url, exc)
        catalog.mark_failed()
        return False
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        catalog.mark_failed()
        return False
    catalog.store(entries)
    return True


# ── Wire, reasoning and cost ─────────────────────────────────────────────


def api_mode_for(model: str | None) -> str:
    """Wire for *model*: Claude -> Messages, GPT-5+ -> Responses, else Chat Completions.

    Decided from the id alone, so the answer never depends on whether the catalog
    has loaded. The gateway converts a responses-only model on Chat Completions.
    """
    name = _base_name(model)
    if name.startswith("claude-"):
        return ANTHROPIC_MESSAGES
    if _RESPONSES_MODEL.match(name):
        return RESPONSES
    return CHAT


def clamp_effort(requested: str, supported: tuple[str, ...]) -> str:
    """Hermes' clamp: the nearest weaker supported level, else the weakest one.

    ``none`` is never a target for an enabled request, and a level outside the
    ladder passes through unchanged.
    """
    if requested in supported or requested not in EFFORT_LADDER:
        return requested
    candidates = [level for level in supported if level in EFFORT_LADDER and level != "none"]
    if not candidates:
        return requested
    rank = EFFORT_LADDER.index
    below = [level for level in candidates if rank(level) < rank(requested)]
    return max(below, key=rank) if below else min(candidates, key=rank)


def wire_effort(model: str | None, reasoning_config: dict | None) -> str | None:
    """``reasoning_effort`` for a Chat Completions request, or None to send nothing."""
    if not reasoning_config:
        return None
    info = catalog.get(model)
    if info is None or not info.reasoning_levels:
        return None
    if reasoning_config.get("enabled", True) is False:
        return "none" if "none" in info.reasoning_levels else None
    effort = str(reasoning_config.get("effort") or "medium").strip().lower()
    return clamp_effort(effort, info.reasoning_levels)


_COST_FIELDS = (
    "cost_usd_input",
    "cost_usd_output",
    "cost_usd_cached_input",
    "cost_usd_audio_input",
    "cost_usd_request",
    "cost_usd_reasoning",
    "cost_usd_web_search",
    "cost_usd_web_fetch",
)


def reported_cost(raw_usage: Any) -> Decimal | None:
    """Billed cost the gateway put in ``usage`` (orgs with cost-in-response on), or None."""
    if not isinstance(raw_usage, dict):
        return None
    parts = [_decimal(raw_usage.get(key)) for key in _COST_FIELDS if key in raw_usage]
    parts = [p for p in parts if p is not None]
    return sum(parts, Decimal(0)) if parts else None


def estimated_cost(info: ModelInfo, usage: Any) -> Decimal | None:
    """List-price cost from the catalog; cache reads and writes priced separately."""
    prompt = info.prices.get("prompt")
    completion = info.prices.get("completion")
    if prompt is None or completion is None:
        return None
    uncached = int(getattr(usage, "input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
    output = int(getattr(usage, "output_tokens", 0) or 0)
    return (
        uncached * prompt
        + cache_read * info.prices.get("input_cache_read", prompt)
        + cache_write * info.prices.get("input_cache_write", prompt)
        + output * completion
    )


# ── Errors ───────────────────────────────────────────────────────────────
#
# The gateway answers errors in an OpenAI-shaped envelope on every wire:
#   {"error": {"message": "...", "type": "...", "code": "<code>", "param": ...}}
# Hermes extracts ``error.code``. Codes not listed defer to Hermes' built-in
# classifier: 401 invalid key, 404 model_not_found, 413, 5xx upstream errors,
# and 429 rate_limit_exceeded (backoff with Retry-After, then rotate/fall back).

_BILLING = {"reason": "billing", "retryable": False, "should_fallback": True}
# The key is fine but cannot use this model: an org provider block, the key's
# model rules, or the plan. Without this, Hermes reads a 403 as a bad key and
# rotates the credential.
_NOT_ENTITLED = {"reason": "model_entitlement", "retryable": False, "should_fallback": True}
_BLOCKED = {"reason": "auth_permanent", "retryable": False, "should_fallback": False}
_OVERLOADED = {"reason": "overloaded", "retryable": True}
_SERVER_ERROR = {"reason": "server_error", "retryable": True}

ERROR_RULES: dict[str, dict[str, Any]] = {
    # 402: prepaid balance is empty, or bonus credits cannot pay for this provider.
    "insufficient_credits": _BILLING,
    "bonus_credits_not_applicable": _BILLING,
    # 429s that a retry cannot fix: spend caps and subscription budgets.
    "usage_limit_exceeded": _BILLING,
    "org_monthly_cap_exceeded": _BILLING,
    "rejected_subscription_exhausted": _BILLING,
    # 403: a coding model that only paid credits can use.
    "coding_model_requires_credits": _BILLING,
    # 403/400: model unavailable to this key.
    "provider_blocked": _NOT_ENTITLED,
    "model_access_denied": _NOT_ENTITLED,
    "org_model_not_allowed": _NOT_ENTITLED,
    "rejected_not_in_plan": _NOT_ENTITLED,
    "email_verification_required": _NOT_ENTITLED,  # free model before the account email is verified
    "provider_key_required": _NOT_ENTITLED,  # BYOK project without a key for this vendor
    "provider_unavailable": _NOT_ENTITLED,
    # 400: no provider serving this model speaks the requested API.
    "provider_no_responses_binding": {"reason": "model_not_found", "retryable": False, "should_fallback": True},
    # 401/403: the key or organization cannot be used at all.
    "organization_blocked": _BLOCKED,
    "key_suspended_abuse": _BLOCKED,
    "api_key_expired": _BLOCKED,
    # 503: every upstream credential for the model is busy, or billing is briefly unreachable.
    "pool_exhausted": _OVERLOADED,
    "billing_unavailable": _SERVER_ERROR,
}


def classify_gateway_error(
    error: Any = None,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Map LLM API error codes onto Hermes FailoverReasons; None defers to Hermes.

    Passed as the ``classify_api_error`` constructor field: it is a
    ProviderProfile dataclass field, so a same-named subclass method would be
    overwritten by the dataclass ``__init__``.
    """
    code = (error_code or "").strip().lower()
    if code.startswith("rejected_model_budget_exhausted"):
        return dict(_BILLING)
    verdict = ERROR_RULES.get(code)
    return dict(verdict) if verdict else None


# ── /usage ───────────────────────────────────────────────────────────────


def _money(value: Decimal) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def usage_snapshot(payload: dict[str, Any]) -> Any:
    """``AccountUsageSnapshot`` for a ``GET /v1/credits`` answer."""
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

    mode = str(payload.get("mode") or "").strip().lower()
    # Balances can be negative (pay as you go, overdraft).
    total = _decimal(payload.get("total_credits"), signed=True)
    credits = _decimal(payload.get("credits"), signed=True)
    bonus = _decimal(payload.get("credits_bonus"))
    details: list[str] = []
    if total is not None:
        line = f"Credits balance: {_money(total)}"
        if bonus:
            line += f" (incl. {_money(bonus)} bonus)"
        details.append(line)
    elif credits is not None:
        details.append(f"Credits balance: {_money(credits)}")
    if mode == "payg":
        details.append("Billing: pay as you go, invoiced after use")
    elif mode == "byok":
        details.append("Billing: your own provider keys (BYOK)")
    elif mode == "subscription":
        details.append("Billing: subscription plan")

    windows = []
    cap = payload.get("monthly_cap")
    if isinstance(cap, dict):
        limit, used = _decimal(cap.get("limit")), _decimal(cap.get("used"))
        if limit and used is not None:
            windows.append(
                AccountUsageWindow(
                    label="Monthly cap",
                    used_percent=float(min(used / limit, Decimal(1)) * 100),
                    detail=f"{_money(used)} of {_money(limit)} used",
                )
            )
    org = payload.get("organization")
    org_name = org.get("name") if isinstance(org, dict) else None
    return AccountUsageSnapshot(
        provider=PROVIDER_ID,
        source="credits_api",
        fetched_at=datetime.now(UTC),
        title="LLM API account",
        plan=str(org_name) if org_name else None,
        windows=tuple(windows),
        details=tuple(details),
    )


def _secret(name: str) -> str:
    from agent.secret_scope import get_secret

    return (get_secret(name) or "").strip()


# ── Profile ──────────────────────────────────────────────────────────────


def _is_ours(base_url: str | None) -> bool:
    """True for LLM API hosts (``api.llmapi.ai``, regional and dev gateways)."""
    host = (urlparse((base_url or "").strip()).hostname or "").lower()
    return host == "llmapi.ai" or host.endswith(".llmapi.ai")


class LLMAPIProfile(ProviderProfile):
    """LLM API: per-model wire, catalog-driven picker, error mapping, credits."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Tool-capable chat models from the catalog; a stale catalog beats none.

        None makes Hermes show fallback_models.
        """
        ensure_catalog(base_url, timeout=timeout)
        return catalog.picker() or None

    def api_mode_for_model(self, model: str, *, api_key: str | None = None) -> str | None:
        return api_mode_for(model)

    def normalize_model_id(self, model: str, *, api_key: str = "") -> str | None:
        """Catalog id for an alias or a case variant (``claude-haiku-4.5`` -> ``claude-haiku-4-5``)."""
        canonical = catalog.canonical(model)
        return canonical if canonical and canonical != model else None

    def invalidate_catalog_cache(self) -> None:
        catalog.mark_stale()

    def default_vision_model(self) -> str | None:
        """Keep image description on LLM API when the main model is text-only."""
        return AUX_MODEL

    def publish_capabilities(self, models: dict[str, ModelInfo], index: dict[str, str]) -> None:
        """Mirror the catalog into ``model_capabilities`` (ids and aliases), in place."""
        declared = getattr(self, "model_capabilities", None)
        if not isinstance(declared, dict):
            return
        entries = {model_id: capabilities(info) for model_id, info in models.items() if info.kind == "chat"}
        aliases = {
            name: entries[model_id]
            for info in models.values()
            for name in info.aliases
            if (model_id := info.id) in entries
        }
        declared.update({**aliases, **entries})

    def get_model_context_length(self, model: str) -> int | None:
        info = catalog.get(model)
        return info.context_length if info and info.context_length else None

    def live_context_length(self, model: str, *, api_key: str = "", base_url: str = "") -> int | None:
        """Context window on a cold cache: Hermes allows I/O here, so load the catalog first."""
        ensure_catalog(base_url, timeout=USAGE_TIMEOUT_SECONDS)
        return self.get_model_context_length(model)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        info = catalog.get(model)
        if info is None:
            return None
        return info.reasoning_levels

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = wire_effort(model, reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def request_headers(self, *, base_url: str, model: str = "", user_initiated: bool = False) -> dict[str, str]:
        return dict(ATTRIBUTION_HEADERS) if _is_ours(base_url) else {}

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Billed cost when the gateway reports it, else a catalog list-price estimate."""
        try:
            from agent.usage_pricing import CostResult
        except ImportError:
            return None
        billed = reported_cost(getattr(usage, "raw_usage", None))
        if billed is not None:
            return CostResult(
                amount_usd=billed,
                status="actual",
                source="provider_cost_api",
                label=f"${billed:.4f}",
                notes=("Billed cost reported by LLM API.",),
            )
        info = catalog.get(model)
        amount = estimated_cost(info, usage) if info else None
        if amount is None:
            return None
        return CostResult(
            amount_usd=amount,
            status="estimated",
            source="provider_models_api",
            label=f"~${amount:.4f}",
            notes=("LLM API list price; the dashboard shows billed cost.",),
        )

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None) -> Any | None:
        """Credit balance from ``GET /v1/credits`` on the credential's gateway.

        Hermes passes the runtime base URL (without ``/v1`` on the Anthropic
        wire) or none at all (``hermes usage``); ``api_base`` handles both.
        """
        key = (api_key or "").strip() or _secret(API_KEY_ENV)
        if not key:
            return None
        try:
            payload = _get_json(f"{api_base(base_url)}/credits", timeout=USAGE_TIMEOUT_SECONDS, api_key=key)
        except urllib.error.HTTPError as exc:
            logger.debug("llmapi: GET /credits -> HTTP %s", exc.code)
            return None
        except Exception as exc:  # noqa: BLE001 - /usage fails open
            logger.debug("llmapi: GET /credits failed: %s", exc)
            return None
        return usage_snapshot(payload) if isinstance(payload, dict) else None


def setup_reasoning_efforts(model: str) -> tuple[str, ...] | None:
    """Levels ``hermes model`` offers for *model*; ``()`` hides the effort prompt."""
    info = catalog.get(model)
    return info.reasoning_levels if info else None


def _supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop profile fields an older Hermes does not define.

    An unknown dataclass field raises TypeError at import, which the plugin
    loader swallows: the provider would silently fail to register.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs.items() if k in known}


PROFILE_FIELDS: dict[str, Any] = {
    "name": PROVIDER_ID,
    "aliases": ("llm-api",),
    "display_name": "LLM API",
    "description": "LLM API — Claude, GPT, Gemini, Kimi, DeepSeek and more on one key",
    "signup_url": SIGNUP_URL,
    # Key first; the *_BASE_URL entry is Hermes' user base-URL override.
    "env_vars": (API_KEY_ENV, BASE_URL_ENV),
    "base_url": BASE_URL,
    "hostname": "api.llmapi.ai",
    "auth_type": "api_key",
    "api_mode": CHAT,
    "supports_vision": True,
    "default_aux_model": AUX_MODEL,
    # Featured models: Hermes lists these first in the picker, then the live
    # catalog, and alone when the catalog is unreachable. Keep them current.
    "fallback_models": (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "gpt-5.5",
        "gpt-6.1-sol",
        "gemini-3.8-flash",
        "kimi-k3",
        "deepseek-v4-pro",
        "glm-5.3",
    ),
    "classify_api_error": classify_gateway_error,
    "setup_reasoning_efforts": setup_reasoning_efforts,
}

llmapi = LLMAPIProfile(**_supported_kwargs(LLMAPIProfile, PROFILE_FIELDS))
catalog.subscribe(llmapi.publish_capabilities)

register_provider(llmapi)
