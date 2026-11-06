# LLM API provider for Hermes Agent

[LLM API](https://llmapi.ai) as a model provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent):
Claude, GPT, Gemini, Kimi, DeepSeek, GLM, Qwen and more on one key, each on the wire it works best on.

```bash
hermes plugins install llmapi
```

Hermes asks for `LLMAPI_API_KEY` during install. Create a key at
[app.llmapi.ai/dashboard/api-keys](https://app.llmapi.ai/dashboard/api-keys). Then run `hermes model`, choose
**LLM API**, and pick a model, or start a session with `hermes chat --provider llmapi`.

Requires Hermes 0.22.6 or later.

## How models are routed

Every request goes to `https://api.llmapi.ai/v1` with the same key. The plugin picks the API per model:

| Models | API | Why |
|---|---|---|
| Claude (`claude-*`) | Anthropic Messages, `/v1/messages` | Hermes adds Anthropic prompt-cache breakpoints on this API, so a tool loop reads the cache from its second turn. On Chat Completions it adds none. |
| GPT-5 and later (`gpt-5*`, `gpt-6*`, ...) | OpenAI Responses, `/v1/responses` | Reasoning items and tool calls round-trip natively. |
| Everything else | Chat Completions, `/v1/chat/completions` | |

The picker lists chat models from the LLM API catalog that support tool calling and have at least a 64K context
window (Hermes' minimum). Context windows, reasoning levels and list prices come from the same catalog. A user
`model_overrides` entry still wins over the catalog.

## Errors

| LLM API error | Hermes treats it as |
|---|---|
| `402 insufficient_credits`, `bonus_credits_not_applicable` | billing: stop and fall back |
| `429 usage_limit_exceeded`, `org_monthly_cap_exceeded`, subscription budget exhausted | billing: retrying cannot help |
| `403 provider_blocked`, `model_access_denied`, `org_model_not_allowed`, `rejected_not_in_plan` | model unavailable for this key: fall back to another model, keep the key |
| `403 organization_blocked`, `key_suspended_abuse`, `401 api_key_expired` | permanent auth failure |
| `429 rate_limit_exceeded` | rate limit: retried after `Retry-After` |
| `503 pool_exhausted` | overloaded: retried |
| anything else | Hermes' built-in handling |

## Usage and cost

- `/usage` (and `hermes usage`) shows the organization's credit balance, bonus credits, billing mode, and the
  monthly cap when one applies.
- Per-request cost: when the organization has *include cost in response* enabled, Hermes shows the cost
  LLM API billed. It is reported on Chat Completions calls and on non-streaming Messages and Responses calls.
  Otherwise Hermes shows an estimate from the catalog's list prices, including cache reads and writes. The
  dashboard is the source of truth for billed cost.

## Configuration

| Variable | Default | |
|---|---|---|
| `LLMAPI_API_KEY` | | Your LLM API key. Required. |
| `LLMAPI_BASE_URL` | `https://api.llmapi.ai/v1` | Another LLM API gateway, e.g. `https://eu.api.llmapi.ai/v1`. |

`hermes doctor` checks the key against `GET /v1/credits`.

## What the plugin does on your machine

- **Network.** All requests go to `api.llmapi.ai`, or to `LLMAPI_BASE_URL` if you set it:
  - `GET /v1/models`: the public model catalog, sent without your key, at most once an hour.
  - `GET /v1/credits`: your balance, sent with your key, only when you run `/usage`.
  - Model requests (`/v1/messages`, `/v1/responses`, `/v1/chat/completions`) are made by Hermes' own clients.
    On the last two the plugin adds `X-Source: hermes-agent` and `X-LLMAPI-App-Id: hermes-provider-llmapi`
    so LLM API can attribute the traffic. They carry no user data.
- **Files.** The plugin reads and writes no files. The catalog is cached in memory only.
- **Credentials.** The key is read through Hermes (`agent.secret_scope`) and only sent to LLM API.
- **Processes and telemetry.** None.

## Development

```bash
uv sync --group dev
uv run pytest                       # unit tests, no Hermes needed
uv run ruff check . && uv run ruff format --check .
```

End-to-end tests run real Hermes turns against Hermes' own fake Messages, Responses and Chat Completions servers:

```bash
git clone https://github.com/NousResearch/hermes-agent && (cd hermes-agent && uv sync --group dev --extra anthropic)
uv pip install --python hermes-agent/.venv/bin/python pytest pyyaml
PYTHONPATH=hermes-agent hermes-agent/.venv/bin/python -m pytest tests
hermes-agent/.venv/bin/hermes plugins validate llmapi --install-deps
```

## License

MIT
