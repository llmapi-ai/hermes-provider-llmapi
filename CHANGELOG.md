# Changelog

## 0.1.1

- Claude on `/v1/messages`: a 403 `permission_error` (the organization blocks the provider, or the key's model
  rules deny the model) now falls back to another model and keeps the key. Hermes read it as a bad key.
- `key_ip_restricted` maps to a permanent auth failure and `tenant_quota_exhausted` to billing.
- Catalog entries without a `kind` that take text and return text count as chat models.

## 0.1.0

First release.

- One provider, three APIs: Claude on the Anthropic Messages API (Hermes prompt caching), GPT-5 and later on
  the Responses API, everything else on Chat Completions, all on `https://api.llmapi.ai/v1` with one key.
- Picker, context windows, image input, reasoning levels and list prices from the public LLM API catalog;
  catalog aliases resolve in `/model`.
- Gateway errors mapped to Hermes recovery: out of credits and spend caps as billing, models the key cannot
  use as model-unavailable with fallback (never a key rotation).
- `/usage` shows the credit balance, bonus credits, billing mode and monthly cap.
- Per-request cost: the billed amount when the gateway reports it, otherwise a list-price estimate.
