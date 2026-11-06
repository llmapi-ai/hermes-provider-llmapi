# Changelog

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
