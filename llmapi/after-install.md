LLM API is installed.

1. Create a key at https://app.llmapi.ai/dashboard/api-keys and save it as `LLMAPI_API_KEY`
   (Hermes asked for it during install, or run `hermes setup`).
2. Run `hermes model`, choose **LLM API**, and pick a model.
3. `/usage` shows your credit balance; `hermes doctor` checks the key.

Claude models run on the Anthropic Messages API, so prompt caching works from the second turn.
