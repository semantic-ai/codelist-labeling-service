# codelist-labeling-service
Service implementing the AI tasks for mapping decisions to codelists (DECIDe UC0.1)

## LLM Provider Configuration

This service uses [LangChain](https://docs.langchain.com/) for LLM integration, making it easy to swap providers without code changes. Configure the provider in `config.json` under the `llm` section.

### Configuration Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"ollama"` | LangChain provider name (e.g. `"ollama"`, `"mistralai"`, `"openai"`, `"anthropic"`) or `"random"` for testing |
| `model_name` | string | `"mistral-nemo"` | Model identifier for the chosen provider |
| `temperature` | float | `0.1` | Sampling temperature (0.0–2.0) |
| `api_key` | string \| null | `null` | API key (required for most cloud providers) |
| `base_url` | string \| null | `"http://ollama:11434"` | Base URL for the API endpoint |
| `timeout` | int | `120` | Request timeout in seconds |

### Provider Examples

#### Ollama (local, default)

No API key or extra package required. Point `base_url` at your Ollama instance.

```json
{
  "llm": {
    "provider": "ollama",
    "model_name": "mistral-nemo",
    "base_url": "http://ollama:11434",
    "temperature": 0.1,
    "timeout": 120
  }
}
```

#### Mistral AI

1. Update `config.json`:
   ```json
   {
     "llm": {
       "provider": "mistralai",
       "model_name": "mistral-medium-latest",
       "api_key": "your-mistral-api-key",
       "base_url": "https://api.mistral.ai/v1",
       "temperature": 0.1,
       "timeout": 120
     }
   }
   ```

2. Rebuild: `docker compose up --build`

#### OpenAI

1. Update `config.json`:
   ```json
   {
     "llm": {
       "provider": "openai",
       "model_name": "gpt-4o-mini",
       "api_key": "your-openai-api-key",
       "base_url": "https://api.openai.com/v1",
       "temperature": 0.1,
       "timeout": 120
     }
   }
   ```

2. Rebuild: `docker compose up --build`


### Per-Codelist Prompts

The `codelist_prompts` section in `config.json` lets you override the system and user messages per codelist. Keys are codelist URIs; `"default"` is the fallback. Both messages support `{code_list}` and `{decision_text}` placeholders.

```json
{
  "codelist_prompts": {
    "default": {
      "system_message": "You are a juridical assistant...",
      "user_message": "Determine the best matching codes...\n{code_list}\n{decision_text}"
    },
    "http://data.lblod.gift/id/conceptscheme/sdg-simple": {
      "system_message": "You are an SDG classification expert...",
      "user_message": "Analyze the decision text for SDGs...\n{code_list}\n{decision_text}"
    }
  }
}
```

## Running the tests

1. Have a Virtuoso running on localhost. The easiest way to achieve this:
```commandline
docker run -eSPARQL_UPDATE='true' -p8890:8890 -v./tests/config/virtuoso.ini:/data/virtuoso.ini redpencil/virtuoso:1.4.0-rc.1
```

2. Install test dependencies
```commandline
pip install -r requirements-test.txt
```

3. Run pytest
```commandline
pytest -v tests/unit
```