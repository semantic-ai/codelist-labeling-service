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

2. Rebuild: `docker compose up`

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

2. Rebuild: `docker compose up`


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

## Logging

Application logs are controlled with `LOG_LEVEL` (default: `INFO`). SPARQL
query and update bodies are disabled by default because they are large,
multiline records; enable them temporarily with `LOG_SPARQL_ALL=true` when
query-level troubleshooting is needed.

LLM calls log only decision identifiers, payload sizes, code/description
counts, and returned codes. Full decision text and codelist descriptions are
not written to normal logs.

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

## Production classifier training

The codelist training task fetches the complete in-memory decision set and
the task's codelist, converts concept URIs to labels, and passes both directly
to the always-multi-label trainer. Empty class lists are valid negative samples;
cleaning, review-vote policy, duplicate merging, rare-label filtering,
negative balancing, and stratified train/validation/test splitting happen inside
the training pipeline.

`ml_training.label_policy` controls which review evidence enters training:

| Policy | Positive labels | Negative records |
|---|---|---|
| `all` | All generated labels | All fetched no-match records |
| `exclude_rejected` | Labels without a rejection vote | All fetched no-match records |
| `approved_only` | Labels with an approval and no rejection | Only no-match records with an approval and no rejection |

For `approved_only`, rejection wins when both approval and rejection votes are
present. Unreviewed records are removed instead of being converted into negative
examples.

Configure publication under `ml_training.huggingface` and provide the token in
the environment variable named by `api_key_env_var` (for example, `HF_TOKEN`).
The repository ID must be a real `namespace/name` repository. Legacy
`huggingface_token`, `huggingface_output_model_id`, and static commit or
concept-scheme registration values are not used by application training.

Training writes a unique run directory below `ml_training.output_dir`
containing the loadable model and tokenizer plus redacted configuration, label
mapping, split and weight summaries, threshold calibration, held-out metrics,
per-label reports, confusion counts, error analysis, and registration payloads.
Transient checkpoints are excluded from the upload. Thresholds are calibrated
on validation data only; the four AIRO values are derived once from held-out
test data as subset accuracy, macro precision, macro recall, and macro F1.

The task is successful only after the complete run directory has been uploaded
to Hugging Face and the AIRO SPARQL registration succeeds. The upload response
provides the authoritative repository URL and commit OID used for registration;
the local registration receipt records those values after upload. The
`training/development_minimum.ipynb` notebook is a validation/reference
notebook and is not a runtime dependency or data export source.

Runtime and test dependencies are pinned in `requirements.txt` and
`requirements-test.txt`. `pydantic`, `pydantic-settings`, `fastapi`, and
`PyYAML` are supplied by the pinned `decide-ai-service-base==0.3.0` wheel and
are intentionally not duplicated as direct service requirements.
