# codelist-labeling-service
Service implementing the AI tasks for mapping decisions to codelists (DECIDe UC0.1)

## LLM Provider Configuration

This service uses [LangChain](https://docs.langchain.com/) for LLM integration, making it easy to swap providers without code changes. Configure the provider in `config.json` under the `llm` section.

By default, the service uses Ollama (local, no API key needed). To switch to an external provider, e.g. Mistral AI:

1. Add the LangChain provider package to `requirements.txt`:
   ```
   langchain-mistralai
   ```

2. Update `config.json`:
   ```json
   {
     "llm": {
       "provider": "mistralai",
       "model_name": "mistral-small-latest",
       "temperature": 0.1,
       "api_key": "your-api-key",
       "base_url": null,
       "timeout": 120
     }
   }
   ```

3. Rebuild: `docker compose up --build`

The process is the same for any LangChain-supported provider (OpenAI, Anthropic, etc.) — just swap the package name (`langchain-openai`, `langchain-anthropic`, ...) and the `provider`/`model_name` values.

Set `provider` to `"random"` to skip the LLM and assign random labels (useful for pipeline testing).

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