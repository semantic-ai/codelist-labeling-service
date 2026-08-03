"""LLM client factory using LangChain's unified chat model interface."""

import json
import logging
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import TypeAdapter

from .llm_task_models import LlmTaskInput

logger = logging.getLogger(__name__)


class LangChainLlmClient:
    """Callable wrapper around a LangChain chat model with manual JSON parsing."""

    def __init__(self, chat_model):
        self._chat_model = chat_model

    def __call__(self, input: LlmTaskInput) -> Any:
        output_adapter = TypeAdapter(input.output_format)

        messages = [
            SystemMessage(content=input.system_message),
            HumanMessage(content=(
                f"{input.user_message}\n\n"
                f"IMPORTANT: Respond ONLY with the JSON value. "
                f"Do not include any text before or after the JSON."
            )),
        ]

        response = self._chat_model.invoke(messages)
        raw_text = response.content
        return self._parse_response(raw_text, output_adapter)

    @staticmethod
    def _parse_response(raw_text: str, output_adapter: TypeAdapter) -> Any:
        """Extract JSON from the LLM text response and validate its output type."""
        text = raw_text.strip()

        # Try direct JSON parse
        try:
            return output_adapter.validate_python(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code blocks
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            try:
                return output_adapter.validate_python(json.loads(match.group(1).strip()))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding a JSON object or array
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            try:
                return output_adapter.validate_python(json.loads(match.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass

        # Unwrap single-key dict (e.g. {"items": [...]}) and validate the inner value
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = next(iter(parsed.values()))
                return output_adapter.validate_python(inner)
        except (json.JSONDecodeError, ValueError):
            pass

        raise ValueError(
            f"Could not parse valid JSON from LLM response. Raw text: {text[:500]}"
        )


def create_llm_client(llm_config) -> LangChainLlmClient | None:
    """Factory to create a LangChain-based LLM client from config.

    Returns None for the 'random' provider (callers handle the random fallback).
    """
    if llm_config.provider == "random":
        return None

    kwargs: dict = {
        "model_provider": llm_config.provider,
        "temperature": llm_config.temperature,
    }

    if llm_config.api_key:
        kwargs["api_key"] = llm_config.api_key.get_secret_value()

    if llm_config.base_url:
        kwargs["base_url"] = llm_config.base_url

    if llm_config.timeout:
        kwargs["timeout"] = llm_config.timeout

    logger.info("Initializing LLM provider: %s, model: %s", llm_config.provider, llm_config.model_name)
    return LangChainLlmClient(init_chat_model(llm_config.model_name, **kwargs))
