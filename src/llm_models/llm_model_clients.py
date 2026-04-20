"""LLM client factory using LangChain's unified chat model interface."""

import json
import logging
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import Any

from .llm_task_models import LlmTaskInput
from ..config import LlmConfig

logger = logging.getLogger(__name__)


class LangChainLlmClient:
    """Callable wrapper around a LangChain chat model with manual JSON parsing."""

    def __init__(self, chat_model):
        self._chat_model = chat_model

    def __call__(self, input: LlmTaskInput) -> BaseModel:
        schema_json = json.dumps(input.output_format.model_json_schema(), indent=2)

        messages = [
            SystemMessage(content=input.system_message),
            HumanMessage(content=(
                f"{input.user_message}\n\n"
                f"IMPORTANT: You must respond ONLY with valid JSON matching this schema:\n"
                f"{schema_json}\n"
                f"Do not include any text before or after the JSON."
            )),
        ]

        response = self._chat_model.invoke(messages)
        raw_text = response.content
        return self._parse_response(raw_text, input.output_format)

    @staticmethod
    def _parse_response(raw_text: str, output_format: type[BaseModel]) -> BaseModel:
        """Extract JSON from the LLM text response and parse into the Pydantic model."""
        text = raw_text.strip()

        # Try direct JSON parse
        try:
            return output_format.model_validate(json.loads(text))
        except (json.JSONDecodeError, Exception):
            pass

        # Try extracting from markdown code blocks
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            try:
                return output_format.model_validate(json.loads(match.group(1).strip()))
            except (json.JSONDecodeError, Exception):
                pass

        # Try finding a JSON object or array
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            try:
                return output_format.model_validate(json.loads(match.group(1)))
            except (json.JSONDecodeError, Exception):
                pass

        raise ValueError(
            f"Could not parse valid JSON from LLM response. Raw text: {text[:500]}"
        )


def create_llm_model(llm_config: LlmConfig) -> Any:
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
    chat_model = init_chat_model(llm_config.model_name, **kwargs)


def create_llm_client(llm_config) -> LangChainLlmClient | None:
    """Factory to create a LangChain-based LLM client from config.

    Returns None for the 'random' provider (callers handle the random fallback).
    """
    return LangChainLlmClient(create_llm_model(llm_config))
