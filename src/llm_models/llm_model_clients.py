from abc import ABC, abstractmethod
import json
import re

import requests
from openai import OpenAI
from pydantic import BaseModel

from .llm_task_models import LlmTaskInput


class RemoteLlmModel(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def __call__(self, input: LlmTaskInput) -> BaseModel:
        pass

    @abstractmethod
    def _format_messages(self, input: LlmTaskInput):
        """Function to format the LlmTaskInput to the specific LLM client input format"""
        pass


class OpenAIModel(RemoteLlmModel):
    """Class implementing the OpenAI LLM client"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = OpenAI()

    def __call__(self, input: LlmTaskInput) -> BaseModel:
        messages = self._format_messages(input)

        kwargs = {
            "model": self.config["model_name"],
            "input": messages,
            "text_format": input.output_format,
        }

        for key, value in self.config.items():
            if key != "model_name":
                kwargs[key] = value

        response = self._client.responses.parse(**kwargs)

        return response.output_parsed

    def _format_messages(self, input: LlmTaskInput):
        messages = [
            {
                "role": "system",
                "content": input.system_message
            },
            {
                "role": "user",
                "content": input.user_message
            }
        ]

        if input.assistant_message:
            messages.append({
                "role": "assistant",
                "content": input.assistant_message
            })

        return messages


class OllamaModel(RemoteLlmModel):
    """Class implementing the Ollama LLM client"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._base_url = config["base_url"]
        self._model = config["model_name"]
        self._temperature = config.get("temperature", 0.1)
        self._timeout = config.get("timeout", 120)

    def __call__(self, input: LlmTaskInput) -> BaseModel:
        prompt = self._format_messages(input)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }

        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()

        raw_text = response.json().get("response", "")
        return self._parse_response(raw_text, input.output_format)

    def _format_messages(self, input: LlmTaskInput) -> str:
        schema_json = json.dumps(input.output_format.model_json_schema(), indent=2)

        prompt = (
            f"System: {input.system_message}\n\n"
            f"{input.user_message}\n\n"
            f"IMPORTANT: You must respond ONLY with valid JSON matching this schema:\n"
            f"{schema_json}\n"
            f"Do not include any text before or after the JSON."
        )

        if input.assistant_message:
            prompt += f"\n\nAssistant: {input.assistant_message}"

        return prompt

    def _parse_response(self, raw_text: str, output_format: type[BaseModel]) -> BaseModel:
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
            f"Could not parse valid JSON from Ollama response. Raw text: {text[:500]}"
        )


def create_llm_client(llm_config) -> RemoteLlmModel | None:
    """Factory to create the appropriate LLM client based on config.

    Returns None for the 'random' provider.
    """
    config_dict = {
        "model_name": llm_config.model_name,
        "temperature": llm_config.temperature,
    }

    if llm_config.provider == "openai":
        return OpenAIModel(config_dict)
    elif llm_config.provider == "ollama":
        config_dict["base_url"] = llm_config.base_url
        config_dict["timeout"] = llm_config.timeout
        return OllamaModel(config_dict)
    elif llm_config.provider == "random":
        return None
    else:
        raise ValueError(f"Unknown LLM provider: {llm_config.provider}")
