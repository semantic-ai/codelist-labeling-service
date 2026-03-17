from abc import ABC, abstractmethod
from openai import OpenAI
from .llm_task_models import LlmTaskInput
from pydantic import BaseModel


class RemoteLlmModel(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def __call__(self, input: LlmTaskInput) -> BaseModel:
        pass

    @abstractmethod
    def _format_messages(self, input: LlmTaskInput) -> BaseModel:
        """Function to format the LlmTaskInput to the specific LLM client input format"""
        pass


class OpenAIModel(RemoteLlmModel):
    """Class implementing the OpenAI LLM client"""

    def __init__(self, config: dict):
        super().__init__(
            config
        )

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
