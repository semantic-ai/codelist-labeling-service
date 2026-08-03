from types import SimpleNamespace

import pytest

from src.llm_models.llm_model_clients import LangChainLlmClient
from src.llm_models.llm_task_models import LlmTaskInput


class FakeChatModel:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.response_content)


def make_list_task_input() -> LlmTaskInput:
    return LlmTaskInput(
        system_message="Classify the text.",
        user_message="Some text",
        output_format=list[str],
    )


def test_client_accepts_and_returns_a_bare_list():
    chat_model = FakeChatModel('["Class A", "Class B"]')
    client = LangChainLlmClient(chat_model)

    result = client(make_list_task_input())

    assert result == ["Class A", "Class B"]


def test_client_unwraps_single_key_object():
    chat_model = FakeChatModel('{"items": ["Class A"]}')
    client = LangChainLlmClient(chat_model)

    result = client(make_list_task_input())

    assert result == ["Class A"]
