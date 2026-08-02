from typing import Any

from pydantic import BaseModel, Field


class LlmTaskInput(BaseModel):
    system_message: str = Field(
        description="String containing the LLM's system rules"
    )
    user_message: str = Field(
        description="String containing the input for the task to be solved"
    )
    assistant_message: str | None = Field(
        description="String containing the start of the LLM's answer",
        default=None
    )
    output_format: Any = Field(
        description="Python type used to validate the LLM response"
    )
