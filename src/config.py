from pydantic import BaseModel, Field, SecretStr, ConfigDict, field_validator
from typing import Literal
from decide_ai_service_base.config import load_config


class AppSettingsConfig(BaseModel):
    """Application-level settings."""

    mode: Literal["development", "production", "staging", "test"] = Field(
        default="development",
        description="Application mode (development, production, etc.)"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="debug",
        description="Logging level (debug, info, warning, error)"
    )

    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize log level to lowercase and strip whitespace."""
        return v.strip().lower() if isinstance(v, str) else v


class LlmConfig(BaseModel):
    """LLM (Large Language Model) configuration."""

    provider: str = Field(
        default="ollama",
        description="LLM provider (e.g. 'ollama', 'openai', 'anthropic') or 'random' for testing"
    )
    model_name: str = Field(
        default="mistral-nemo",
        description="LLM model name"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the provider"
    )
    base_url: str | None = Field(
        default="http://ollama:11434",
        description="Base URL for the LLM server (for Ollama/compatible providers)"
    )
    timeout: int | None = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds"
    )


class MLTrainingConfig(BaseModel):
    """Machine Learning training configuration."""

    transformer: str = Field(
        default="distilbert/distilbert-base-uncased",
        description="Base transformer model for fine-tuning"
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        description="Learning rate for training"
    )
    epochs: int = Field(
        default=2,
        ge=1,
        description="Number of training epochs"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay for regularization"
    )
    huggingface_token: SecretStr | None = Field(
        default=None,
        description="HuggingFace API token for model upload"
    )
    huggingface_output_model_id: str | None = Field(
        default=None,
        description="Target model ID on HuggingFace Hub"
    )


class MLInferenceConfig(BaseModel):
    """Machine Learning inference configuration."""

    huggingface_model_id: str | None = Field(
        default=None,
        description="HuggingFace model ID to load for inference"
    )
    huggingface_token: SecretStr | None = Field(
        default=None,
        description="HuggingFace API token for downloading private models"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to emit an annotation"
    )


class CodelistPromptConfig(BaseModel):
    """Prompt configuration for a specific codelist."""

    system_message: str = Field(
        description="System message for the LLM"
    )
    user_message: str = Field(
        description="User message template with {code_list} and {decision_text} placeholders"
    )


DEFAULT_SYSTEM_MESSAGE = (
    "You are a juridical and administrative assistant that must determine "
    "the best matching codes from a list with a given text."
)
DEFAULT_USER_MESSAGE = (
    "Determine the best matching codes from the following list for the given public decision.\n\n"
    '"""'
    "CODE LIST:\n"
    "{code_list}\n"
    '"""\n\n'
    '"""'
    "DECISION TEXT:\n"
    "{decision_text}\n"
    '"""'
    "Provide your answer as a list of strings representing the matching codes. "
    "Provide all matching codes (can be a single one), but only those that are truly matching "
    "and only from the given list! If none of the codes match, return an empty list."
)


class AppConfig(BaseModel):
    """Root application configuration model."""

    model_config = ConfigDict(extra="forbid")  # Reject extra fields not defined in the model

    llm: LlmConfig = Field(
        default_factory=LlmConfig,
        description="LLM configuration"
    )
    ml_training: MLTrainingConfig = Field(
        default_factory=MLTrainingConfig,
        description="Machine learning training configuration"
    )
    ml_inference: MLInferenceConfig = Field(
        default_factory=MLInferenceConfig,
        description="Machine learning inference configuration"
    )
    codelist_prompts: dict[str, CodelistPromptConfig] = Field(
        default_factory=lambda: {
            "default": CodelistPromptConfig(
                system_message=DEFAULT_SYSTEM_MESSAGE,
                user_message=DEFAULT_USER_MESSAGE,
            )
        },
        description="Prompt configurations keyed by codelist URI, with 'default' as fallback"
    )

    def get_codelist_prompt(self, codelist_uri: str) -> CodelistPromptConfig:
        """Return the prompt config for a codelist URI, falling back to 'default'."""
        return self.codelist_prompts.get(codelist_uri, self.codelist_prompts["default"])


def get_config() -> AppConfig:
    return load_config(AppConfig)