from pydantic import BaseModel, Field, SecretStr, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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

    output_dir: str = Field(
        default="./outputs/experiment",
        description="Directory for training outputs and artifacts"
    )
    label_policy: Literal["all", "exclude_rejected"] = Field(
        default="exclude_rejected",
        description="Whether to keep labels rejected during human review"
    )
    min_label_samples: int = Field(default=1, ge=1)
    negative_ratio: float | None = Field(default=None, gt=0)
    test_size: float = Field(default=0.15, gt=0, lt=1)
    validation_size: float = Field(default=0.15, gt=0, lt=1)
    seed: int = Field(default=42)
    model_name: str = Field(
        default="distilbert/distilbert-base-uncased",
        description="Base transformer model for fine-tuning"
    )
    max_length: int = Field(default=512, ge=1)
    lr_scheduler_type: str = Field(default="cosine")
    num_train_epochs: int = Field(default=2, ge=1)
    train_batch_size: int = Field(default=8, ge=1)
    eval_batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    metric_for_best_model: str = Field(default="macro_average_precision")
    loss: Literal["bce", "weighted_bce"] = Field(default="weighted_bce")
    pos_weight_strategy: Literal["none", "full", "sqrt", "clipped"] = Field(default="none")
    pos_weight_cap: float | None = Field(default=None, gt=0)
    global_threshold_grid: list[float] = Field(default_factory=lambda: [0.5])
    per_label_threshold_min_positives: int = Field(default=10, ge=1)
    per_label_threshold_shrinkage: float = Field(default=0.35, ge=0, le=1)
    eval_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch")
    save_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch")
    load_best_model_at_end: bool = True
    greater_is_better: bool = True
    save_total_limit: int = Field(default=2, ge=1)
    logging_steps: int = Field(default=20, ge=1)
    report_to: str = Field(default="none")

    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        description="Learning rate for training"
    )
    epochs: int = Field(
        default=2,
        ge=1,
        description="Deprecated alias for num_train_epochs"
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
    huggingface: dict[str, str | bool] = Field(default_factory=lambda: {
        "repo_id": "your-organization/your-model-repository",
        "repo_type": "model",
        "private": False,
        "commit_message": "Upload fine-tuned codelist classifier with model card",
        "api_key_env_var": "HF_TOKEN",
    })
    model_registration: dict[str, str] = Field(default_factory=dict)


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
        description="User message template with {code_list}, {decision_text}, and {action_codes} placeholders"
    )


DEFAULT_USER_MESSAGE_PER_ACTION = (
    "Determine the best matching codes from the following list for EACH action in the given decision text.\n\n"
    '"""'
    "CODE LIST:\n"
    "{code_list}\n"
    '"""\n\n'
    '"""'
    "ACTION CODES:\n"
    "{action_codes}\n"
    '"""\n\n'
    '"""'
    "DECISION TEXT:\n"
    "{decision_text}\n"
    '"""'
    "For EACH action code listed above, provide the matching taxonomy codes from the code list. "
    "Return a JSON object where each key is an action code and the value is a list of matching codes. "
    "Only include codes that are truly matching and only from the given code list! "
    "If none of the codes match for an action, use an empty list for that action. "
    'Example format: {{"ACT-1": ["Code A", "Code B"], "ACT-2": []}}'
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


class AppConfig(BaseSettings):
    """Root application configuration model."""

    model_config = SettingsConfigDict(
        extra="forbid", # Reject extra fields not defined in the model
        env_nested_delimiter="__",  # allows SEGMENTATION__LLM__API_KEY etc.
        env_ignore_empty=True,      # treat empty string env vars as unset
    )

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