from __future__ import annotations

import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from helpers import update
from src.task.codelist import Codelist

from ..config import MLTrainingConfig, get_config
from .data import PreparedDataset, prepare_dataset
from .ld import build_airo_model_insert_query
from .metrics import (
    canonical_airo_metrics,
    calibrate_thresholds,
    extract_logits,
    multilabel_metrics,
    sigmoid_probabilities,
    threshold_predictions,
    threshold_independent_metrics,
)


logger = logging.getLogger(__name__)


#------------------------------------------------------------
# Reproducibility / Seeding
#------------------------------------------------------------



def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch, including deterministic CUDA settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#------------------------------------------------------------
# Trainer & metrics computation 
#------------------------------------------------------------


class WeightedBCETrainer(Trainer):
    """Trainer using cached per-label BCE positive weights when configured."""

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, use_weighted_loss: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.use_weighted_loss = use_weighted_loss
        self.loss_fct: torch.nn.BCEWithLogitsLoss | None = None
        self.loss_device: torch.device | None = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        device = logits.device
        if self.loss_fct is None or self.loss_device != device:
            weight = self.pos_weight.to(device) if self.use_weighted_loss and self.pos_weight is not None else None
            self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
            self.loss_device = device
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_prediction) -> dict[str, float]:
    logits = extract_logits(eval_prediction)
    labels = (
        eval_prediction.label_ids
        if hasattr(eval_prediction, "label_ids")
        else eval_prediction[1]
    )
    return threshold_independent_metrics(labels, sigmoid_probabilities(logits))


#------------------------------------------------------------
# Publication / Hugging Face Hub helpers 
#------------------------------------------------------------


def _validate_publication_config(config: MLTrainingConfig) -> str:
    repository_id = config.huggingface.repo_id
    if repository_id.startswith("your-") or repository_id.endswith("/your-model-repository"):
        raise ValueError(
            "Set ml_training.huggingface.repo_id to a real Hugging Face repository "
            "before starting training."
        )
    token = os.getenv(config.huggingface.api_key_env_var)
    if not token:
        raise RuntimeError(
            f"Set the {config.huggingface.api_key_env_var} environment variable before training."
        )
    return token


def _set_up_repository(config: MLTrainingConfig, token: str) -> HfApi:
    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.huggingface.repo_id,
        repo_type=config.huggingface.repo_type,
        exist_ok=True,
        private=config.huggingface.private,
        token=token,
    )
    return api


def _upload_metadata(upload_result: Any) -> tuple[str, str]:
    commit_oid = getattr(upload_result, "oid", None)
    repository_url = getattr(upload_result, "repo_url", None)
    if hasattr(repository_url, "url"):
        repository_url = repository_url.url
    if not commit_oid or not repository_url:
        raise RuntimeError("Hugging Face upload returned no commit OID or repository URL.")
    return str(commit_oid), str(repository_url)


#------------------------------------------------------------
# Run directory & dataset preparation
#------------------------------------------------------------


def _run_directory(config: MLTrainingConfig) -> Path:
    model_name = Path(config.model_name).name or "codelist-classifier"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = (
        Path(config.output_dir)
        / model_name
        / f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def _to_dataset(frame: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame[["text", "labels"]], preserve_index=False)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )


def _precision_flags() -> tuple[bool, bool]:
    cuda_available = torch.cuda.is_available()
    bf16_supported = (
        cuda_available
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    return cuda_available and not bf16_supported, bf16_supported


def _training_arguments(config: MLTrainingConfig, checkpoint_directory: Path) -> TrainingArguments:
    if config.eval_strategy == "no":
        raise ValueError("eval_strategy must enable validation for model selection and calibration.")
    if config.load_best_model_at_end and config.save_strategy != config.eval_strategy:
        raise ValueError(
            "save_strategy must equal eval_strategy when load_best_model_at_end is enabled."
        )
    fp16, bf16 = _precision_flags()
    return TrainingArguments(
        output_dir=str(checkpoint_directory),
        lr_scheduler_type=config.lr_scheduler_type,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=f"eval_{config.metric_for_best_model}",
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        fp16=fp16,
        bf16=bf16,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        seed=config.seed,
    )


#------------------------------------------------------------
# Serialization helpers
#------------------------------------------------------------


def _redacted_config(config: MLTrainingConfig) -> dict[str, Any]:
    values = config.model_dump(mode="json")
    values.pop("huggingface_token", None)
    values.pop("huggingface_output_model_id", None)
    return values


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


#------------------------------------------------------------
# Artifact & model-card writing
#------------------------------------------------------------


def _write_artifacts(
    run_directory: Path,
    config: MLTrainingConfig,
    prepared: PreparedDataset,
    calibration: dict[str, Any],
    test_metrics: dict[str, float],
    canonical_metrics: dict[str, float],
    per_label_report: pd.DataFrame,
    confusion_rows: list[dict[str, Any]],
    error_analysis: pd.DataFrame,
    registration_payload: dict[str, Any],
) -> None:
    _write_json(run_directory / "config.json", _redacted_config(config))
    _write_json(run_directory / "label_mapping.json", {
        "label2id": prepared.label2id,
        "id2label": {str(key): value for key, value in prepared.id2label.items()},
        "active_labels": prepared.active_labels,
        "dropped_labels": prepared.dropped_labels,
    })
    _write_json(run_directory / "metrics.json", {
        "test": test_metrics,
        "airo": canonical_metrics,
    })
    _write_json(run_directory / "threshold.json", calibration)
    _write_json(run_directory / "split_distribution.json", prepared.split_distribution)
    _write_json(run_directory / "class_weights.json", prepared.weight_rows)
    _write_json(run_directory / "dataset_quality.json", prepared.quality_stats)
    _write_json(run_directory / "confusion_matrix.json", confusion_rows)
    _write_json(run_directory / "model_registration.json", registration_payload)
    per_label_report.to_csv(run_directory / "per_label_report.csv", index=False)
    error_analysis.to_json(run_directory / "error_analysis.jsonl", orient="records", lines=True)


def _write_model_card(
    run_directory: Path,
    repository_id: str,
    config: MLTrainingConfig,
    prepared: PreparedDataset,
    test_metrics: dict[str, float],
    calibration: dict[str, Any],
) -> None:
    metric_lines = "\n".join(
        f"- **{name}:** {value:.4f}" for name, value in test_metrics.items()
    )
    label_lines = "\n".join(f"- `{label}`" for label in prepared.active_labels)
    content = "\n".join([
        "---",
        "tags:",
        "- text-classification",
        "- multi-label-classification",
        "- codelist-labeling",
        f"base_model: {config.model_name}",
        "---",
        "",
        f"# {repository_id.rsplit('/', 1)[-1]}",
        "",
        f"Fine-tuned multi-label codelist classifier based on `{config.model_name}`.",
        "",
        f"## Labels ({len(prepared.active_labels)})",
        label_lines,
        "",
        "## Held-out evaluation",
        metric_lines,
        "",
        f"Validation global threshold: `{calibration['global_threshold']:.4f}`.",
        "Per-label thresholds are stored in `threshold.json`.",
        "",
    ])
    (run_directory / "README.md").write_text(content, encoding="utf-8")


def _build_trainer(
    config: MLTrainingConfig,
    prepared: PreparedDataset,
    checkpoint_directory: Path,
) -> tuple[WeightedBCETrainer, Any, Dataset, Dataset, Dataset]:
    """Create the tokenizer, datasets, model, and trainer for one run."""
    training_args = _training_arguments(config, checkpoint_directory)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = _to_dataset(prepared.train_df, tokenizer, config.max_length)
    validation_dataset = _to_dataset(prepared.validation_df, tokenizer, config.max_length)
    test_dataset = _to_dataset(prepared.test_df, tokenizer, config.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(prepared.active_labels),
        id2label=prepared.id2label,
        label2id=prepared.label2id,
        problem_type="multi_label_classification",
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
    ] if config.load_best_model_at_end else []
    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
        pos_weight=torch.tensor(prepared.positive_weights, dtype=torch.float32),
        use_weighted_loss=config.loss == "weighted_bce",
    )
    return trainer, tokenizer, train_dataset, validation_dataset, test_dataset


def _calibrate_model(
    trainer: WeightedBCETrainer,
    validation_dataset: Dataset,
    prepared: PreparedDataset,
    config: MLTrainingConfig,
) -> dict[str, Any]:
    """Train the model and derive thresholds from the validation split."""
    trainer.train()
    validation_predictions = trainer.predict(validation_dataset)
    validation_probabilities = sigmoid_probabilities(extract_logits(validation_predictions))
    validation_targets = np.asarray(prepared.validation_df["labels"].tolist()).astype(int)
    return calibrate_thresholds(
        validation_targets,
        validation_probabilities,
        config.global_threshold_grid,
        prepared.active_labels,
        config.per_label_threshold_min_positives,
        config.per_label_threshold_shrinkage,
    )


def _evaluate_model(
    trainer: WeightedBCETrainer,
    test_dataset: Dataset,
    prepared: PreparedDataset,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the held-out split and build the report artifacts."""
    test_predictions = trainer.predict(test_dataset)
    test_probabilities = sigmoid_probabilities(extract_logits(test_predictions))
    test_targets = np.asarray(prepared.test_df["labels"].tolist()).astype(int)
    test_metrics = {
        **threshold_independent_metrics(test_targets, test_probabilities),
        **multilabel_metrics(
            test_targets,
            test_probabilities,
            calibration["per_label_thresholds"],
            prepared.active_labels,
        ),
    }
    test_predictions_binary = threshold_predictions(
        test_probabilities,
        calibration["per_label_thresholds"],
        prepared.active_labels,
    )
    report = classification_report(
        test_targets,
        test_predictions_binary,
        target_names=prepared.active_labels,
        output_dict=True,
        zero_division=0,
    )
    per_label_report = pd.DataFrame(report).T.loc[prepared.active_labels].reset_index()
    per_label_report = per_label_report.rename(columns={"index": "label"})
    per_label_report["threshold"] = [
        calibration["per_label_thresholds"][label] for label in prepared.active_labels
    ]

    confusion_matrix = multilabel_confusion_matrix(test_targets, test_predictions_binary)
    confusion_rows = []
    for label_index, label in enumerate(prepared.active_labels):
        true_negative, false_positive, false_negative, true_positive = confusion_matrix[label_index].ravel()
        confusion_rows.append({
            "label": label,
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
        })

    error_rows = []
    for row_index, record in prepared.test_df.iterrows():
        true_labels = {
            prepared.active_labels[index]
            for index, value in enumerate(test_targets[row_index])
            if value
        }
        predicted_labels = {
            prepared.active_labels[index]
            for index, value in enumerate(test_predictions_binary[row_index])
            if value
        }
        error_rows.append({
            "text": record["text"],
            "true_labels": sorted(true_labels),
            "predicted_labels": sorted(predicted_labels),
            "false_positives": sorted(predicted_labels - true_labels),
            "false_negatives": sorted(true_labels - predicted_labels),
            "is_exact_match": true_labels == predicted_labels,
        })

    return {
        "test_metrics": test_metrics,
        "canonical_metrics": canonical_airo_metrics(test_metrics),
        "per_label_report": per_label_report,
        "confusion_rows": confusion_rows,
        "error_analysis": pd.DataFrame(error_rows),
    }


def _persist_run_outputs(
    run_directory: Path,
    tokenizer: Any,
    trainer: WeightedBCETrainer,
    config: MLTrainingConfig,
    prepared: PreparedDataset,
    calibration: dict[str, Any],
    evaluation: dict[str, Any],
    registration_payload: dict[str, Any],
) -> None:
    """Save the trained model and all local run artifacts."""
    trainer.save_model(str(run_directory))
    tokenizer.save_pretrained(str(run_directory))
    _write_model_card(
        run_directory,
        config.huggingface.repo_id,
        config,
        prepared,
        evaluation["test_metrics"],
        calibration,
    )
    _write_artifacts(
        run_directory,
        config,
        prepared,
        calibration,
        evaluation["test_metrics"],
        evaluation["canonical_metrics"],
        evaluation["per_label_report"],
        evaluation["confusion_rows"],
        evaluation["error_analysis"],
        registration_payload,
    )


def _publish_and_register(
    run_directory: Path,
    config: MLTrainingConfig,
    token: str,
    concept_scheme_uri: str,
    codelist: Codelist,
    active_labels: list[str],
    canonical_metrics: dict[str, float],
) -> tuple[str, str]:
    """Upload the run, write its receipt, and register it in the triplestore."""
    api = _set_up_repository(config, token)
    upload_result = api.upload_folder(
        folder_path=str(run_directory),
        repo_id=config.huggingface.repo_id,
        repo_type=config.huggingface.repo_type,
        commit_message=config.huggingface.commit_message,
        token=token,
        ignore_patterns=[
            "checkpoints/*",
            "checkpoint-*",
            "registration_receipt.json",
        ],
    )
    commit_oid, repository_url = _upload_metadata(upload_result)
    _write_json(run_directory / "registration_receipt.json", {
        "hub_model_id": config.huggingface.repo_id,
        "hf_repo_url": repository_url,
        "commit_oid": commit_oid,
        "concept_scheme_uri": concept_scheme_uri,
        "results": canonical_metrics,
    })

    label_to_uri = codelist.build_label_to_uri_map()
    try:
        supported_label_uris = [label_to_uri[label] for label in active_labels]
    except KeyError as error:
        raise ValueError(f"Active training label is not present in the codelist: {error.args[0]!r}") from error

    query = build_airo_model_insert_query(
        hub_model_id=config.huggingface.repo_id,
        commit_oid=commit_oid,
        hf_repo_url=repository_url,
        results=canonical_metrics,
        concept_scheme_uri=concept_scheme_uri,
        supported_label_uris=supported_label_uris,
    )
    logger.info("AIRO model registration query:\n%s", query)
    update(query, sudo=True)
    logger.info("Published and registered model %s at commit %s", config.huggingface.repo_id, commit_oid)
    return commit_oid, repository_url


#------------------------------------------------------------
# Main entry point
#------------------------------------------------------------


def train(
    decisions: list[dict[str, Any]],
    codelist: Codelist,
    concept_scheme_uri: str,
) -> dict[str, Any]:
    """Train, publish, and register one dynamic always-multi-label model."""
    # 1. Validate publication settings and prepare the reproducible dataset.
    config = get_config().ml_training
    token = _validate_publication_config(config)
    set_seed(config.seed)
    codelist_labels = codelist.get_labels()
    prepared = prepare_dataset(decisions, codelist_labels, config)
    logger.info(
        "Training dataset stats: total_samples=%d review_status_counts=%s "
        "label_policy=%s final_samples=%d final_label_counts=%s",
        prepared.quality_stats["raw_records"],
        prepared.quality_stats["review_status_counts"],
        prepared.quality_stats["label_policy"],
        prepared.quality_stats["prepared_records"],
        prepared.quality_stats["final_label_counts"],
    )

    # 2. Build and train the model, then calibrate thresholds on validation data.
    run_directory = _run_directory(config)
    trainer, tokenizer, _, validation_dataset, test_dataset = _build_trainer(
        config,
        prepared,
        run_directory / "checkpoints",
    )
    calibration = _calibrate_model(trainer, validation_dataset, prepared, config)
    evaluation = _evaluate_model(trainer, test_dataset, prepared, calibration)
    canonical_metrics = evaluation["canonical_metrics"]

    # 3. Persist the model and reports locally before publishing them.
    registration_payload = {
        **{
            key: value
            for key, value in config.model_registration.items()
            if key not in {"hub_model_id", "commit_oid", "hf_repo_url", "concept_scheme_uri", "results"}
        },
        "hub_model_id": config.huggingface.repo_id,
        "concept_scheme_uri": concept_scheme_uri,
        "results": canonical_metrics,
    }
    _persist_run_outputs(
        run_directory,
        tokenizer,
        trainer,
        config,
        prepared,
        calibration,
        evaluation,
        registration_payload,
    )

    # 4. Publish the run and register the exact uploaded commit.
    commit_oid, repository_url = _publish_and_register(
        run_directory,
        config,
        token,
        concept_scheme_uri,
        codelist,
        prepared.active_labels,
        canonical_metrics,
    )
    return {
        "run_directory": str(run_directory),
        "commit_oid": commit_oid,
        "hf_repo_url": repository_url,
        "results": canonical_metrics,
    }