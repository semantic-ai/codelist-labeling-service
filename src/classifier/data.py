from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


APPROVED_VOTE_URI = "http://mu.semte.ch/vocabularies/ext/annotation-review#approve"
REJECTED_VOTE_URI = "http://mu.semte.ch/vocabularies/ext/annotation-review#reject"
NO_MATCH_URI = "http://mu.semte.ch/vocabularies/ext/no-match-found"


# -----------------------------------------------------------------------------
# Prepared dataset contract
# -----------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    """Cleaned, encoded, and deterministically partitioned training data."""

    cleaned_records: list[dict[str, Any]]
    active_labels: list[str]
    dropped_labels: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    positive_weights: np.ndarray
    weight_rows: list[dict[str, Any]]
    split_distribution: list[dict[str, Any]]
    quality_stats: dict[str, Any]


# -----------------------------------------------------------------------------
# Record validation and review-vote normalization
# -----------------------------------------------------------------------------


def _as_label_list(value: Any, record_index: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Record {record_index} field 'classes' must be a list.")
    if any(not isinstance(label, str) or not label for label in value):
        raise ValueError(f"Record {record_index} field 'classes' contains an invalid label.")
    return sorted(set(value))


def _merge_votes(target: dict[str, list[str]], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for label, votes in source.items():
        if not isinstance(label, str) or not isinstance(votes, list):
            continue
        label_votes = target.setdefault(label, [])
        for vote in votes:
            if vote not in label_votes:
                label_votes.append(vote)


def _review_status(label_votes: dict[str, list[str]]) -> str:
    votes = [vote for label_votes in label_votes.values() for vote in label_votes]
    if any(vote == REJECTED_VOTE_URI for vote in votes):
        return "rejected"
    if any(vote == APPROVED_VOTE_URI for vote in votes):
        return "accepted"
    return "unvalidated"


def clean_records(
    decisions: Iterable[dict[str, Any]],
    codelist_labels: Iterable[str],
    label_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, normalize, deduplicate, and apply review votes to records."""
    # Validate the codelist and policy before processing any records.
    allowed_labels = set(codelist_labels)
    if not allowed_labels:
        raise ValueError("The task codelist contains no labels.")
    if label_policy not in {"all", "exclude_rejected", "approved_only"}:
        raise ValueError(
            "label_policy must be 'all', 'exclude_rejected', or 'approved_only'."
        )

    # Normalize records and merge duplicate text entries in input order.
    raw_records = list(decisions)
    records_by_text: dict[str, dict[str, Any]] = {}
    empty_texts_removed = 0
    duplicate_rows_merged = 0
    duplicate_texts: set[str] = set()
    records_removed_by_policy = 0
    labels_before_policy = 0

    for record_index, source_record in enumerate(raw_records):
        if not isinstance(source_record, dict):
            raise ValueError(f"Record {record_index} must be an object.")
        missing_fields = {"text", "classes"} - source_record.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Record {record_index} is missing required field(s): {missing}.")

        text = source_record["text"]
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            empty_texts_removed += 1
            continue

        classes = _as_label_list(source_record["classes"], record_index)
        unknown_labels = sorted(set(classes) - allowed_labels)
        if unknown_labels:
            raise ValueError(
                f"Record {record_index} contains labels outside the task codelist: {unknown_labels}."
            )

        normalized = dict(source_record)
        normalized["text"] = text
        normalized["classes"] = classes
        normalized["label_votes"] = {}
        _merge_votes(normalized["label_votes"], source_record.get("label_votes"))
        labels_before_policy += len(classes)

        if text not in records_by_text:
            records_by_text[text] = normalized
            continue

        duplicate_rows_merged += 1
        duplicate_texts.add(text)
        existing_classes = set(records_by_text[text]["classes"])
        records_by_text[text]["classes"] = sorted(existing_classes | set(classes))
        _merge_votes(records_by_text[text]["label_votes"], normalized["label_votes"])

    cleaned_records = list(records_by_text.values())

    # Apply the selected review policy after duplicate votes have been merged.
    if label_policy == "exclude_rejected":
        for record in cleaned_records:
            record["classes"] = [
                label
                for label in record["classes"]
                if REJECTED_VOTE_URI not in record["label_votes"].get(label, [])
            ]
    elif label_policy == "approved_only":
        approved_records = []
        for record in cleaned_records:
            record["classes"] = [
                label
                for label in record["classes"]
                if APPROVED_VOTE_URI in record["label_votes"].get(label, [])
                and REJECTED_VOTE_URI not in record["label_votes"].get(label, [])
            ]
            no_match_votes = record["label_votes"].get(NO_MATCH_URI, [])
            approved_no_match = (
                APPROVED_VOTE_URI in no_match_votes
                and REJECTED_VOTE_URI not in no_match_votes
            )
            if record["classes"] or approved_no_match:
                approved_records.append(record)
            else:
                records_removed_by_policy += 1
        cleaned_records = approved_records

    # Keep quality counters alongside the cleaned records for run artifacts.
    quality_stats = {
        "raw_records": len(raw_records),
        "empty_texts_removed": empty_texts_removed,
        "duplicate_rows_merged": duplicate_rows_merged,
        "duplicate_groups": len(duplicate_texts),
        "labels_before_policy": labels_before_policy,
        "labels_after_policy": sum(len(record["classes"]) for record in cleaned_records),
        "records_removed_by_policy": records_removed_by_policy,
        "cleaned_records": len(cleaned_records),
        "review_status_counts": {
            status: sum(
                _review_status(record["label_votes"]) == status
                for record in cleaned_records
            )
            for status in ("unvalidated", "rejected", "accepted")
        },
        "label_policy": label_policy,
    }
    return cleaned_records, quality_stats


# -----------------------------------------------------------------------------
# Class weighting
# -----------------------------------------------------------------------------


def transform_pos_weights(
    raw_weights: np.ndarray,
    strategy: str,
    cap: float | None,
) -> np.ndarray:
    if strategy == "none":
        return np.ones_like(raw_weights, dtype=np.float32)
    if strategy == "full":
        return raw_weights.astype(np.float32)
    if strategy == "sqrt":
        return np.sqrt(raw_weights).astype(np.float32)
    if strategy == "clipped":
        if cap is None or cap <= 0:
            raise ValueError("pos_weight_cap must be positive for clipped weights.")
        return np.minimum(raw_weights, cap).astype(np.float32)
    raise ValueError("pos_weight_strategy must be 'none', 'full', 'sqrt', or 'clipped'.")


# -----------------------------------------------------------------------------
# Sampling and deterministic split creation
# -----------------------------------------------------------------------------


def _downsample_negatives(
    frame: pd.DataFrame,
    negative_ratio: float | None,
    seed: int,
) -> pd.DataFrame:
    if negative_ratio is None:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    positive_rows = frame[frame["filtered_classes"].map(bool)]
    negative_rows = frame[~frame["filtered_classes"].map(bool)]
    maximum_negatives = int(len(positive_rows) * negative_ratio)
    if len(negative_rows) > maximum_negatives:
        negative_rows = negative_rows.sample(n=maximum_negatives, random_state=seed)
    return pd.concat([positive_rows, negative_rows]).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


def _split_frame(
    frame: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test frames."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as error:
        raise RuntimeError(
            "iterative-stratification is required before preparing training data."
        ) from error

    if len(frame) < 3:
        raise ValueError("At least three cleaned records are required for train/validation/test splits.")

    targets = np.asarray(frame["labels"].tolist(), dtype=np.float32)
    indices = np.arange(len(frame))
    try:
        # Split out the test set first, then split the remainder for validation.
        test_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_validation_indices, test_indices = next(
            test_splitter.split(indices, targets)
        )
        validation_ratio = validation_size / (1.0 - test_size)
        validation_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=validation_ratio, random_state=seed
        )
        train_sub_indices, validation_sub_indices = next(
            validation_splitter.split(
                train_validation_indices,
                targets[train_validation_indices],
            )
        )
    except (ValueError, StopIteration) as error:
        raise ValueError(
            "Unable to create stratified train/validation/test splits. "
            "Provide more records and at least one positive example per active label "
            "in each held-out split."
        ) from error

    train_indices = train_validation_indices[train_sub_indices]
    validation_indices = train_validation_indices[validation_sub_indices]
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    validation_frame = frame.iloc[validation_indices].reset_index(drop=True)
    test_frame = frame.iloc[test_indices].reset_index(drop=True)

    # Fail early when a split cannot represent every active label.
    for split_name, split_frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        if len(split_frame) == 0:
            raise ValueError(f"The {split_name} split is empty; increase the dataset size.")
        split_targets = np.asarray(split_frame["labels"].tolist(), dtype=np.float32)
        if np.any(split_targets.sum(axis=0) == 0):
            raise ValueError(
                f"The {split_name} split does not contain every active label; "
                "increase per-label support or adjust split sizes."
            )
    return train_frame, validation_frame, test_frame


# -----------------------------------------------------------------------------
# End-to-end dataset preparation
# -----------------------------------------------------------------------------


def prepare_dataset(
    decisions: list[dict[str, Any]],
    codelist_labels: list[str],
    config: Any,
) -> PreparedDataset:
    """Prepare the dynamic task data for always-multi-label training."""
    # Clean records before deriving the dynamic label vocabulary.
    cleaned_records, quality_stats = clean_records(
        decisions,
        codelist_labels,
        config.label_policy,
    )
    if not cleaned_records:
        raise ValueError("Cleaning removed every record; no training data remains.")

    # Keep labels with enough positive examples and assign deterministic IDs.
    label_counts: dict[str, int] = {}
    for record in cleaned_records:
        for label in record["classes"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    all_labels = sorted(label_counts)
    active_labels = [
        label for label in all_labels if label_counts[label] >= config.min_label_samples
    ]
    dropped_labels = [label for label in all_labels if label not in active_labels]
    if not active_labels:
        raise ValueError(
            "No active positive labels remain after min_label_samples filtering."
        )

    active_label_set = set(active_labels)
    label2id = {label: index for index, label in enumerate(active_labels)}
    id2label = {index: label for label, index in label2id.items()}
    encoded_records: list[dict[str, Any]] = []
    for record in cleaned_records:
        filtered_classes = sorted(set(record["classes"]) & active_label_set)
        labels = np.zeros(len(active_labels), dtype=np.float32)
        for label in filtered_classes:
            labels[label2id[label]] = 1.0
        encoded_record = dict(record)
        encoded_record["filtered_classes"] = filtered_classes
        encoded_record["labels"] = labels.tolist()
        encoded_records.append(encoded_record)

    if not any(record["filtered_classes"] for record in encoded_records):
        raise ValueError("No positive examples remain after rare-label filtering.")

    # Encode labels, then optionally reduce the number of all-negative examples.
    frame = pd.DataFrame(encoded_records)
    prepared_frame = _downsample_negatives(frame, config.negative_ratio, config.seed)
    train_frame, validation_frame, test_frame = _split_frame(
        prepared_frame,
        config.test_size,
        config.validation_size,
        config.seed,
    )

    # Derive positive weights from the training split only.
    train_targets = np.asarray(train_frame["labels"].tolist(), dtype=np.float32)
    positive_counts = train_targets.sum(axis=0)
    negative_counts = len(train_frame) - positive_counts
    raw_weights = np.divide(
        negative_counts,
        positive_counts,
        out=np.ones_like(positive_counts, dtype=np.float32),
        where=positive_counts > 0,
    )
    positive_weights = transform_pos_weights(
        raw_weights,
        config.pos_weight_strategy,
        config.pos_weight_cap,
    )
    weight_rows = [
        {
            "label": label,
            "train_positives": int(positive_counts[index]),
            "train_negatives": int(negative_counts[index]),
            "raw_positive_weight": float(raw_weights[index]),
            "applied_positive_weight": float(positive_weights[index]),
        }
        for index, label in enumerate(active_labels)
    ]
    # Record split-level statistics for reproducibility and diagnostics.
    split_distribution = []
    for split_name, split_frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        split_targets = np.asarray(split_frame["labels"].tolist(), dtype=np.float32)
        split_distribution.append({
            "split": split_name,
            "records": len(split_frame),
            "all_negative": int((split_targets.sum(axis=1) == 0).sum()),
            "label_counts": {
                label: int(split_targets[:, index].sum())
                for index, label in enumerate(active_labels)
            },
        })

    quality_stats.update({
        "active_labels": len(active_labels),
        "dropped_labels": len(dropped_labels),
        "prepared_records": len(prepared_frame),
        "all_negative_records": int((prepared_frame["filtered_classes"].map(len) == 0).sum()),
        "final_label_counts": {
            label: int(prepared_frame["filtered_classes"].map(
                lambda labels: label in labels
            ).sum())
            for label in active_labels
        },
    })
    return PreparedDataset(
        cleaned_records=encoded_records,
        active_labels=active_labels,
        dropped_labels=dropped_labels,
        label2id=label2id,
        id2label=id2label,
        train_df=train_frame,
        validation_df=validation_frame,
        test_df=test_frame,
        positive_weights=positive_weights,
        weight_rows=weight_rows,
        split_distribution=split_distribution,
        quality_stats=quality_stats,
    )