from types import SimpleNamespace
from unittest.mock import MagicMock, call

import torch

from src.classifier.predict import predict


def test_predict_uses_overflow_windows_for_multi_label_classification():
    tokenizer = MagicMock(side_effect=[
        {
            "input_ids": torch.ones((1, 700), dtype=torch.long),
            "attention_mask": torch.ones((1, 700), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "overflow_to_sample_mapping": torch.tensor([0, 0]),
        },
    ])
    model = MagicMock(return_value=SimpleNamespace(
        logits=torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    ))

    result = predict(
        "long text",
        model,
        tokenizer,
        {0: "first", 1: "second"},
        "multi_label_classification",
        confidence_threshold=0.8,
    )

    assert [label for label, _ in result] == ["first", "second"]
    assert tokenizer.call_count == 2
    assert tokenizer.call_args_list[0] == call(
        "long text",
        return_tensors="pt",
        truncation=False,
    )
    assert tokenizer.call_args_list[1] == call(
        "long text",
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=64,
        padding=True,
    )


def test_predict_keeps_legacy_path_for_short_single_label_input():
    tokenizer = MagicMock(side_effect=[
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    ])
    model = MagicMock(return_value=SimpleNamespace(
        logits=torch.tensor([[4.0, 0.0]])
    ))

    result = predict(
        "short text",
        model,
        tokenizer,
        {0: "first", 1: "second"},
        "single_label_classification",
        confidence_threshold=0.5,
    )

    assert result[0][0] == "first"
    assert tokenizer.call_count == 2
    assert tokenizer.call_args_list[0] == call(
        "short text",
        return_tensors="pt",
        truncation=False,
    )
    assert tokenizer.call_args_list[1] == call(
        "short text",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )


def test_predict_averages_windows_for_single_label_classification():
    tokenizer = MagicMock(side_effect=[
        {
            "input_ids": torch.ones((1, 700), dtype=torch.long),
            "attention_mask": torch.ones((1, 700), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "overflow_to_sample_mapping": torch.tensor([0, 0]),
        },
    ])
    model = MagicMock(return_value=SimpleNamespace(
        logits=torch.tensor([[4.0, 0.0], [0.0, 2.0]])
    ))

    result = predict(
        "long text",
        model,
        tokenizer,
        {0: "first", 1: "second"},
        "single_label_classification",
        confidence_threshold=0.5,
    )

    assert result[0][0] == "first"