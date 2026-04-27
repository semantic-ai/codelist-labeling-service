import torch
import torch.nn.functional as F


def predict(
    text: str,
    model,
    tokenizer,
    id2label: dict[int, str],
    problem_type: str,
    confidence_threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Run inference on a single text and return (label, confidence) pairs above threshold.

    Returns an empty list if no label exceeds the threshold.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits

    if problem_type == "single_label_classification":
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        return [(id2label[idx], conf)] if conf >= confidence_threshold else []

    elif problem_type == "multi_label_classification":
        probs = torch.sigmoid(logits).squeeze(0)
        return [
            (id2label[i], float(c))
            for i, c in enumerate(probs.tolist())
            if c >= confidence_threshold
        ]

    else:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")
