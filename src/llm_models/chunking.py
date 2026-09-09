import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar


logger = logging.getLogger(__name__)

Result = TypeVar("Result")

DEFAULT_SEPARATORS = (
    r"(?=^#{1,6}\s)",
    "\n\n",
    "\n",
    r"(?<=[.!?])(?=\s+)",
    " ",
    "",
)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def compute_budget(
    max_input_chars: int | None,
    overhead_chars: int,
    safety_margin: int = 0,
) -> int | None:
    if max_input_chars is None:
        return None

    budget = max_input_chars - overhead_chars - safety_margin
    if budget <= 0:
        raise ValueError("LLM prompt overhead leaves no room for document text.")
    return budget


def _split_with_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    if separator.startswith("(?") or separator.startswith("(?m"):
        return [part for part in re.split(separator, text, flags=re.MULTILINE) if part]
    return [part for part in text.split(separator) if part]


def _separator_joiner(separator: str) -> str:
    return "" if separator == "" or separator.startswith("(?") else separator


def _split_recursive(
    text: str,
    max_chars: int,
    separators: Sequence[str],
) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[offset:offset + max_chars] for offset in range(0, len(text), max_chars)]

    separator = next(
        (candidate for candidate in separators if candidate == "" or re.search(candidate, text)),
        "",
    )
    separator_index = separators.index(separator)
    remaining_separators = separators[separator_index + 1:]
    pieces = _split_with_separator(text, separator)
    joiner = _separator_joiner(separator)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(piece) <= max_chars:
            current = piece
        else:
            chunks.extend(_split_recursive(piece, max_chars, remaining_separators))
            current = ""

    if current:
        chunks.append(current)
    return chunks


def split_text(
    text: str,
    max_chars: int | None,
    overlap_chars: int = 0,
    separators: Sequence[str] | None = None,
) -> list[str]:
    if max_chars is None or len(text) <= max_chars:
        return [text]
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars.")

    chunk_size = max_chars - overlap_chars if overlap_chars else max_chars
    chunks = _split_recursive(text, chunk_size, separators or DEFAULT_SEPARATORS)
    if overlap_chars == 0:
        return chunks

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap_chars:]
        boundary = " " if prefix and current and not prefix[-1].isspace() and not current[0].isspace() else ""
        overlapped.append(prefix + boundary + current)
    return overlapped


def chunk_context_note(index: int, total: int) -> str:
    return (
        f"NOTE: this is part {index} of {total} of a longer document. "
        "Only answer for content actually present in this part."
    )


def map_over_chunks(
    text: str,
    budget_chars: int | None,
    call_fn: Callable[[str, int, int], Result],
    merge_fn: Callable[[list[Result]], Result],
    *,
    overlap_chars: int = 0,
    max_chunks: int = 20,
    delay_seconds: float = 0,
    label: str = "document",
) -> Result:
    chunks = split_text(text, budget_chars, overlap_chars)
    if len(chunks) == 1:
        return call_fn(chunks[0], 1, 1)

    if len(chunks) > max_chunks:
        logger.error(
            "Chunk count for %s exceeds limit (%d > %d); processing first %d chunks.",
            label,
            len(chunks),
            max_chunks,
            max_chunks,
        )
        chunks = chunks[:max_chunks]

    total = len(chunks)
    logger.info(
        "Processing %s in %d chunks (chars=%s, estimated_tokens=%s)",
        label,
        total,
        [len(chunk) for chunk in chunks],
        [estimate_tokens(chunk) for chunk in chunks],
    )

    results: list[Result] = []
    last_error: Exception | None = None
    for index, chunk in enumerate(chunks, start=1):
        try:
            results.append(call_fn(chunk, index, total))
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Chunk %d/%d failed for %s: %s",
                index,
                total,
                label,
                exc,
            )
        if delay_seconds > 0 and index < total:
            time.sleep(delay_seconds)

    if not results:
        raise RuntimeError(f"All chunks failed for {label}.") from last_error
    return merge_fn(results)


def merge_dict_of_lists(
    results: list[dict[str, list[str]]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for result in results:
        for key, values in result.items():
            destination = merged.setdefault(key, [])
            destination.extend(value for value in values if value not in destination)
    return merged