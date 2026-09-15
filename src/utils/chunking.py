import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar


logger = logging.getLogger(__name__)

Result = TypeVar("Result")

# This is only an approximation for converting a token budget to characters.
# The real ratio depends on the language and the model tokenizer.
APPROX_CHARS_PER_TOKEN = 4

DEFAULT_SEPARATORS = (
    r"(?=^#{1,6}\s)",
    "\n\n",
    "\n",
    r"(?<=[.!?])(?=\s+)",
    " ",
    "",
)


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count using four characters per token."""
    return math.ceil(len(text) / 4)


def compute_budget(
    max_input_chars: int | None,
    overhead_chars: int,
    safety_margin: int = 0,
) -> int | None:
    """Return the document-character budget left after prompt overhead.

    A ``None`` limit means that the caller does not want character-based
    chunking.  Otherwise, prompt text and the optional safety margin are
    subtracted from the configured maximum.
    """
    if max_input_chars is None:
        return None

    budget = max_input_chars - overhead_chars - safety_margin
    if budget <= 0:
        raise ValueError("LLM prompt overhead leaves no room for document text.")
    return budget


def compute_llm_chunk_budget(
    max_input_tokens: int | None,
    prompt_text: str,
    safety_margin_tokens: int = 0,
) -> int | None:
    """Convert an LLM token limit into a character budget for document chunks.

    The LLM splitter works on characters, while the configuration is expressed
    in tokens.  We estimate the prompt's token count, reserve the remaining
    tokens for document text, and convert those tokens to characters using
    ``APPROX_CHARS_PER_TOKEN``.
    """
    if max_input_tokens is None:
        return None
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be greater than zero.")
    if safety_margin_tokens < 0:
        raise ValueError("safety_margin_tokens must not be negative.")

    prompt_tokens = estimate_tokens(prompt_text)
    document_token_budget = max_input_tokens - prompt_tokens - safety_margin_tokens
    if document_token_budget <= 0:
        raise ValueError("LLM prompt overhead leaves no room for document text.")

    return document_token_budget * APPROX_CHARS_PER_TOKEN


def _split_with_separator(text: str, separator: str) -> list[str]:
    """Split text with either a literal or regular-expression separator."""
    if separator == "":
        return list(text)

    separator_is_regex = separator.startswith("(?")
    if separator_is_regex:
        # The built-in regexes use lookahead/lookbehind, so structural text at
        # the split boundary remains part of a resulting piece.
        pieces = re.split(separator, text, flags=re.MULTILINE)
    else:
        pieces = text.split(separator)

    return [piece for piece in pieces if piece]


def _separator_joiner(separator: str) -> str:
    """Return the text needed to join pieces split by this separator."""
    if separator == "" or separator.startswith("(?"):
        return ""
    return separator


def _split_recursive(
    text: str,
    max_chars: int,
    separators: Sequence[str],
) -> list[str]:
    """Split text at progressively smaller boundaries until chunks fit.

    Separators are ordered from the most meaningful boundary to the least
    meaningful one.  For example, a paragraph is kept together when possible;
    only an oversized paragraph is split into sentences, words, or characters.
    """
    if len(text) <= max_chars:
        return [text]

    if not separators:
        # There is no structural boundary left, so use a hard character limit.
        return [text[offset:offset + max_chars] for offset in range(0, len(text), max_chars)]

    # Pick the first separator that occurs in this piece of text.  The empty
    # separator is the final fallback and splits the text one character at a time.
    separator = ""
    for candidate in separators:
        if candidate == "" or re.search(candidate, text):
            separator = candidate
            break

    separator_index = separators.index(separator)
    remaining_separators = separators[separator_index + 1:]
    pieces = _split_with_separator(text, separator)
    joiner = _separator_joiner(separator)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        # Try to append this piece to the chunk currently being assembled.
        if current:
            candidate = f"{current}{joiner}{piece}"
        else:
            candidate = piece

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(piece) <= max_chars:
            current = piece
        else:
            # This piece is too large even on its own.  Try the next finer
            # separator, then continue assembling chunks at this level.
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
    """Split text into bounded chunks and optionally add overlap.

    ``max_chars`` is the final size limit for each returned chunk.  Overlap is
    taken from the end of the previous chunk so context is carried forward to
    the next model call.
    """
    if max_chars is None or len(text) <= max_chars:
        return [text]
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars.")

    if overlap_chars:
        # The splitter reserves space for the prefix that will be added later.
        chunk_size = max_chars - overlap_chars
    else:
        chunk_size = max_chars

    chunks = _split_recursive(text, chunk_size, separators or DEFAULT_SEPARATORS)
    if overlap_chars == 0:
        return chunks

    # Build each following chunk with the end of the previous chunk attached.
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        overlap = previous[-overlap_chars:]
        needs_separator = (
            overlap
            and current
            and not overlap[-1].isspace()
            and not current[0].isspace()
        )
        boundary = " " if needs_separator else ""
        overlapped.append(overlap + boundary + current)
    return overlapped


def chunk_context_note(index: int, total: int) -> str:
    """Return the note added to prompts when input is split into multiple parts."""
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
    """Call ``call_fn`` for each chunk and merge successful results.

    Failed chunks are logged and skipped so one bad model request does not
    discard results from all other chunks.  The function raises only when every
    chunk fails.
    """
    chunks = split_text(text, budget_chars, overlap_chars)
    if len(chunks) == 1:
        # Avoid invoking the merge function when no split was needed.  This
        # preserves the normal single-request result shape for callers.
        logger.info(
            "Processing %s as one chunk (chars=%d, estimated_tokens=%d)",
            label,
            len(chunks[0]),
            estimate_tokens(chunks[0]),
        )
        return call_fn(chunks[0], 1, 1)

    if len(chunks) > max_chunks:
        # Avoid an unbounded number of model calls for unexpectedly long input.
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
        logger.info(
            "Starting chunk %d/%d for %s (chars=%d, estimated_tokens=%d)",
            index,
            total,
            label,
            len(chunk),
            estimate_tokens(chunk),
        )

        started_at = time.monotonic()
        try:
            result = call_fn(chunk, index, total)
            results.append(result)
            elapsed = time.monotonic() - started_at
            logger.info(
                "Finished chunk %d/%d for %s (successful=%d, elapsed_seconds=%.2f)",
                index,
                total,
                label,
                len(results),
                elapsed,
            )
        except Exception as exc:
            last_error = exc
            elapsed = time.monotonic() - started_at
            logger.warning(
                "Chunk %d/%d failed for %s (chars=%d, elapsed_seconds=%.2f): %s",
                index,
                total,
                label,
                len(chunk),
                elapsed,
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
    """Merge list values by key while preserving their first-seen order."""
    merged: dict[str, list[str]] = {}
    for result in results:
        for key, values in result.items():
            destination = merged.setdefault(key, [])
            for value in values:
                if value not in destination:
                    destination.append(value)
    return merged