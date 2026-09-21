import pytest

from src.utils.chunking import (
    compute_budget,
    map_over_chunks,
    merge_dict_of_lists,
    split_text,
)


def test_split_text_returns_original_text_when_it_fits():
    text = "# Heading\n\nA short paragraph."

    assert split_text(text, max_chars=100) == [text]


def test_split_text_prefers_structural_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = split_text(text, max_chars=35)

    assert chunks == ["First paragraph.\n\nSecond paragraph.", "Third paragraph."]
    assert all(len(chunk) <= 35 for chunk in chunks)


def test_split_text_falls_back_to_hard_character_limit():
    chunks = split_text("abcdefghij", max_chars=4)

    assert chunks == ["abcd", "efgh", "ij"]


def test_split_text_adds_bounded_overlap():
    chunks = split_text("one two three four", max_chars=9, overlap_chars=3)

    assert chunks == ["one", "one two", "two three", "ree four"]
    assert all(len(chunk) <= 9 for chunk in chunks)


def test_split_text_preserves_whitespace_at_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence."

    chunks = split_text(text, max_chars=20)

    assert "".join(chunks) == text


def test_map_over_chunks_bypasses_merge_for_small_document():
    merge_calls = []

    result = map_over_chunks(
        "small",
        10,
        lambda chunk, index, total: (chunk, index, total),
        lambda results: merge_calls.append(results),
    )

    assert result == ("small", 1, 1)
    assert merge_calls == []


def test_map_over_chunks_merges_successful_results_and_skips_failure():
    def call(chunk, index, total):
        if index == 2:
            raise ValueError("temporary failure")
        return {"action": [chunk]}

    result = map_over_chunks(
        "aa bb cc",
        2,
        call,
        merge_dict_of_lists,
        max_chunks=10,
    )

    assert result == {"action": ["aa", "cc"]}


def test_map_over_chunks_raises_when_all_chunks_fail():
    with pytest.raises(RuntimeError, match="All chunks failed"):
        map_over_chunks(
            "aa bb",
            2,
            lambda *_: (_ for _ in ()).throw(ValueError("failed")),
            merge_dict_of_lists,
        )


def test_compute_budget_accounts_for_prompt_overhead():
    assert compute_budget(100, overhead_chars=25, safety_margin=5) == 70
    assert compute_budget(None, overhead_chars=25) is None