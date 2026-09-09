import pytest

from src.classifier.ld import build_airo_model_insert_query


def test_build_airo_model_insert_query_registers_all_training_metrics():
    query = build_airo_model_insert_query(
        hub_model_id="example/model",
        commit_oid="commit-123",
        hf_repo_url="https://huggingface.co/example/model",
        results={
            "eval_accuracy": 0.8,
            "eval_precision": 0.7,
            "eval_recall": 0.6,
            "eval_f1": 0.65,
        },
        concept_scheme_uri="http://example.org/concept-scheme",
    )

    assert "<http://lblod.data.gift/id/ai-models/example%2Fmodel>" in query
    assert query.count("a dqv:QualityMeasurement") == 4
    assert query.count("a dqv:Metric") == 4
    assert "dqv:value \"0.8\"^^xsd:decimal" in query
    assert "dqv:value \"0.65\"^^xsd:decimal" in query


def test_build_airo_model_insert_query_rejects_non_numeric_metrics():
    with pytest.raises(ValueError, match="Invalid value for eval_f1"):
        build_airo_model_insert_query(
            hub_model_id="example/model",
            commit_oid="commit-123",
            hf_repo_url="https://huggingface.co/example/model",
            results={"eval_f1": "0.65"},
            concept_scheme_uri="http://example.org/concept-scheme",
        )