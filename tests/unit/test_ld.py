import re

import pytest

from src.classifier.ld import build_airo_model_insert_query


MODEL_ID = "org/model-with-slash"
SCHEME_URI = "http://example.org/schemes/test"


def build(results):
    return build_airo_model_insert_query(
        hub_model_id=MODEL_ID,
        commit_oid="commit-123",
        hf_repo_url="https://huggingface.co/org/model-with-slash",
        results=results,
        concept_scheme_uri=SCHEME_URI,
    )


def test_builds_canonical_model_graph():
    query = build({"eval_accuracy": 0.9})

    assert "<http://lblod.data.gift/id/ai-models/org%2Fmodel-with-slash> a airo:AIModel, sd:SoftwareVersion" in query
    assert "ext:classificationLevel <http://lblod.data.gift/id/concept/ai-model-level/local-os-llm>" in query
    assert 'schema:datePublished "' in query
    assert "^^xsd:date" in query
    assert "<http://lblod.data.gift/id/components/codelist-classifier/v1.0.0> airo:hasModel" in query
    assert "<http://lblod.data.gift/id/inputs/text-input> a airo:Input" in query
    assert "<http://lblod.data.gift/id/outputs/org%2Fmodel-with-slash> a airo:Output" in query
    assert f"ext:forConceptScheme <{SCHEME_URI}>" in query
    assert "schema:codeRepository <https://huggingface.co/org/model-with-slash>" in query


def test_maps_only_aggregate_quality_metrics():
    query = build({
        "eval_accuracy": 0.9,
        "eval_precision": 0.8,
        "eval_recall": 0.7,
        "eval_f1": 0.75,
        "eval_loss": 0.1,
        "eval_runtime": 2.0,
        "eval_unknown": 1.0,
    })

    assert query.count("dqv:hasQualityMeasurement") == 1
    assert query.count("a dqv:QualityMeasurement") == 4
    assert "dqv:QualityMeasurement" not in query.split("dqv:hasQualityMeasurement", 1)[0]
    assert "^^xsd:decimal" in query
    assert '"0.9"^^xsd:decimal' in query
    assert '"0.8"^^xsd:decimal' in query
    assert '"0.7"^^xsd:decimal' in query
    assert '"0.75"^^xsd:decimal' in query
    assert "eval_loss" not in query
    assert "eval_runtime" not in query
    assert "ext:forClass" not in query


def test_empty_results_omit_measurement_link():
    query = build({})

    assert "dqv:hasQualityMeasurement" not in query
    assert "a dqv:QualityMeasurement" not in query
    assert re.search(r"airo:producesOutput\s+<[^>]+>\s+;\s*\.", query)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_rejects_invalid_metric_values(value):
    with pytest.raises(ValueError):
        build({"eval_accuracy": value})