"""
Unit tests for ClassifierAnnotatingTask (src/task/classify.py) and the AIRO
registration / discovery helpers in src/classifier/ld.py.

Test strategy
─────────────
Method                              SPARQL          HF / collaborators
build_airo_model_insert_query       real Virtuoso   –
fetch_models_for_codelist           real Virtuoso   –
ClassifierAnnotatingTask.process    mocked          mocked

The autouse `isolate_test_graphs` fixture (in conftest) wipes all test graphs
before and after each test.
"""

import logging
from unittest.mock import MagicMock

import pytest
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import GRAPHS, SPARQL_PREFIXES
from src.classifier.ld import build_airo_model_insert_query, fetch_models_for_codelist
from src.task.classify import ClassifierAnnotatingTask
from src.task.codelist import Codelist, CodelistEntry
from src.config import AppConfig, MLInferenceConfig

from tests.unit.conftest import (
    CONCEPT_SCHEME_URI,
    CONCEPT_URI,
    EXPRESSION_URI,
    TASK_URI,
    sparql_ask,
    sparql_update,
)


OTHER_SCHEME_URI = "http://test.example.org/codelists/other-scheme"
HUB_MODEL_ID = "test-org/m"
MODEL_URI_1 = "http://test.example.org/models/m1"
MODEL_URI_2 = "http://test.example.org/models/m2"


# ─────────────────────────────────────────────────────────────────────────────
# build_airo_model_insert_query — writes airo:producesOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAiroModelInsertQuery:
    def test_writes_produces_output_triple(self):
        q = build_airo_model_insert_query(
            hub_model_id=HUB_MODEL_ID,
            commit_oid="commit-1",
            code_git_sha="sha-1",
            hf_repo_url="https://huggingface.co/test-org/m",
            hf_tree_url="https://huggingface.co/test-org/m/tree/main/",
            source_repo_url="https://github.com/test/repo.git",
            results={},
            codelist_uri=CONCEPT_SCHEME_URI,
        )
        sparql_update(q)

        airo = SPARQL_PREFIXES["airo"]
        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    ?m a <{airo}AIModel> ;
                       <{airo}producesOutput> {sparql_escape_uri(CONCEPT_SCHEME_URI)} .
                }}
            }}
        """)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_models_for_codelist — discovery query
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_registered_models():
    """One AIModel for CONCEPT_SCHEME_URI, one for OTHER_SCHEME_URI."""
    airo = SPARQL_PREFIXES["airo"]
    dcterms = SPARQL_PREFIXES["dcterms"]
    sd = SPARQL_PREFIXES["sd"]
    xsd = "http://www.w3.org/2001/XMLSchema#"
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(MODEL_URI_1)}
                    a <{airo}AIModel> ;
                    <{airo}producesOutput> {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    <{dcterms}title> "{HUB_MODEL_ID}" ;
                    <{sd}dataPublished> "2026-05-01T00:00:00+00:00"^^<{xsd}dateTime> .
                {sparql_escape_uri(MODEL_URI_2)}
                    a <{airo}AIModel> ;
                    <{airo}producesOutput> {sparql_escape_uri(OTHER_SCHEME_URI)} ;
                    <{dcterms}title> "other-org/n" ;
                    <{sd}dataPublished> "2026-05-01T00:00:00+00:00"^^<{xsd}dateTime> .
            }}
        }}
    """)
    yield


class TestFetchModelsForCodelist:
    def test_returns_only_models_matching_codelist(self, two_registered_models):
        result = fetch_models_for_codelist(CONCEPT_SCHEME_URI)
        assert len(result) == 1
        assert result[0]["model_uri"] == MODEL_URI_1
        assert result[0]["hub_model_id"] == HUB_MODEL_ID

    def test_returns_empty_when_no_matches(self):
        assert fetch_models_for_codelist(CONCEPT_SCHEME_URI) == []

    def test_returns_only_latest_model_for_codelist(self):
        airo = SPARQL_PREFIXES["airo"]
        dcterms = SPARQL_PREFIXES["dcterms"]
        sd = SPARQL_PREFIXES["sd"]
        xsd = "http://www.w3.org/2001/XMLSchema#"
        sparql_update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    <http://test.example.org/models/old>
                        a <{airo}AIModel> ;
                        <{airo}producesOutput> {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                        <{dcterms}title> "old" ;
                        <{sd}dataPublished> "2026-01-01T00:00:00+00:00"^^<{xsd}dateTime> .
                    <http://test.example.org/models/new>
                        a <{airo}AIModel> ;
                        <{airo}producesOutput> {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                        <{dcterms}title> "new" ;
                        <{sd}dataPublished> "2026-05-08T00:00:00+00:00"^^<{xsd}dateTime> .
                }}
            }}
        """)
        result = fetch_models_for_codelist(CONCEPT_SCHEME_URI)
        assert len(result) == 1
        assert result[0]["hub_model_id"] == "new"


# ─────────────────────────────────────────────────────────────────────────────
# ClassifierAnnotatingTask.process — multi-model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier_task() -> ClassifierAnnotatingTask:
    """ClassifierAnnotatingTask without __init__ (no SPARQL on setup)."""
    task = object.__new__(ClassifierAnnotatingTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.classifier_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    return task


@pytest.fixture
def mock_inference_config():
    return AppConfig(ml_inference=MLInferenceConfig(confidence_threshold=0.5))


def _codelist_with_one_concept() -> Codelist:
    cl = Codelist([CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy")])
    cl.concept_scheme_uri = CONCEPT_SCHEME_URI
    return cl


def _hf_model_mock() -> MagicMock:
    m = MagicMock()
    m.config.problem_type = "single_label_classification"
    m.config.id2label = {0: "Affordable and Clean Energy"}
    return m


class TestClassifierAnnotatingTaskMultiModel:
    def test_runs_each_discovered_model_with_distinct_agent(
        self, classifier_task, mock_inference_config, mocker
    ):
        models = [
            {"model_uri": MODEL_URI_1, "hub_model_id": HUB_MODEL_ID},
            {"model_uri": MODEL_URI_2, "hub_model_id": HUB_MODEL_ID},
        ]
        mocker.patch.object(classifier_task, "get_target_graph",
                            return_value="http://test.example.org/graphs/target")
        mocker.patch.object(classifier_task, "get_job_confidence_threshold", return_value=None)
        mocker.patch.object(classifier_task, "fetch_codelist_uri_for_task",
                            return_value=CONCEPT_SCHEME_URI)
        mocker.patch.object(classifier_task, "fetch_codelist",
                            return_value=_codelist_with_one_concept())
        mocker.patch.object(
            classifier_task, "fetch_decisions_without_annotations_with_text",
            return_value=[{"uri": EXPRESSION_URI, "text": "some content"}],
        )
        mocker.patch.object(classifier_task, "create_output_container", return_value="container")

        mocker.patch("src.task.classify.get_config", return_value=mock_inference_config)
        mocker.patch("src.task.classify.fetch_models_for_codelist", return_value=models)
        mocker.patch("src.task.classify.AutoTokenizer.from_pretrained",
                     return_value=MagicMock())
        mocker.patch("src.task.classify.AutoModelForSequenceClassification.from_pretrained",
                     return_value=_hf_model_mock())
        mocker.patch("src.task.classify.classifier_predict",
                     return_value=[("Affordable and Clean Energy", 0.9)])
        link_cls = mocker.patch("src.task.classify.LinkingAnnotation")

        classifier_task.process()

        agents = [call.args[3] for call in link_cls.call_args_list]
        assert agents == [MODEL_URI_1, MODEL_URI_2]
        assert link_cls.return_value.add_to_triplestore_if_not_exists.call_count == 2

    def test_handles_no_discovered_models(
        self, classifier_task, mock_inference_config, mocker
    ):
        mocker.patch.object(classifier_task, "get_target_graph",
                            return_value="http://test.example.org/graphs/target")
        mocker.patch.object(classifier_task, "get_job_confidence_threshold", return_value=None)
        mocker.patch.object(classifier_task, "fetch_codelist_uri_for_task",
                            return_value=CONCEPT_SCHEME_URI)
        mocker.patch("src.task.classify.get_config", return_value=mock_inference_config)
        mocker.patch("src.task.classify.fetch_models_for_codelist", return_value=[])
        link_cls = mocker.patch("src.task.classify.LinkingAnnotation")

        classifier_task.process()

        link_cls.assert_not_called()
