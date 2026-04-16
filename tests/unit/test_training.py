"""
Unit tests for ClassifierTrainingTask (src/task/training.py).

Test strategy
─────────────
Method                              SPARQL          ML training
fetch_decisions_with_classes()      real Virtuoso   –
convert_classes_to_original_names() –               –  (pure dict mapping)
process()                           all mocked      mocked (train() replaced)

The autouse `isolate_test_graphs` fixture wipes all test graphs before and
after every test.

Known bug documented by tests
──────────────────────────────
convert_classes_to_original_names is decorated @staticmethod but keeps
'self' as its first parameter.  When process() calls
  self.convert_classes_to_original_names(decisions, codelist_entries)
Python omits the implicit instance argument, so the method receives
  self=decisions, decisions=codelist_entries, codelist=<missing>
and raises TypeError.  TestProcess.test_bug_convert_classes_wrong_signature
documents this.  The remaining process() tests mock the broken method so that
the orchestration logic around it can still be verified.
"""

import logging
from unittest.mock import MagicMock, patch

import helpers
import pytest
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import GRAPHS
from src.task.codelist import Codelist, CodelistEntry
from src.task.training import ClassifierTrainingTask
from src.config import AppConfig, MLTrainingConfig, LlmConfig

from tests.unit.conftest import (
    ANNOTATION_URI,
    CONCEPT_SCHEME_URI,
    CONCEPT_URI,
    CONCEPT_URI_2,
    EXPRESSION_CONTENT,
    EXPRESSION_CONTENT_2,
    EXPRESSION_URI,
    EXPRESSION_URI_2,
    NS,
    TASK_URI,
)

# ---------------------------------------------------------------------------
# Test-local constants (cross-scheme isolation scenario)
# ---------------------------------------------------------------------------

OTHER_SCHEME_URI = "http://test.example.org/codelists/other-scheme"
OTHER_CONCEPT_URI = "http://test.example.org/concepts/other-scheme-concept"
OTHER_ANNOTATION_URI = "http://test.example.org/annotations/annotation-other"

_PUBLIC_GRAPH = GRAPHS["public"]


# ---------------------------------------------------------------------------
# Task fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def training_task() -> ClassifierTrainingTask:
    """Minimal ClassifierTrainingTask created without __init__ (no SPARQL on setup)."""
    task = object.__new__(ClassifierTrainingTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.training_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    return task


# ---------------------------------------------------------------------------
# Triplestore data fixtures
#
# Layout:
#   EXPRESSION_URI   – annotated with CONCEPT_URI   (in CONCEPT_SCHEME_URI) ✓
#   EXPRESSION_URI_2 – annotated with OTHER_CONCEPT_URI (in OTHER_SCHEME_URI) ✗
#
# Only EXPRESSION_URI should appear in fetch_decisions_with_classes() results
# when the task's concept scheme is CONCEPT_SCHEME_URI.
# ---------------------------------------------------------------------------

@pytest.fixture
def task_scheme_concepts():
    """Register CONCEPT_URI and CONCEPT_URI_2 under CONCEPT_SCHEME_URI."""
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(_PUBLIC_GRAPH)} {{
                {sparql_escape_uri(CONCEPT_URI)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Affordable and Clean Energy"@en .
                {sparql_escape_uri(CONCEPT_URI_2)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Climate Action"@en .
            }}
        }}
    """)
    yield


@pytest.fixture
def other_scheme_concept():
    """Register OTHER_CONCEPT_URI under OTHER_SCHEME_URI (a different scheme)."""
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(_PUBLIC_GRAPH)} {{
                {sparql_escape_uri(OTHER_CONCEPT_URI)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(OTHER_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Irrelevant Domain"@en .
            }}
        }}
    """)
    yield


@pytest.fixture
def decision_in_task_scheme(task_scheme_concepts):
    """
    EXPRESSION_URI as an eli:Expression with content, annotated with CONCEPT_URI
    which belongs to CONCEPT_SCHEME_URI.
    """
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI)} .
            }}
        }}
    """)
    yield


@pytest.fixture
def decision_in_other_scheme(other_scheme_concept):
    """
    EXPRESSION_URI_2 as an eli:Expression, annotated with OTHER_CONCEPT_URI
    which belongs to OTHER_SCHEME_URI (not the task's scheme).
    This decision must NOT appear in fetch_decisions_with_classes() results.
    """
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI_2)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT_2}" .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(OTHER_ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI_2)} ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(OTHER_CONCEPT_URI)} .
            }}
        }}
    """)
    yield


@pytest.fixture
def decision_with_two_task_scheme_classes(task_scheme_concepts):
    """
    EXPRESSION_URI annotated with both CONCEPT_URI and CONCEPT_URI_2,
    both from CONCEPT_SCHEME_URI.
    """
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI)} .
                {sparql_escape_uri(OTHER_ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI_2)} .
            }}
        }}
    """)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# ClassifierTrainingTask.fetch_decisions_with_classes()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchDecisionsWithClasses:

    @pytest.fixture(autouse=True)
    def patch_codelist_uri(self, training_task, mocker):
        """Mock fetch_codelist_uri_for_task so tests stay focused on data retrieval."""
        mocker.patch.object(
            training_task,
            "fetch_codelist_uri_for_task",
            return_value=CONCEPT_SCHEME_URI,
        )

    def test_returns_one_result_for_annotated_expression(
        self, training_task, decision_in_task_scheme
    ):
        results = training_task.fetch_decisions_with_classes()

        assert len(results) == 1

    def test_result_contains_decision_uri(
        self, training_task, decision_in_task_scheme
    ):
        result = training_task.fetch_decisions_with_classes()[0]

        assert result["decision"] == EXPRESSION_URI

    def test_result_classes_contains_concept_uri(
        self, training_task, decision_in_task_scheme
    ):
        result = training_task.fetch_decisions_with_classes()[0]

        assert CONCEPT_URI in result["classes"]

    def test_result_text_includes_expression_content(
        self, training_task, decision_in_task_scheme
    ):
        result = training_task.fetch_decisions_with_classes()[0]

        assert EXPRESSION_CONTENT in result["text"]

    def test_excludes_decision_annotated_only_with_other_scheme_concept(
        self, training_task, decision_in_other_scheme
    ):
        """
        Core isolation test: EXPRESSION_URI_2 is annotated with OTHER_CONCEPT_URI
        which lives in OTHER_SCHEME_URI, not in the task's CONCEPT_SCHEME_URI.
        The VALUES ?scheme filter in the subquery must exclude it entirely.
        """
        results = training_task.fetch_decisions_with_classes()
        decision_uris = [r["decision"] for r in results]

        assert EXPRESSION_URI_2 not in decision_uris

    def test_cross_scheme_annotation_yields_empty_results(
        self, training_task, decision_in_other_scheme
    ):
        """When the only annotated decision belongs to a different scheme, the
        result list is empty."""
        results = training_task.fetch_decisions_with_classes()

        assert results == []

    def test_returns_empty_list_when_no_annotations(self, training_task):
        results = training_task.fetch_decisions_with_classes()

        assert results == []

    def test_returns_multiple_classes_for_decision_with_two_annotations(
        self, training_task, decision_with_two_task_scheme_classes
    ):
        """GROUP_CONCAT collects all concept URIs for the same decision."""
        result = training_task.fetch_decisions_with_classes()[0]

        assert CONCEPT_URI in result["classes"]
        assert CONCEPT_URI_2 in result["classes"]

    def test_classes_contains_only_task_scheme_concepts_when_both_schemes_present(
        self,
        training_task,
        decision_in_task_scheme,
        decision_in_other_scheme,
    ):
        """
        When both a task-scheme and an other-scheme annotation exist in the store,
        only the task-scheme decision appears and the other-scheme concept is absent.
        """
        results = training_task.fetch_decisions_with_classes()
        decision_uris = [r["decision"] for r in results]

        assert EXPRESSION_URI in decision_uris
        assert EXPRESSION_URI_2 not in decision_uris

        all_classes = [c for r in results for c in r["classes"]]
        assert OTHER_CONCEPT_URI not in all_classes

    def test_each_result_has_decision_classes_and_text_keys(
        self, training_task, decision_in_task_scheme
    ):
        result = training_task.fetch_decisions_with_classes()[0]

        assert "decision" in result
        assert "classes" in result
        assert "text" in result

    def test_classes_is_a_list(
        self, training_task, decision_in_task_scheme
    ):
        result = training_task.fetch_decisions_with_classes()[0]

        assert isinstance(result["classes"], list)

    def test_excludes_non_classifying_annotation(
        self, training_task, task_scheme_concepts
    ):
        """An oa:linking annotation (not oa:classifying) must not count."""
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                    {sparql_escape_uri(EXPRESSION_URI)} a {sparql_escape_uri(NS["eli"] + "Expression")} .
                }}
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    {sparql_escape_uri(ANNOTATION_URI)}
                        a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                        {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "linking")} ;
                        {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                        {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI)} .
                }}
            }}
        """)

        results = training_task.fetch_decisions_with_classes()

        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# ClassifierTrainingTask.convert_classes_to_original_names()
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertClassesToOriginalNames:
    """
    The method converts concept URIs to their human-readable labels using
    build_uri_to_label_map().

    NOTE: The method is decorated @staticmethod but has 'self' as its first
    parameter.  When called via an instance or the class, the argument
    binding is shifted: 'self' receives 'decisions', 'decisions' receives
    'codelist', and 'codelist' is missing → TypeError.
    test_bug_wrong_signature documents this.

    The remaining tests call the underlying function directly to verify the
    conversion logic is correct once the signature is fixed.
    """

    @pytest.fixture
    def fn(self):
        """The raw function behind the staticmethod descriptor, bypassing the
        broken signature so the conversion logic can be tested independently."""
        return ClassifierTrainingTask.__dict__[
            "convert_classes_to_original_names"
        ].__func__ if hasattr(
            ClassifierTrainingTask.__dict__["convert_classes_to_original_names"],
            "__func__",
        ) else ClassifierTrainingTask.__dict__["convert_classes_to_original_names"]

    @pytest.fixture
    def codelist(self):
        return Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy"),
            CodelistEntry(uri=CONCEPT_URI_2, label="Climate Action"),
        ])

    def test_bug_wrong_signature_raises_typeerror_when_called_normally(
        self, training_task, codelist
    ):
        """
        Documents the @staticmethod + extra 'self' parameter bug.
        Calling convert_classes_to_original_names(decisions, codelist) from an
        instance provides only 2 args but the method expects 3.
        """
        decisions = [{"decision": EXPRESSION_URI, "classes": [CONCEPT_URI], "text": ""}]
        with pytest.raises(TypeError):
            training_task.convert_classes_to_original_names(decisions, codelist)

    def test_replaces_uri_with_label(self, fn, codelist):
        decisions = [{"decision": EXPRESSION_URI, "classes": [CONCEPT_URI], "text": ""}]
        # Call with correct 3 args: (self_as_decisions, decisions_as_codelist, codelist)
        # is not possible without the bug.  Call the raw function with extra dummy arg:
        result = fn(None, decisions, codelist)
        assert result[0]["classes"] == ["Affordable and Clean Energy"]

    def test_replaces_multiple_uris(self, fn, codelist):
        decisions = [{
            "decision": EXPRESSION_URI,
            "classes": [CONCEPT_URI, CONCEPT_URI_2],
            "text": "",
        }]
        result = fn(None, decisions, codelist)
        assert set(result[0]["classes"]) == {
            "Affordable and Clean Energy", "Climate Action"
        }

    def test_preserves_uri_when_not_in_map(self, fn, codelist):
        """URIs absent from the codelist are left as-is (uri_to_label.get(c, c))."""
        unknown_uri = "http://test.example.org/concepts/unknown"
        decisions = [{
            "decision": EXPRESSION_URI,
            "classes": [unknown_uri],
            "text": "",
        }]
        result = fn(None, decisions, codelist)
        assert result[0]["classes"] == [unknown_uri]

    def test_handles_empty_decisions_list(self, fn, codelist):
        result = fn(None, [], codelist)
        assert result == []

    def test_handles_decision_with_no_classes(self, fn, codelist):
        decisions = [{"decision": EXPRESSION_URI, "classes": [], "text": ""}]
        result = fn(None, decisions, codelist)
        assert result[0]["classes"] == []


# ─────────────────────────────────────────────────────────────────────────────
# ClassifierTrainingTask.process()  – all collaborators mocked
# ─────────────────────────────────────────────────────────────────────────────

class TestProcess:
    """
    process() is tested with every collaborator mocked.
    convert_classes_to_original_names is also mocked to work around the
    @staticmethod signature bug; a separate test documents that bug.
    """

    @pytest.fixture
    def mock_codelist(self):
        codelist = Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy"),
            CodelistEntry(uri=CONCEPT_URI_2, label="Climate Action"),
        ])
        return codelist

    @pytest.fixture
    def mock_config(self):
        return AppConfig(
            llm=LlmConfig(provider="ollama", model_name="test"),
            ml_training=MLTrainingConfig(
                huggingface_output_model_id="test-org/test-model",
            ),
        )

    @pytest.fixture
    def labeled_decisions(self):
        return [
            {"decision": EXPRESSION_URI,   "classes": ["Affordable and Clean Energy"], "text": EXPRESSION_CONTENT},
            {"decision": EXPRESSION_URI_2,  "classes": ["Climate Action"],              "text": EXPRESSION_CONTENT_2},
        ]

    def test_bug_convert_classes_wrong_signature_raises_typeerror(
        self, training_task, mock_codelist, labeled_decisions, mocker
    ):
        """
        Documents the @staticmethod + 'self' parameter bug.
        process() calls self.convert_classes_to_original_names(decisions, codelist)
        which resolves to a 2-arg call on a 3-param function → TypeError.
        """
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=labeled_decisions
        )

        with pytest.raises(TypeError):
            training_task.process()

    def test_skips_train_when_no_labeled_decisions(
        self, training_task, mock_codelist, mocker
    ):
        """If every decision has an empty classes list, train() must not be called."""
        unlabeled = [{"decision": EXPRESSION_URI, "classes": [], "text": ""}]
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=unlabeled
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=unlabeled,
        )
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        mock_train.assert_not_called()

    def test_skips_train_when_fetch_returns_empty_list(
        self, training_task, mock_codelist, mocker
    ):
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=[]
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=[],
        )
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        mock_train.assert_not_called()

    def test_calls_train_when_labeled_decisions_exist(
        self, training_task, mock_codelist, labeled_decisions, mock_config, mocker
    ):
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=labeled_decisions
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=labeled_decisions,
        )
        mocker.patch("src.task.training.get_config", return_value=mock_config)
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        mock_train.assert_called_once()

    def test_train_receives_at_most_ten_decisions(
        self, training_task, mock_codelist, mock_config, mocker
    ):
        """process() slices decisions[:10] before calling train()."""
        many_decisions = [
            {"decision": f"http://test.example.org/expr/{i}", "classes": ["Label"], "text": "t"}
            for i in range(15)
        ]
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=many_decisions
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=many_decisions,
        )
        mocker.patch("src.task.training.get_config", return_value=mock_config)
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        passed_decisions = mock_train.call_args[0][0]
        assert len(passed_decisions) == 10

    def test_train_receives_codelist_labels(
        self, training_task, mock_codelist, labeled_decisions, mock_config, mocker
    ):
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=labeled_decisions
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=labeled_decisions,
        )
        mocker.patch("src.task.training.get_config", return_value=mock_config)
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        passed_labels = mock_train.call_args[0][1]
        assert passed_labels == mock_codelist.get_labels()

    def test_train_receives_model_id_from_config(
        self, training_task, mock_codelist, labeled_decisions, mock_config, mocker
    ):
        mocker.patch.object(training_task, "fetch_codelist", return_value=mock_codelist)
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=labeled_decisions
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=labeled_decisions,
        )
        mocker.patch("src.task.training.get_config", return_value=mock_config)
        mock_train = mocker.patch("src.task.training.train")

        training_task.process()

        passed_model_id = mock_train.call_args[0][2]
        assert passed_model_id == "test-org/test-model"

    def test_fetch_codelist_is_called_once(
        self, training_task, mock_codelist, labeled_decisions, mock_config, mocker
    ):
        mock_fetch = mocker.patch.object(
            training_task, "fetch_codelist", return_value=mock_codelist
        )
        mocker.patch.object(
            training_task, "fetch_decisions_with_classes", return_value=labeled_decisions
        )
        mocker.patch.object(
            ClassifierTrainingTask, "convert_classes_to_original_names",
            return_value=labeled_decisions,
        )
        mocker.patch("src.task.training.get_config", return_value=mock_config)
        mocker.patch("src.task.training.train")

        training_task.process()

        mock_fetch.assert_called_once()
