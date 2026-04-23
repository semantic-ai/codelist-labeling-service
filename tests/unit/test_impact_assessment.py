"""
Unit tests for the ImpactAssessment task (src/task/impact.py).

Test strategy
─────────────
Method                      SPARQL          LLM
fetch_eli_expressions       real Virtuoso   –
fetch_policy_labels         real Virtuoso   –  (fetch_codelist_uri_for_task mocked)
_process_single             –               mocked
store                       real Virtuoso   –
process                     all mocked      mocked  (orchestration only)

The ImpactAssessmentTask instance is always created via object.__new__ so
that __init__ (which makes SPARQL calls) is bypassed.  All required
attributes are set manually by the `impact_task` fixture in conftest.py.

Previously documented bugs (now fixed)
───────────────────────────────────────
1. Class-name collision: resolved by renaming the task class to
   ImpactAssessmentTask.  ImpactAssessmentOutput in conftest.py still
   re-declares the Pydantic schema separately for use in fixtures.

2. process() was calling `policy_label.uri` instead of
   `annotation_uri` — fixed.  TestProcess.test_store_is_called_with_annotation_uri
   now asserts the correct attribute is used.
"""

import pytest
from unittest.mock import patch
from escape_helpers import sparql_escape_uri
from langchain_core.messages import HumanMessage, SystemMessage

from decide_ai_service_base.sparql_config import GRAPHS

from src.task.impact import (
    ImpactAssessmentTask,
    ImpactAssessment,
    ConfidenceLevel,
    ImpactDirection,
    PolicyLabel,
    ProcessItem,
)

from tests.unit.conftest import (
    ANNOTATION_URI,
    CONCEPT_SCHEME_URI,
    CONCEPT_URI,
    EXPRESSION_CONTENT,
    EXPRESSION_URI,
    LANGUAGE_URI,
    NS,
    TASK_URI,
    WORK_URI,
    ImpactAssessmentOutput,
    sparql_ask,
    sparql_count,
)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_eli_expressions
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchEliExpressions:
    """fetch_eli_expressions() queries four named graphs and assembles ProcessItems."""

    def test_returns_one_process_item_per_expression(
        self, impact_task, expression_triples
    ):
        results = impact_task.fetch_eli_expressions()

        assert len(results) == 1

    def test_process_item_carries_correct_expression_uri(
        self, impact_task, expression_triples
    ):
        item = impact_task.fetch_eli_expressions()[0]

        assert item.expression_uri == EXPRESSION_URI

    def test_process_item_carries_expression_content(
        self, impact_task, expression_triples
    ):
        item = impact_task.fetch_eli_expressions()[0]

        assert item.expression_content == EXPRESSION_CONTENT

    def test_process_item_carries_language_uri(
        self, impact_task, expression_triples
    ):
        item = impact_task.fetch_eli_expressions()[0]

        assert item.language == LANGUAGE_URI

    def test_process_item_carries_work_uri(
        self, impact_task, expression_triples
    ):
        item = impact_task.fetch_eli_expressions()[0]

        assert item.work_uri == WORK_URI

    def test_returns_empty_list_when_task_has_no_input(self, impact_task):
        """Without any triples in the store the method must return [] gracefully."""
        results = impact_task.fetch_eli_expressions()

        assert results == []

    def test_returns_list_of_process_items(self, impact_task, expression_triples):
        results = impact_task.fetch_eli_expressions()

        assert all(isinstance(r, ProcessItem) for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_policy_labels
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchPolicyLabels:
    """
    fetch_policy_labels() looks up OA annotations targeting an expression and
    joins them with SKOS concepts in the public graph.

    fetch_codelist_uri_for_task() is mocked throughout this class because it
    queries without a named-graph clause, which is not reliable against a
    plain Virtuoso endpoint without graph-union semantics configured.
    """

    def test_returns_one_policy_label_per_annotation(
        self, impact_task, annotation_triples, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        results = impact_task.fetch_policy_labels(EXPRESSION_URI)

        assert len(results) == 1

    def test_policy_label_carries_annotation_uri(
        self, impact_task, annotation_triples, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        label = impact_task.fetch_policy_labels(EXPRESSION_URI)[0]

        assert label.annotation_uri == ANNOTATION_URI

    def test_policy_label_carries_concept_uri(
        self, impact_task, annotation_triples, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        label = impact_task.fetch_policy_labels(EXPRESSION_URI)[0]

        assert label.policy_concept_uri == CONCEPT_URI

    def test_policy_label_carries_pref_label(
        self, impact_task, annotation_triples, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        label = impact_task.fetch_policy_labels(EXPRESSION_URI)[0]

        assert label.policy_label == "Affordable and Clean Energy"

    def test_returns_empty_list_when_no_annotations_exist(
        self, impact_task, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        results = impact_task.fetch_policy_labels(EXPRESSION_URI)

        assert results == []

    def test_delegates_codelist_resolution_to_helper(
        self, impact_task, annotation_triples, mocker
    ):
        mock_codelist = mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        impact_task.fetch_policy_labels(EXPRESSION_URI)

        mock_codelist.assert_called_once_with()

    def test_returns_list_of_policy_label_objects(
        self, impact_task, annotation_triples, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_codelist_uri_for_task", return_value=CONCEPT_SCHEME_URI
        )

        results = impact_task.fetch_policy_labels(EXPRESSION_URI)

        assert all(isinstance(r, PolicyLabel) for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# _process_single
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessSingle:
    """_process_single() builds LLM messages and returns the model's response."""

    def test_calls_llm_invoke_once(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        impact_task._process_single(sample_process_item, sample_policy_label)

        impact_task.llm.invoke.assert_called_once()

    def test_passes_system_and_human_messages(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        impact_task._process_single(sample_process_item, sample_policy_label)

        messages = impact_task.llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_human_message_contains_expression_content(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        impact_task._process_single(sample_process_item, sample_policy_label)

        human_msg = impact_task.llm.invoke.call_args[0][0][1]
        assert sample_process_item.expression_content in human_msg.content

    def test_human_message_contains_policy_label(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        impact_task._process_single(sample_process_item, sample_policy_label)

        human_msg = impact_task.llm.invoke.call_args[0][0][1]
        assert sample_policy_label.policy_label in human_msg.content

    def test_returns_llm_output_unchanged(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        result = impact_task._process_single(sample_process_item, sample_policy_label)

        assert result is sample_assessment

    def test_preserves_positive_impact_direction(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        assert sample_assessment.impact_direction == ImpactDirection.POSITIVE
        impact_task.llm.invoke.return_value = sample_assessment

        result = impact_task._process_single(sample_process_item, sample_policy_label)

        assert result.impact_direction == ImpactDirection.POSITIVE

    def test_preserves_negative_impact_direction(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        negative_assessment = sample_assessment.model_copy(
            update={
                "impact_direction": ImpactDirection.NEGATIVE,
                "confidence": ConfidenceLevel.LOW,
                "summary": "The policy negatively affects clean energy access.",
            }
        )
        impact_task.llm.invoke.return_value = negative_assessment

        result = impact_task._process_single(sample_process_item, sample_policy_label)

        assert result.impact_direction == ImpactDirection.NEGATIVE

    def test_preserves_uncertain_impact_direction(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        uncertain_assessment = sample_assessment.model_copy(
            update={
                "impact_direction": ImpactDirection.UNCERTAIN,
                "confidence": ConfidenceLevel.LOW,
            }
        )
        impact_task.llm.invoke.return_value = uncertain_assessment

        result = impact_task._process_single(sample_process_item, sample_policy_label)

        assert result.impact_direction == ImpactDirection.UNCERTAIN

    def test_preserves_all_assessment_fields(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
    ):
        impact_task.llm.invoke.return_value = sample_assessment

        result = impact_task._process_single(sample_process_item, sample_policy_label)

        assert result.label == sample_assessment.label
        assert result.confidence == sample_assessment.confidence
        assert result.reasoning == sample_assessment.reasoning
        assert result.direct_effects == sample_assessment.direct_effects
        assert result.second_order_effects == sample_assessment.second_order_effects
        assert result.key_uncertainties == sample_assessment.key_uncertainties
        assert result.summary == sample_assessment.summary


# ─────────────────────────────────────────────────────────────────────────────
# store
# ─────────────────────────────────────────────────────────────────────────────

class TestStore:
    """store() writes ext:has_impact to the AI named graph via SPARQL UPDATE."""

    def test_inserts_impact_direction_triple(
        self, impact_task, bare_annotation_triple, sample_assessment
    ):
        impact_task.store(ANNOTATION_URI, sample_assessment)

        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    {sparql_escape_uri(ANNOTATION_URI)}
                        {sparql_escape_uri(NS["oa"] + "hasBody")}
                        {sparql_escape_uri("http://mu.semte.ch/vocabularies/ext/impact/positive")} .
                }}
            }}
        """)

    def test_impact_direction_value_is_string_literal(
        self, impact_task, bare_annotation_triple, sample_assessment
    ):
        """The stored value matches the enum's .value, not the enum name."""
        impact_task.store(ANNOTATION_URI, sample_assessment)

        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    {sparql_escape_uri(ANNOTATION_URI)} {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri("http://mu.semte.ch/vocabularies/ext/impact/positive")}  .
                }}
            }}
        """)

    def test_does_not_insert_duplicate_on_repeated_call(
        self, impact_task, bare_annotation_triple, sample_assessment
    ):
        """The FILTER NOT EXISTS guard in store() prevents duplicates."""
        impact_task.store(ANNOTATION_URI, sample_assessment)
        impact_task.store(ANNOTATION_URI, sample_assessment)

        count = sparql_count(GRAPHS["ai"], ANNOTATION_URI, f"{NS['oa']}hasBody")
        assert count == 1

    def test_raises_runtime_error_when_update_fails(
        self, impact_task, sample_assessment, mocker
    ):
        """SPARQL errors are wrapped in RuntimeError with a descriptive message."""
        mocker.patch(
            "src.task.impact.update",
            side_effect=Exception("connection refused"),
        )

        with pytest.raises(RuntimeError, match="Failed to insert impact"):
            impact_task.store(ANNOTATION_URI, sample_assessment)

    def test_runtime_error_preserves_original_cause(
        self, impact_task, sample_assessment, mocker
    ):
        original = Exception("timeout")
        mocker.patch("src.task.impact.update", side_effect=original)

        with pytest.raises(RuntimeError) as exc_info:
            impact_task.store(ANNOTATION_URI, sample_assessment)

        assert exc_info.value.__cause__ is original


# ─────────────────────────────────────────────────────────────────────────────
# process  (orchestration – all collaborators mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestProcess:
    """
    process() is tested with all collaborator methods mocked so that the
    routing logic can be verified without live data or a real LLM.
    """

    def test_skips_store_when_no_expressions(self, impact_task, mocker):
        mocker.patch.object(impact_task, "fetch_eli_expressions", return_value=[])
        mock_store = mocker.patch.object(impact_task, "store")

        impact_task.process()

        mock_store.assert_not_called()

    def test_skips_store_when_expression_has_no_policy_labels(
        self, impact_task, sample_process_item, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_eli_expressions", return_value=[sample_process_item]
        )
        mocker.patch.object(impact_task, "fetch_policy_labels", return_value=[])
        mock_store = mocker.patch.object(impact_task, "store")

        impact_task.process()

        mock_store.assert_not_called()

    def test_calls_fetch_policy_labels_with_expression_uri(
        self, impact_task, sample_process_item, mocker
    ):
        mocker.patch.object(
            impact_task, "fetch_eli_expressions", return_value=[sample_process_item]
        )
        mock_labels = mocker.patch.object(
            impact_task, "fetch_policy_labels", return_value=[]
        )

        impact_task.process()

        mock_labels.assert_called_once_with(EXPRESSION_URI)

    def test_calls_process_single_for_each_label(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
        mocker,
    ):
        mocker.patch.object(
            impact_task, "fetch_eli_expressions", return_value=[sample_process_item]
        )
        mocker.patch.object(
            impact_task, "fetch_policy_labels", return_value=[sample_policy_label]
        )
        mock_single = mocker.patch.object(
            impact_task, "_process_single", return_value=sample_assessment
        )
        mocker.patch.object(impact_task, "store")

        impact_task.process()

        mock_single.assert_called_once_with(sample_process_item, sample_policy_label)

    def test_store_is_called_with_annotation_uri(
        self,
        impact_task,
        sample_process_item,
        sample_policy_label,
        sample_assessment,
        mocker,
    ):
        """process() passes policy_label.annotation_uri (not .uri) to store()."""
        mocker.patch.object(
            impact_task, "fetch_eli_expressions", return_value=[sample_process_item]
        )
        mocker.patch.object(
            impact_task, "fetch_policy_labels", return_value=[sample_policy_label]
        )
        mocker.patch.object(
            impact_task, "_process_single", return_value=sample_assessment
        )
        mock_store = mocker.patch.object(impact_task, "store")

        impact_task.process()

        mock_store.assert_called_once_with(
            sample_policy_label.annotation_uri, sample_assessment
        )
