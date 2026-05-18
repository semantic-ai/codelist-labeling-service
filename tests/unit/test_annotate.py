"""
Unit tests for ModelAnnotatingTask and ModelBatchAnnotatingTask
(src/task/annotate.py).

Test strategy
─────────────
Method / concern                         SPARQL              LLM
ModelAnnotatingTask.process()            real Virtuoso       mocked
ModelBatchAnnotatingTask
  .fetch_decisions_without_annotations() real Virtuoso       –
  .process()                             all mocked          mocked

ModelAnnotatingTask instances are constructed through the shortcut path
(source + codelist_entries supplied) to skip the DecisionTask SPARQL call.
get_config() and create_llm_client() are mocked at the module level so that
no real Ollama/OpenAI connection is attempted during __init__.

ModelBatchAnnotatingTask instances bypass __init__ entirely via object.__new__
because their parent DecisionTask.__init__ issues a SPARQL query we do not need.

Isolation
─────────
The autouse `isolate_test_graphs` fixture in conftest.py wipes all test-
relevant named graphs before and after every test.  This catches:
  • oa:Annotation triples written with random uuid4 URIs by
    LinkingAnnotation.add_to_triplestore_if_not_exists()
  • leftover triples from any test whose teardown failed mid-way
"""

import logging
from unittest.mock import MagicMock, call, patch

import helpers
import pytest
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import GRAPHS
from src.task.annotate import ModelAnnotatingTask, ModelBatchAnnotatingTask
from src.task.codelist import Codelist, CodelistEntry
from src.llm_models.llm_task_models import EntityLinkingTaskOutput
from src.config import AppConfig, LlmConfig

from tests.unit.conftest import (
    ANNOTATION_URI,
    CONCEPT_URI,
    CONCEPT_URI_2,
    CONCEPT_SCHEME_URI,
    EXPRESSION_CONTENT,
    EXPRESSION_URI,
    EXPRESSION_URI_2,
    NS,
    TASK_URI,
    sparql_ask,
    sparql_count_annotations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codelist(*entries: tuple[str, str]) -> Codelist:
    """Build a Codelist from (uri, label) tuples."""
    return Codelist([CodelistEntry(uri=u, label=l) for u, l in entries])


def _default_codelist() -> Codelist:
    return _make_codelist(
        (CONCEPT_URI, "Affordable and Clean Energy"),
        (CONCEPT_URI_2, "Climate Action"),
    )


def _make_config(provider: str = "ollama") -> AppConfig:
    return AppConfig(llm=LlmConfig(provider=provider, model_name="test-model"))


# ---------------------------------------------------------------------------
# ModelAnnotatingTask fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def annotating_task(mocker) -> ModelAnnotatingTask:
    """
    ModelAnnotatingTask with:
      • source = EXPRESSION_URI  (skips DecisionTask.__init__ SPARQL call)
      • default two-entry codelist
      • get_config() and create_llm_client() mocked at the module level
      • _llm is a MagicMock whose return_value can be set per test
    """
    mocker.patch("src.task.annotate.get_config", return_value=_make_config())
    mock_llm = MagicMock()
    mocker.patch("src.task.annotate.create_llm_client", return_value=mock_llm)

    task = ModelAnnotatingTask(
        task_uri=TASK_URI,
        source=EXPRESSION_URI,
        codelist_entries=_default_codelist(),
    )
    return task


# ---------------------------------------------------------------------------
# ModelBatchAnnotatingTask fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_task() -> ModelBatchAnnotatingTask:
    """
    ModelBatchAnnotatingTask created via object.__new__ to skip the
    DecisionTask SPARQL call.  All required attributes are set manually.
    """
    task = object.__new__(ModelBatchAnnotatingTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.batch_annotating_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    return task


# ─────────────────────────────────────────────────────────────────────────────
# ModelAnnotatingTask – process()
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAnnotatingTaskProcess:

    def test_inserts_annotation_when_llm_matches_single_class(
        self, annotating_task, expression_content_triple
    ):
        """A single matching label returned by the LLM produces one annotation."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy"]
        )

        annotating_task.process()

        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    ?ann a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                         {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                         {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                         {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI)} .
                }}
            }}
        """)

    def test_inserts_one_annotation_per_matched_class(
        self, annotating_task, expression_content_triple
    ):
        """Two matching labels produce two separate annotations."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy", "Climate Action"]
        )

        annotating_task.process()

        count = sparql_count_annotations(EXPRESSION_URI)
        assert count == 2

    def test_annotation_targets_source_expression(
        self, annotating_task, expression_content_triple
    ):
        """The oa:hasTarget of every inserted annotation is the task's source URI."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy"]
        )

        annotating_task.process()

        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    ?ann {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} .
                }}
            }}
        """)

    def test_annotation_body_is_resolved_concept_uri(
        self, annotating_task, expression_content_triple
    ):
        """The oa:hasBody of the annotation is the resolved SKOS concept URI."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Climate Action"]
        )

        annotating_task.process()

        assert sparql_ask(f"""
            ASK {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    ?ann a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                         {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI_2)} .
                }}
            }}
        """)

    def test_does_not_duplicate_annotation_on_repeated_call(
        self, annotating_task, expression_content_triple
    ):
        """Calling process() twice produces exactly one annotation (idempotent guard)."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy"]
        )

        annotating_task.process()
        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 1

    def test_skips_llm_call_when_no_expression_content(self, annotating_task):
        """
        When the source expression has no content in the triplestore, process()
        logs a warning and returns without calling the LLM.
        """
        annotating_task.process()

        annotating_task._llm.assert_not_called()
        assert sparql_count_annotations(EXPRESSION_URI) == 0

    def test_inserts_no_annotation_when_no_expression_content(
        self, annotating_task
    ):
        """No annotation is created when fetch_data() returns empty text."""
        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 0

    def test_skips_llm_call_when_codelist_is_empty(
        self, annotating_task, expression_content_triple, mocker
    ):
        """
        When the codelist has no entries, process() logs an error and skips
        the LLM call.
        """
        annotating_task._codelist_entries = Codelist([])
        annotating_task._label_to_uri = {}

        annotating_task.process()

        annotating_task._llm.assert_not_called()

    def test_inserts_no_annotation_when_codelist_is_empty(
        self, annotating_task, expression_content_triple
    ):
        annotating_task._codelist_entries = Codelist([])
        annotating_task._label_to_uri = {}

        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 0

    def test_skips_annotation_when_llm_returns_unknown_class(
        self, annotating_task, expression_content_triple
    ):
        """A label not present in the codelist is silently ignored."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Unknown SDG Goal 99"]
        )

        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 0

    def test_skips_annotation_when_llm_returns_empty_list(
        self, annotating_task, expression_content_triple
    ):
        """An empty designated_classes list results in zero annotations."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=[]
        )

        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 0

    def test_falls_back_to_random_label_when_llm_raises(
        self, annotating_task, expression_content_triple
    ):
        """
        When the LLM call raises an exception, process() falls back to a
        randomly selected label from the codelist and still inserts one
        annotation.
        """
        annotating_task._llm.side_effect = RuntimeError("LLM unavailable")

        annotating_task.process()

        # Exactly one annotation with some concept from the codelist
        count = sparql_count_annotations(EXPRESSION_URI)
        assert count == 1

    def test_llm_receives_codelist_labels_in_user_message(
        self, annotating_task, expression_content_triple
    ):
        """The user message passed to the LLM contains the codelist labels."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=[]
        )

        annotating_task.process()

        annotating_task._llm.assert_called_once()
        llm_input = annotating_task._llm.call_args[0][0]
        assert "Affordable and Clean Energy" in llm_input.user_message
        assert "Climate Action" in llm_input.user_message

    def test_llm_receives_expression_content_in_user_message(
        self, annotating_task, expression_content_triple
    ):
        """The user message passed to the LLM contains the fetched expression text."""
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=[]
        )

        annotating_task.process()

        llm_input = annotating_task._llm.call_args[0][0]
        assert EXPRESSION_CONTENT in llm_input.user_message


# ─────────────────────────────────────────────────────────────────────────────
# ModelBatchAnnotatingTask – fetch_decisions_without_annotations()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchDecisionsWithoutAnnotations:

    def test_returns_unannotated_expression(
        self, batch_task, unannotated_expressions
    ):
        """Both expressions appear when neither has an annotation."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 in result

    def test_returns_only_unannotated_when_one_is_annotated(
        self, batch_task, one_annotated_one_plain_expression
    ):
        """Only the expression without an annotation is returned."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 not in result

    def test_excludes_expression_with_classifying_annotation(
        self, batch_task, two_annotated_expressions
    ):
        """Neither expression is returned when both have classifying annotations."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert EXPRESSION_URI not in result
        assert EXPRESSION_URI_2 not in result

    def test_returns_empty_list_when_no_expressions(self, batch_task):
        """Returns an empty list when no eli:Expression triples exist."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert result == []

    def test_returns_list_of_strings(
        self, batch_task, unannotated_expressions
    ):
        """Every returned value is a plain string (the expression URI)."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert all(isinstance(uri, str) for uri in result)

    def test_ignores_non_classifying_annotations(
        self, batch_task, unannotated_expressions, mocker
    ):
        """
        An annotation whose oa:motivatedBy is NOT oa:classifying does not
        count as annotated – the expression must still be returned.
        """
        # Insert an annotation with a different motivation
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                    {sparql_escape_uri(ANNOTATION_URI)}
                        a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                        {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                        {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "linking")} .
                }}
            }}
        """)

        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI
        )

        assert EXPRESSION_URI in result

    def test_filters_by_target_node(
        self, batch_task, unannotated_expressions
    ):
        """When sh:targetNode is used, only the specified nodes are returned."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI,
            target_nodes=[EXPRESSION_URI],
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 not in result

    def test_filters_by_target_class(
        self, batch_task, unannotated_expressions
    ):
        """When sh:targetClass is used, all instances of that class are returned."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI,
            target_classes=["http://data.europa.eu/eli/ontology#Expression"],
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 in result

    def test_filters_by_target_node_and_class(
        self, batch_task, unannotated_expressions
    ):
        """When both sh:targetNode and sh:targetClass are used, results are the union."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI,
            target_nodes=[EXPRESSION_URI],
            target_classes=["http://data.europa.eu/eli/ontology#Expression"],
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 in result

    def test_defaults_to_eli_expression_when_no_shapes(
        self, batch_task, unannotated_expressions
    ):
        """Without shape targets, defaults to eli:Expression (same as test_returns_unannotated_expression)."""
        result = ModelBatchAnnotatingTask.fetch_decisions_without_annotations(
            concept_scheme_uri=CONCEPT_SCHEME_URI,
        )

        assert EXPRESSION_URI in result
        assert EXPRESSION_URI_2 in result


# ─────────────────────────────────────────────────────────────────────────────
# ModelBatchAnnotatingTask – process()  (orchestration, all mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelBatchAnnotatingTaskProcess:

    def _mock_batch_dependencies(self, batch_task, mocker):
        """Mock the common dependencies for batch process tests."""
        mocker.patch.object(batch_task, "get_target_graph", return_value=None)
        mocker.patch.object(batch_task, "fetch_shape_targets", return_value=([], []))
        mocker.patch.object(batch_task, "fetch_property_path_for_text", return_value=None)

    def test_calls_annotating_task_for_each_decision(
        self, batch_task, mocker
    ):
        """process() creates and runs a ModelAnnotatingTask per unannotated decision."""
        mock_codelist = _default_codelist()
        mocker.patch.object(batch_task, "fetch_codelist", return_value=mock_codelist)
        self._mock_batch_dependencies(batch_task, mocker)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[EXPRESSION_URI, EXPRESSION_URI_2],
        )
        mock_annotating_process = mocker.patch.object(
            ModelAnnotatingTask, "process", return_value=None
        )
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        batch_task.process()

        assert mock_annotating_process.call_count == 2

    def test_skips_when_no_unannotated_decisions(self, batch_task, mocker):
        """process() exits immediately when there are no decisions to annotate."""
        mock_codelist = _default_codelist()
        mocker.patch.object(batch_task, "fetch_codelist", return_value=mock_codelist)
        self._mock_batch_dependencies(batch_task, mocker)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[],
        )
        mock_annotating_process = mocker.patch.object(
            ModelAnnotatingTask, "process", return_value=None
        )
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        batch_task.process()

        mock_annotating_process.assert_not_called()

    def test_passes_same_codelist_to_every_annotating_task(
        self, batch_task, mocker
    ):
        """
        The codelist is fetched once and reused for all ModelAnnotatingTask
        instances – no per-decision codelist fetch.
        """
        mock_codelist = _default_codelist()
        mock_fetch_codelist = mocker.patch.object(
            batch_task, "fetch_codelist", return_value=mock_codelist
        )
        self._mock_batch_dependencies(batch_task, mocker)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[EXPRESSION_URI, EXPRESSION_URI_2],
        )
        mocker.patch.object(ModelAnnotatingTask, "process", return_value=None)
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        batch_task.process()

        # fetch_codelist must have been called exactly once
        mock_fetch_codelist.assert_called_once()

    def test_passes_task_uri_to_each_annotating_task(
        self, batch_task, mocker
    ):
        """Each ModelAnnotatingTask is initialised with the batch task's own URI."""
        mocker.patch.object(batch_task, "fetch_codelist", return_value=_default_codelist())
        self._mock_batch_dependencies(batch_task, mocker)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[EXPRESSION_URI],
        )
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        created_tasks: list[ModelAnnotatingTask] = []

        original_init = ModelAnnotatingTask.__init__

        def capture_init(self, task_uri, source=None, codelist_entries=None, property_path_for_text=None):
            original_init(self, task_uri, source=source, codelist_entries=codelist_entries, property_path_for_text=property_path_for_text)
            created_tasks.append(self)

        mocker.patch.object(ModelAnnotatingTask, "__init__", capture_init)
        mocker.patch.object(ModelAnnotatingTask, "process", return_value=None)

        batch_task.process()

        assert len(created_tasks) == 1
        assert created_tasks[0].task_uri == TASK_URI

    def test_passes_decision_uri_as_source_to_each_annotating_task(
        self, batch_task, mocker
    ):
        """Each ModelAnnotatingTask receives the decision URI as its source."""
        mocker.patch.object(batch_task, "fetch_codelist", return_value=_default_codelist())
        self._mock_batch_dependencies(batch_task, mocker)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[EXPRESSION_URI],
        )
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        created_tasks: list[ModelAnnotatingTask] = []

        original_init = ModelAnnotatingTask.__init__

        def capture_init(self, task_uri, source=None, codelist_entries=None, property_path_for_text=None):
            original_init(self, task_uri, source=source, codelist_entries=codelist_entries, property_path_for_text=property_path_for_text)
            created_tasks.append(self)

        mocker.patch.object(ModelAnnotatingTask, "__init__", capture_init)
        mocker.patch.object(ModelAnnotatingTask, "process", return_value=None)

        batch_task.process()

        assert created_tasks[0].source == EXPRESSION_URI

    def test_passes_property_path_to_each_annotating_task(
        self, batch_task, mocker
    ):
        """Each ModelAnnotatingTask receives the property_path_for_text from the job."""
        property_path = "https://data.europarl.europa.eu/def/epvoc#expressionContent"
        mocker.patch.object(batch_task, "fetch_codelist", return_value=_default_codelist())
        mocker.patch.object(batch_task, "get_target_graph", return_value=None)
        mocker.patch.object(batch_task, "fetch_shape_targets", return_value=([], []))
        mocker.patch.object(batch_task, "fetch_property_path_for_text", return_value=property_path)
        mocker.patch(
            "src.task.annotate.ModelBatchAnnotatingTask.fetch_decisions_without_annotations",
            return_value=[EXPRESSION_URI],
        )
        mocker.patch("src.task.annotate.get_config", return_value=_make_config())
        mocker.patch("src.task.annotate.create_llm_client", return_value=MagicMock())

        created_tasks: list[ModelAnnotatingTask] = []

        original_init = ModelAnnotatingTask.__init__

        def capture_init(self, task_uri, source=None, codelist_entries=None, property_path_for_text=None):
            original_init(self, task_uri, source=source, codelist_entries=codelist_entries, property_path_for_text=property_path_for_text)
            created_tasks.append(self)

        mocker.patch.object(ModelAnnotatingTask, "__init__", capture_init)
        mocker.patch.object(ModelAnnotatingTask, "process", return_value=None)

        batch_task.process()

        assert created_tasks[0]._property_path_for_text == property_path


# ─────────────────────────────────────────────────────────────────────────────
# ModelAnnotatingTask – fetch_text_with_property_path()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchTextWithPropertyPath:

    def test_fetches_text_using_configured_property(
        self, annotating_task, expression_content_triple
    ):
        """fetch_text_with_property_path() returns the text stored under the given property."""
        epvoc_content = "https://data.europarl.europa.eu/def/epvoc#expressionContent"
        result = annotating_task.fetch_text_with_property_path(epvoc_content)

        assert EXPRESSION_CONTENT in result

    def test_returns_empty_string_when_property_not_present(
        self, annotating_task, expression_content_triple
    ):
        """Returns empty string when the property does not exist on the resource."""
        result = annotating_task.fetch_text_with_property_path(
            "http://example.org/nonexistent-property"
        )

        assert result == ""

    def test_process_uses_property_path_when_set(
        self, annotating_task, expression_content_triple
    ):
        """process() uses fetch_text_with_property_path when _property_path_for_text is set."""
        epvoc_content = "https://data.europarl.europa.eu/def/epvoc#expressionContent"
        annotating_task._property_path_for_text = epvoc_content
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy"]
        )

        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 1

    def test_process_falls_back_to_fetch_data_when_no_property_path(
        self, annotating_task, expression_content_triple
    ):
        """process() falls back to fetch_data() when _property_path_for_text is None."""
        annotating_task._property_path_for_text = None
        annotating_task._llm.return_value = EntityLinkingTaskOutput(
            designated_classes=["Affordable and Clean Energy"]
        )

        annotating_task.process()

        assert sparql_count_annotations(EXPRESSION_URI) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CodeListTask – fetch_property_path_for_text()  (SPARQL injection protection)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchPropertyPathForText:

    def test_returns_none_when_not_configured(self, batch_task):
        """Returns None when job has no ext:propertyPathForText."""
        result = batch_task.fetch_property_path_for_text()
        assert result is None

    def test_returns_uri_when_configured(self, batch_task):
        """Returns the URI when ext:propertyPathForText is set on the job."""
        epvoc_content = "https://data.europarl.europa.eu/def/epvoc#expressionContent"
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                    {sparql_escape_uri(TASK_URI)}
                        <http://purl.org/dc/terms/isPartOf> <http://test.example.org/jobs/job-1> .
                    <http://test.example.org/jobs/job-1>
                        <http://mu.semte.ch/vocabularies/ext/propertyPathForText> {sparql_escape_uri(epvoc_content)} .
                }}
            }}
        """)

        result = batch_task.fetch_property_path_for_text()
        assert result == epvoc_content


# ─────────────────────────────────────────────────────────────────────────────
# CodeListTask – fetch_shape_targets()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchShapeTargets:

    def test_returns_empty_lists_when_no_shapes(self, batch_task):
        """Returns empty lists when job has no ext:shapeForTargets."""
        target_nodes, target_classes = batch_task.fetch_shape_targets()
        assert target_nodes == []
        assert target_classes == []

    def test_returns_target_nodes(self, batch_task):
        """Returns target nodes from sh:targetNode on shapes."""
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                    {sparql_escape_uri(TASK_URI)}
                        <http://purl.org/dc/terms/isPartOf> <http://test.example.org/jobs/job-1> .
                    <http://test.example.org/jobs/job-1>
                        <http://mu.semte.ch/vocabularies/ext/shapeForTargets> <http://test.example.org/shapes/shape-1> .
                    <http://test.example.org/shapes/shape-1>
                        <http://www.w3.org/ns/shacl#targetNode> {sparql_escape_uri(EXPRESSION_URI)} .
                }}
            }}
        """)

        target_nodes, target_classes = batch_task.fetch_shape_targets()
        assert EXPRESSION_URI in target_nodes
        assert target_classes == []

    def test_returns_target_classes(self, batch_task):
        """Returns target classes from sh:targetClass on shapes."""
        eli_expression = "http://data.europa.eu/eli/ontology#Expression"
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                    {sparql_escape_uri(TASK_URI)}
                        <http://purl.org/dc/terms/isPartOf> <http://test.example.org/jobs/job-1> .
                    <http://test.example.org/jobs/job-1>
                        <http://mu.semte.ch/vocabularies/ext/shapeForTargets> <http://test.example.org/shapes/shape-1> .
                    <http://test.example.org/shapes/shape-1>
                        <http://www.w3.org/ns/shacl#targetClass> {sparql_escape_uri(eli_expression)} .
                }}
            }}
        """)

        target_nodes, target_classes = batch_task.fetch_shape_targets()
        assert target_nodes == []
        assert eli_expression in target_classes

    def test_returns_both_from_multiple_shapes(self, batch_task):
        """Returns both target nodes and classes when multiple shapes are configured."""
        eli_expression = "http://data.europa.eu/eli/ontology#Expression"
        helpers.update(f"""
            INSERT DATA {{
                GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                    {sparql_escape_uri(TASK_URI)}
                        <http://purl.org/dc/terms/isPartOf> <http://test.example.org/jobs/job-1> .
                    <http://test.example.org/jobs/job-1>
                        <http://mu.semte.ch/vocabularies/ext/shapeForTargets>
                            <http://test.example.org/shapes/shape-1> ,
                            <http://test.example.org/shapes/shape-2> .
                    <http://test.example.org/shapes/shape-1>
                        <http://www.w3.org/ns/shacl#targetClass> {sparql_escape_uri(eli_expression)} .
                    <http://test.example.org/shapes/shape-2>
                        <http://www.w3.org/ns/shacl#targetNode> {sparql_escape_uri(EXPRESSION_URI)} .
                }}
            }}
        """)

        target_nodes, target_classes = batch_task.fetch_shape_targets()
        assert EXPRESSION_URI in target_nodes
        assert eli_expression in target_classes
