"""
Unit-test fixtures shared across all task test modules.

Responsibilities:
  • declare canonical test-data URIs / literals
  • provide low-level SPARQL helpers that talk directly to Virtuoso
  • expose an autouse isolation fixture that guarantees a clean triplestore
    state before every test (catches both failed-teardown leftovers and
    annotations with unpredictable UUIDs written by add_to_triplestore_if_not_exists)
  • expose per-test data fixtures that insert/remove triples around each test
  • expose re-declared Pydantic output models for tasks that shadow theirs
  • expose task-instance fixtures with mocked LLMs
"""

import logging
from unittest.mock import MagicMock

import helpers
import pytest
import requests  # kept only for requests.exceptions.ConnectionError
from escape_helpers import sparql_escape_uri
from pydantic import BaseModel, Field

from decide_ai_service_base.sparql_config import GRAPHS
from src.task.impact import (
    ImpactAssessmentTask,
    ConfidenceLevel,
    ImpactDirection,
    PolicyLabel,
    ProcessItem,
)

# ---------------------------------------------------------------------------
# Canonical test-data URIs / strings
# ---------------------------------------------------------------------------

TASK_URI = "http://test.example.org/tasks/impact-task-1"
JOB_URI = "http://test.example.org/jobs/job-1"
CONTAINER_URI = "http://test.example.org/containers/container-1"
EXPRESSION_URI = "http://test.example.org/expressions/expression-1"
EXPRESSION_URI_2 = "http://test.example.org/expressions/expression-2"
WORK_URI = "http://test.example.org/works/work-1"
CONCEPT_SCHEME_URI = "http://test.example.org/codelists/sdg-goals"
CONCEPT_URI = "http://test.example.org/concepts/sdg-goal-1"
CONCEPT_URI_2 = "http://test.example.org/concepts/sdg-goal-2"
ANNOTATION_URI = "http://test.example.org/annotations/annotation-1"
LANGUAGE_URI = "http://publications.europa.eu/resource/authority/language/ENG"
EXPRESSION_CONTENT = (
    "This policy promotes targeted investment in renewable energy infrastructure "
    "to reduce carbon emissions, lower household energy costs, and create "
    "sustainable employment in the clean-energy sector."
)
EXPRESSION_CONTENT_2 = (
    "This policy introduces mandatory recycling targets for industrial waste "
    "and establishes penalties for non-compliance."
)

# Full-form namespace URIs used in INSERT DATA statements (no PREFIX aliases)
NS = {
    "task": "http://redpencil.data.gift/vocabularies/tasks/",
    "nfo": "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#",
    "eli": "http://data.europa.eu/eli/ontology#",
    "epvoc": "https://data.europarl.europa.eu/def/epvoc#",
    "oa": "http://www.w3.org/ns/oa#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "ext": "http://mu.semte.ch/vocabularies/ext/",
    "prov": "http://www.w3.org/ns/prov#",
}

# All named graphs that tests write to – used by the isolation fixture
_TEST_GRAPHS = [
    GRAPHS["jobs"],
    GRAPHS["data_containers"],
    GRAPHS["expressions"],
    GRAPHS["works"],
    GRAPHS["ai"],
    GRAPHS["public"],
]

# ---------------------------------------------------------------------------
# SPARQL helpers
# All calls go through helpers.update() / helpers.query() so that the same
# HTTP method and endpoint configuration is used everywhere.
# ---------------------------------------------------------------------------

def sparql_update(query_str: str) -> None:
    """Execute a SPARQL update via helpers.update()."""
    helpers.update(query_str)


def sparql_ask(query_str: str) -> bool:
    """Run an ASK query via helpers.query(); return the boolean result."""
    return helpers.query(query_str).get("boolean", False)


def sparql_count(graph: str, subject: str, predicate: str) -> int:
    """Return the number of triples matching (subject, predicate, *) in graph."""
    result = helpers.query(
        f"SELECT (COUNT(?o) AS ?n) WHERE {{"
        f" GRAPH {sparql_escape_uri(graph)} {{ {sparql_escape_uri(subject)} {sparql_escape_uri(predicate)} ?o }} }}"
    )
    return int(result["results"]["bindings"][0]["n"]["value"])


def sparql_count_annotations(expression_uri: str) -> int:
    """Count oa:Annotation triples in the AI graph that target expression_uri."""
    result = helpers.query(
        f"SELECT (COUNT(?ann) AS ?n) WHERE {{"
        f" GRAPH {sparql_escape_uri(GRAPHS['ai'])} {{"
        f"   ?ann a {sparql_escape_uri(NS['oa'] + 'Annotation')} ;"
        f"        {sparql_escape_uri(NS['oa'] + 'hasTarget')} {sparql_escape_uri(expression_uri)} ."
        f" }}"
        f"}}"
    )
    return int(result["results"]["bindings"][0]["n"]["value"])


# ---------------------------------------------------------------------------
# Isolation fixture  (autouse – runs before every test)
#
# WHY THIS IS NECESSARY
# ─────────────────────
# 1. add_to_triplestore_if_not_exists() writes oa:Annotation triples with
#    random uuid4 URIs — targeted subject-URI deletes can never reach them.
# 2. If a test fails before its yield, leftover triples remain and pollute
#    the next test.
#
# STRATEGY
# ────────
# Each test graph is wiped wholesale with a plain DELETE WHERE.
# A VALUES-based per-subject filter was tried but Virtuoso rejects it with
# 400 inside DELETE WHERE { GRAPH <g> { VALUES … } }, so the simple
# full-graph wipe is used instead.
#
# NOTE: assumes the Virtuoso endpoint is dedicated to testing.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_test_graphs():
    """Guarantee a clean triplestore state before and after every test.

    If Virtuoso is unreachable the fixture yields silently so that mock-only
    tests can still run.  Tests that genuinely need the triplestore will fail
    on their own data-insertion fixtures with a clear ConnectionError.
    """
    try:
        _wipe_test_data()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
        yield
        return
    yield
    try:
        _wipe_test_data()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
        pass


def _wipe_test_data() -> None:
    # Clear every test graph entirely with one DELETE per graph.
    # A VALUES-based filter looks elegant but Virtuoso rejects it with 400
    # inside DELETE WHERE { GRAPH <g> { VALUES ... } }, causing the isolation
    # fixture to swallow the error and leave stale triples for the next test.
    for graph in _TEST_GRAPHS:
        sparql_update(
            f"DELETE WHERE {{ GRAPH {sparql_escape_uri(graph)} {{ ?s ?p ?o }} }}"
        )


# ---------------------------------------------------------------------------
# ImpactAssessmentOutput – re-declaration of the Pydantic output schema
#
# impact.py defines `class ImpactAssessment(BaseModel)` and then redefines
# `class ImpactAssessment(CodeListTask)`, so the Pydantic model is shadowed
# at module level.  We recreate it here for use in fixtures and assertions.
# ---------------------------------------------------------------------------

class ImpactAssessmentOutput(BaseModel):
    """Mirrors the first ImpactAssessment(BaseModel) declared in impact.py."""

    label: str = Field(description="The classification label being assessed")
    impact_direction: ImpactDirection = Field(description="Overall direction of the policy impact")
    confidence: ConfidenceLevel = Field(description="Confidence level in this assessment")
    reasoning: str = Field(description="Step-by-step reasoning")
    direct_effects: list[str] = Field(description="Direct effects on the label domain")
    second_order_effects: list[str] = Field(description="Indirect or downstream effects")
    key_uncertainties: list[str] = Field(description="Main factors that could change the assessment")
    summary: str = Field(description="One-sentence summary of the impact")


# ---------------------------------------------------------------------------
# Shared value fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_assessment() -> ImpactAssessmentOutput:
    """Fully-populated LLM output object for impact-assessment tests."""
    return ImpactAssessmentOutput(
        label="Affordable and Clean Energy",
        impact_direction=ImpactDirection.POSITIVE,
        confidence=ConfidenceLevel.HIGH,
        reasoning=(
            "1. The policy targets renewable energy capacity directly. "
            "2. Investment subsidies lower the financial barrier. "
            "3. Long-term grid integration decreases fossil-fuel dependency."
        ),
        direct_effects=[
            "Increased public funding for solar and wind installations.",
            "Reduced energy costs for low-income households.",
        ],
        second_order_effects=[
            "Job creation in clean-energy manufacturing.",
            "Decreased urban air pollution levels.",
        ],
        key_uncertainties=[
            "Maturity of grid-scale storage technology.",
            "Political stability of multi-year funding commitments.",
        ],
        summary=(
            "Strong positive impact on clean-energy access through "
            "targeted subsidies and infrastructure investment."
        ),
    )


@pytest.fixture
def sample_process_item() -> ProcessItem:
    return ProcessItem(
        expression_uri=EXPRESSION_URI,
        expression_content=EXPRESSION_CONTENT,
        language=LANGUAGE_URI,
        work_uri=WORK_URI,
    )


@pytest.fixture
def sample_policy_label() -> PolicyLabel:
    return PolicyLabel(
        annotation_uri=ANNOTATION_URI,
        policy_concept_uri=CONCEPT_URI,
        policy_label="Affordable and Clean Energy",
    )


# ---------------------------------------------------------------------------
# Task-instance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def impact_task() -> ImpactAssessmentTask:
    """
    ImpactAssessmentTask created without calling __init__ (no SPARQL on setup).
    The LLM is replaced by a MagicMock.
    """
    task = object.__new__(ImpactAssessmentTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.impact_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    task.provider = "ollama"
    task.llm = MagicMock()
    return task


# ---------------------------------------------------------------------------
# Triplestore data fixtures  (ImpactAssessment tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def expression_triples():
    """
    Inserts the ELI expression graph into named graphs.

    Graphs written:
        GRAPHS["jobs"]            task:inputContainer
        GRAPHS["data_containers"] task:hasResource
        GRAPHS["expressions"]     eli:Expression + epvoc:expressionContent
        GRAPHS["works"]           eli:Work + eli:is_realized_by
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                {sparql_escape_uri(TASK_URI)} {sparql_escape_uri(NS["task"] + "inputContainer")} {sparql_escape_uri(CONTAINER_URI)} .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                {sparql_escape_uri(CONTAINER_URI)} {sparql_escape_uri(NS["task"] + "hasResource")} {sparql_escape_uri(EXPRESSION_URI)} .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" ;
                    {sparql_escape_uri(NS["eli"] + "language")} {sparql_escape_uri(LANGUAGE_URI)} .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["works"])} {{
                {sparql_escape_uri(WORK_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Work")} ;
                    {sparql_escape_uri(NS["eli"] + "is_realized_by")} {sparql_escape_uri(EXPRESSION_URI)} .
            }}
        }}
    """)
    yield


@pytest.fixture
def annotation_triples():
    """
    Inserts OA annotation + SKOS concept triples for fetch_policy_labels().

    Graphs written:
        GRAPHS["ai"]      oa:Annotation
        GRAPHS["public"]  skos:Concept
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} {sparql_escape_uri(CONCEPT_URI)} .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["public"])} {{
                {sparql_escape_uri(CONCEPT_URI)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Affordable and Clean Energy" .
            }}
        }}
    """)
    yield


@pytest.fixture
def bare_annotation_triple():
    """
    Inserts only `<annotation_uri> a oa:Annotation` – minimum required for
    the WHERE clause in ImpactAssessment.store() to match.
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)} a {sparql_escape_uri(NS["oa"] + "Annotation")} .
            }}
        }}
    """)
    yield


# ---------------------------------------------------------------------------
# Triplestore data fixtures  (ModelAnnotatingTask / ModelBatchAnnotatingTask)
# ---------------------------------------------------------------------------

@pytest.fixture
def expression_content_triple():
    """
    Inserts an eli:Expression with epvoc:expressionContent into the expressions
    graph so that DecisionTask.fetch_data() returns non-empty text.
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
            }}
        }}
    """)
    yield


@pytest.fixture
def unannotated_expressions():
    """
    Inserts two eli:Expression triples without any annotations, for use in
    fetch_decisions_without_annotations tests.
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
                {sparql_escape_uri(EXPRESSION_URI_2)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT_2}" .
            }}
        }}
    """)
    yield


@pytest.fixture
def one_annotated_one_plain_expression():
    """
    Inserts EXPRESSION_URI (no annotation) and EXPRESSION_URI_2 (with a
    classifying annotation) so that fetch_decisions_without_annotations()
    returns only EXPRESSION_URI.
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
                {sparql_escape_uri(EXPRESSION_URI_2)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT_2}" .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI_2)} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} .
            }}
        }}
    """)
    yield


@pytest.fixture
def two_annotated_expressions():
    """
    Inserts two eli:Expression triples each with a classifying annotation,
    so that fetch_decisions_without_annotations() returns an empty list.
    """
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                {sparql_escape_uri(EXPRESSION_URI)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT}" .
                {sparql_escape_uri(EXPRESSION_URI_2)}
                    a {sparql_escape_uri(NS["eli"] + "Expression")} ;
                    {sparql_escape_uri(NS["epvoc"] + "expressionContent")} "{EXPRESSION_CONTENT_2}" .
            }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                {sparql_escape_uri(ANNOTATION_URI)}
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI)} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} .
                <http://test.example.org/annotations/annotation-2>
                    a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} {sparql_escape_uri(EXPRESSION_URI_2)} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} .
            }}
        }}
    """)
    yield
