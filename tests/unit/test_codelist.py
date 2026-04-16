"""
Unit tests for Codelist, CodelistEntry, and CodeListTask
(src/task/codelist.py).

Test strategy
─────────────
Class / method                           SPARQL          Notes
CodelistEntry                            –               pure Pydantic
Codelist.from_uri()                      real Virtuoso   concepts in public graph
Codelist.build_label_to_uri_map()        –               pure Python
Codelist.resolve_label_to_uri()          –               pure Python
CodeListTask.fetch_codelist_uri_for_task real Virtuoso   job triples in jobs graph
CodeListTask.fetch_codelist()            –               both collaborators mocked

The autouse `isolate_test_graphs` fixture in conftest.py wipes all test
graphs before and after every test, so each Virtuoso test starts clean.
"""

import logging

import helpers
import pytest
from escape_helpers import sparql_escape_uri
from pydantic import ValidationError

from decide_ai_service_base.sparql_config import GRAPHS
from src.task.codelist import Codelist, CodelistEntry, CodeListTask
from src.task.annotate import ModelAnnotatingTask  # concrete CodeListTask subclass

from tests.unit.conftest import (
    CONCEPT_SCHEME_URI,
    CONCEPT_URI,
    CONCEPT_URI_2,
    JOB_URI,
    NS,
    TASK_URI,
)

# ---------------------------------------------------------------------------
# Extra URIs used only in this module (no cross-test contamination risk –
# the autouse wiper clears ALL data from all test graphs before each test)
# ---------------------------------------------------------------------------

_PUBLIC_GRAPH = "http://mu.semte.ch/graphs/public"
_CONCEPT_URI_3 = "http://test.example.org/concepts/sdg-goal-3"
_DCT_IS_PART_OF = "http://purl.org/dc/terms/isPartOf"
_EXT_CODELIST = "http://mu.semte.ch/vocabularies/ext/codelist"


# ---------------------------------------------------------------------------
# Shared fixture: minimal CodeListTask instance (bypasses __init__)
# ---------------------------------------------------------------------------

@pytest.fixture
def codelist_task() -> CodeListTask:
    """
    Concrete CodeListTask instance created without __init__ so no SPARQL is
    issued during fixture setup.  Uses ModelAnnotatingTask as the concrete
    subclass.
    """
    task = object.__new__(ModelAnnotatingTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.codelist_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    return task


# ---------------------------------------------------------------------------
# Virtuoso data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_english_concepts():
    """
    Insert two SKOS concepts with English prefLabels into the public graph.
    """
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
def mixed_language_concepts():
    """
    Insert concepts with English, Dutch, and untagged prefLabels so that
    language-filter behaviour can be verified.
    """
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(_PUBLIC_GRAPH)} {{
                {sparql_escape_uri(CONCEPT_URI)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Clean Water"@en .
                {sparql_escape_uri(CONCEPT_URI_2)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "Schoon Water"@nl .
                {sparql_escape_uri(_CONCEPT_URI_3)}
                    a {sparql_escape_uri(NS["skos"] + "Concept")} ;
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "No Tag Label" .
            }}
        }}
    """)
    yield


@pytest.fixture
def task_with_codelist_job():
    """
    Insert the job triples that fetch_codelist_uri_for_task() needs:
      <TASK_URI> dct:isPartOf <JOB_URI> .
      <JOB_URI>  ext:codelist  <CONCEPT_SCHEME_URI> .
    """
    helpers.update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                {sparql_escape_uri(TASK_URI)} {sparql_escape_uri(_DCT_IS_PART_OF)} {sparql_escape_uri(JOB_URI)} .
                {sparql_escape_uri(JOB_URI)}  {sparql_escape_uri(_EXT_CODELIST)}    {sparql_escape_uri(CONCEPT_SCHEME_URI)} .
            }}
        }}
    """)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# CodelistEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestCodelistEntry:

    def test_stores_uri_and_label(self):
        entry = CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy")

        assert entry.uri == CONCEPT_URI
        assert entry.label == "Affordable and Clean Energy"

    def test_uri_is_required(self):
        with pytest.raises(ValidationError):
            CodelistEntry(label="Missing URI")

    def test_label_is_required(self):
        with pytest.raises(ValidationError):
            CodelistEntry(uri=CONCEPT_URI)

    def test_equality_by_value(self):
        a = CodelistEntry(uri=CONCEPT_URI, label="X")
        b = CodelistEntry(uri=CONCEPT_URI, label="X")

        assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# Codelist.from_uri()
# ─────────────────────────────────────────────────────────────────────────────

class TestCodelistFromUri:

    def test_returns_one_entry_per_concept(
        self, two_english_concepts
    ):
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)

        assert len(codelist) == 2

    def test_entry_has_correct_uri(self, two_english_concepts):
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)
        uris = {e.uri for e in codelist}

        assert CONCEPT_URI in uris
        assert CONCEPT_URI_2 in uris

    def test_entry_has_correct_label(self, two_english_concepts):
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)
        labels = {e.label for e in codelist}

        assert "Affordable and Clean Energy" in labels
        assert "Climate Action" in labels

    def test_returns_codelist_instance(self, two_english_concepts):
        result = Codelist.from_uri(CONCEPT_SCHEME_URI)

        assert isinstance(result, Codelist)

    def test_returns_empty_codelist_when_scheme_is_unknown(self):
        """No concepts → empty list, no exception."""
        result = Codelist.from_uri("http://test.example.org/codelists/unknown")

        assert result == []
        assert isinstance(result, Codelist)

    def test_includes_english_labelled_concepts(self, mixed_language_concepts):
        """Concepts with @en prefLabel are included."""
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)
        labels = {e.label for e in codelist}

        assert "Clean Water" in labels

    def test_includes_untagged_label(self, mixed_language_concepts):
        """Concepts whose prefLabel has no language tag are included."""
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)
        labels = {e.label for e in codelist}

        assert "No Tag Label" in labels

    def test_excludes_non_english_labels(self, mixed_language_concepts):
        """Concepts whose only prefLabel is in a language other than English
        are excluded by the FILTER."""
        codelist = Codelist.from_uri(CONCEPT_SCHEME_URI)
        labels = {e.label for e in codelist}

        assert "Schoon Water" not in labels

    def test_only_returns_concepts_from_requested_scheme(
        self, two_english_concepts
    ):
        """Concepts from a different scheme are not returned."""
        other_scheme = "http://test.example.org/codelists/other"
        result = Codelist.from_uri(other_scheme)

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Codelist.build_label_to_uri_map()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildLabelToUriMap:

    @pytest.fixture
    def single_entry_codelist(self):
        return Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy")
        ])

    def test_maps_exact_label(self, single_entry_codelist):
        m = single_entry_codelist.build_label_to_uri_map()

        assert m["Affordable and Clean Energy"] == CONCEPT_URI

    def test_maps_lowercase_label(self, single_entry_codelist):
        m = single_entry_codelist.build_label_to_uri_map()

        assert m["affordable and clean energy"] == CONCEPT_URI

    def test_maps_underscored_label(self, single_entry_codelist):
        m = single_entry_codelist.build_label_to_uri_map()

        assert m["Affordable_and_Clean_Energy"] == CONCEPT_URI

    def test_maps_underscored_and_lowercased_label(self, single_entry_codelist):
        m = single_entry_codelist.build_label_to_uri_map()

        assert m["affordable_and_clean_energy"] == CONCEPT_URI

    def test_returns_four_keys_per_entry(self, single_entry_codelist):
        """Each entry contributes exactly four normalisation variants."""
        m = single_entry_codelist.build_label_to_uri_map()

        assert len(m) == 4

    def test_returns_empty_dict_for_empty_codelist(self):
        m = Codelist([]).build_label_to_uri_map()

        assert m == {}

    def test_all_variants_resolve_to_same_uri(self, single_entry_codelist):
        m = single_entry_codelist.build_label_to_uri_map()

        assert all(v == CONCEPT_URI for v in m.values())

    def test_two_entries_produce_independent_uris(self):
        codelist = Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Goal One"),
            CodelistEntry(uri=CONCEPT_URI_2, label="Goal Two"),
        ])
        m = codelist.build_label_to_uri_map()

        assert m["Goal One"] == CONCEPT_URI
        assert m["Goal Two"] == CONCEPT_URI_2


# ─────────────────────────────────────────────────────────────────────────────
# Codelist.resolve_label_to_uri()
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveLabel:
    """
    resolve_label_to_uri() tries four exact-match normalisations then a
    case-insensitive prefix (startswith) fuzzy fallback.
    """

    @pytest.fixture
    def codelist(self):
        return Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy"),
            CodelistEntry(uri=CONCEPT_URI_2, label="Climate Action"),
        ])

    @pytest.fixture
    def label_map(self, codelist):
        return codelist.build_label_to_uri_map()

    # -- Exact and normalised matches ----------------------------------------

    def test_exact_label_match(self, codelist, label_map):
        assert codelist.resolve_label_to_uri(
            "Affordable and Clean Energy", label_map
        ) == CONCEPT_URI

    def test_lowercase_match(self, codelist, label_map):
        assert codelist.resolve_label_to_uri(
            "affordable and clean energy", label_map
        ) == CONCEPT_URI

    def test_underscored_label_match(self, codelist, label_map):
        assert codelist.resolve_label_to_uri(
            "Affordable_and_Clean_Energy", label_map
        ) == CONCEPT_URI

    def test_underscored_and_lowercased_match(self, codelist, label_map):
        assert codelist.resolve_label_to_uri(
            "affordable_and_clean_energy", label_map
        ) == CONCEPT_URI

    def test_uppercase_input_matches_via_lowercase_normalisation(
        self, codelist, label_map
    ):
        """Mixed-case that isn't in the map falls through to .lower() check."""
        assert codelist.resolve_label_to_uri(
            "AFFORDABLE AND CLEAN ENERGY", label_map
        ) == CONCEPT_URI

    # -- Fuzzy prefix fallback -----------------------------------------------

    def test_truncated_prefix_matches_via_fuzzy_fallback(
        self, codelist, label_map
    ):
        """LLM often truncates long labels; the startswith check catches this."""
        assert codelist.resolve_label_to_uri(
            "Affordable and Clean", label_map
        ) == CONCEPT_URI

    def test_fuzzy_fallback_is_case_insensitive(self, codelist, label_map):
        assert codelist.resolve_label_to_uri(
            "climate", label_map
        ) == CONCEPT_URI_2

    def test_fuzzy_fallback_strips_trailing_whitespace(
        self, codelist, label_map
    ):
        assert codelist.resolve_label_to_uri(
            "Climate Action  ", label_map
        ) == CONCEPT_URI_2

    def test_fuzzy_prefix_with_underscores(self, codelist, label_map):
        """Underscores in the fuzzy input are converted to spaces."""
        assert codelist.resolve_label_to_uri(
            "Affordable_and", label_map
        ) == CONCEPT_URI

    # -- No match ------------------------------------------------------------

    def test_returns_none_for_completely_unknown_label(
        self, codelist, label_map
    ):
        assert codelist.resolve_label_to_uri(
            "Zero Hunger", label_map
        ) is None

    def test_empty_codelist_disables_fuzzy_fallback(self):
        """
        An empty codelist means the fuzzy startswith loop has nothing to
        iterate — a truncated label that would normally match via prefix
        returns None.
        """
        empty = Codelist([])
        assert empty.resolve_label_to_uri("Affordable and Clean", {}) is None

    def test_empty_map_still_resolves_via_fuzzy_fallback(self, codelist):
        """
        When label_to_uri is empty the exact-match paths all fail, but the
        fuzzy startswith loop iterates over self and can still find a match.
        """
        assert codelist.resolve_label_to_uri(
            "Affordable and Clean", {}
        ) == CONCEPT_URI

    # -- Disambiguation ------------------------------------------------------

    def test_exact_match_does_not_bleed_into_other_entries(
        self, codelist, label_map
    ):
        """The first matching concept is returned, not any later one."""
        result = codelist.resolve_label_to_uri("Climate Action", label_map)

        assert result == CONCEPT_URI_2
        assert result != CONCEPT_URI


# ─────────────────────────────────────────────────────────────────────────────
# CodeListTask.fetch_codelist_uri_for_task()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchCodelistUriForTask:

    def test_returns_codelist_uri_when_job_is_linked(
        self, codelist_task, task_with_codelist_job
    ):
        result = codelist_task.fetch_codelist_uri_for_task()

        assert result == CONCEPT_SCHEME_URI

    def test_raises_value_error_when_no_job_linked(self, codelist_task):
        """No job triples in the triplestore → informative ValueError."""
        with pytest.raises(ValueError, match="No codelist URI found"):
            codelist_task.fetch_codelist_uri_for_task()

    def test_error_message_includes_task_uri(self, codelist_task):
        with pytest.raises(ValueError, match=TASK_URI):
            codelist_task.fetch_codelist_uri_for_task()

    def test_returns_string(self, codelist_task, task_with_codelist_job):
        result = codelist_task.fetch_codelist_uri_for_task()

        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# CodeListTask.fetch_codelist()
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchCodelist:

    def test_delegates_uri_resolution_to_fetch_codelist_uri_for_task(
        self, codelist_task, mocker
    ):
        mock_uri = mocker.patch.object(
            codelist_task, "fetch_codelist_uri_for_task",
            return_value=CONCEPT_SCHEME_URI,
        )
        mocker.patch.object(Codelist, "from_uri", return_value=Codelist([]))

        codelist_task.fetch_codelist()

        mock_uri.assert_called_once_with()

    def test_passes_scheme_uri_to_from_uri(self, codelist_task, mocker):
        mocker.patch.object(
            codelist_task, "fetch_codelist_uri_for_task",
            return_value=CONCEPT_SCHEME_URI,
        )
        mock_from_uri = mocker.patch.object(
            Codelist, "from_uri", return_value=Codelist([])
        )

        codelist_task.fetch_codelist()

        mock_from_uri.assert_called_once_with(CONCEPT_SCHEME_URI)

    def test_returns_codelist_from_from_uri(self, codelist_task, mocker):
        expected = Codelist([
            CodelistEntry(uri=CONCEPT_URI, label="Affordable and Clean Energy")
        ])
        mocker.patch.object(
            codelist_task, "fetch_codelist_uri_for_task",
            return_value=CONCEPT_SCHEME_URI,
        )
        mocker.patch.object(Codelist, "from_uri", return_value=expected)

        result = codelist_task.fetch_codelist()

        assert result is expected

    def test_propagates_value_error_when_no_job_linked(
        self, codelist_task, mocker
    ):
        mocker.patch.object(
            codelist_task, "fetch_codelist_uri_for_task",
            side_effect=ValueError("No codelist URI found"),
        )

        with pytest.raises(ValueError, match="No codelist URI found"):
            codelist_task.fetch_codelist()
