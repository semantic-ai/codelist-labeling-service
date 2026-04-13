import logging

from pydantic import BaseModel, Field
from helpers import query
from escape_helpers import sparql_escape_uri

logger = logging.getLogger(__name__)


class CodelistEntry(BaseModel):
    uri: str = Field(description="URI of the SKOS concept")
    label: str = Field(description="Label of the concept")


def fetch_codelist(concept_scheme_uri: str) -> list[CodelistEntry]:
    # TODO: also fetch skos:definition to give the LLM more context for classification
    """Fetch all SKOS concepts from a concept scheme in the triplestore."""
    q = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?concept ?label
    WHERE {{
        ?concept skos:inScheme {sparql_escape_uri(concept_scheme_uri)} ;
                 skos:prefLabel ?label .
        FILTER(LANG(?label) = "en" || LANG(?label) = "")
    }}
    """

    response = query(q, sudo=True)
    bindings = response.get("results", {}).get("bindings", [])

    entries = [
        CodelistEntry(
            uri=b["concept"]["value"],
            label=b["label"]["value"]
        )
        for b in bindings
        if "concept" in b and "label" in b
    ]

    logger.info("Fetched %d concepts from scheme %s", len(entries), concept_scheme_uri)
    return entries


def fetch_codelist_uri_for_task(task_uri: str) -> str:
    """Resolve the SKOS ConceptScheme URI from the Job linked to this task."""
    q = f"""
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>

    SELECT ?codelist
    WHERE {{
        {sparql_escape_uri(task_uri)} dct:isPartOf ?job .
        ?job ext:codelist ?codelist .
    }}
    """
    response = query(q, sudo=True)
    bindings = response.get("results", {}).get("bindings", [])
    if not bindings:
        raise ValueError(
            f"No codelist URI found for task {task_uri}. "
            f"Ensure the job has ext:codelist set."
        )
    return bindings[0]["codelist"]["value"]


def build_label_to_uri_map(entries: list[CodelistEntry]) -> dict[str, str]:
    """Build a label -> URI mapping for reverse lookup after LLM response."""
    mapping = {}
    for entry in entries:
        mapping[entry.label] = entry.uri
        mapping[entry.label.lower()] = entry.uri
        mapping[entry.label.replace(" ", "_")] = entry.uri
        mapping[entry.label.replace(" ", "_").lower()] = entry.uri
    return mapping


def resolve_label_to_uri(label: str, label_to_uri: dict[str, str], entries: list[CodelistEntry]) -> str | None:
    """Resolve an LLM-returned label to a concept URI.

    Tries exact match first, then falls back to prefix/substring matching
    for cases where the LLM truncates long labels.
    """
    # Exact match (with normalization variants)
    uri = (
        label_to_uri.get(label)
        or label_to_uri.get(label.replace("_", " "))
        or label_to_uri.get(label.lower())
        or label_to_uri.get(label.replace("_", " ").lower())
    )
    if uri:
        return uri

    # Fuzzy fallback: check if an entry starts with the LLM's label
    normalized = label.replace("_", " ").lower().strip()
    for entry in entries:
        if entry.label.lower().startswith(normalized):
            return entry.uri

    return None
