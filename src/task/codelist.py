import re
from abc import ABC
from string import Template
from decide_ai_service_base.task import DecisionTask
import logging
from pydantic import BaseModel, Field
from helpers import query
from escape_helpers import sparql_escape_uri


logger = logging.getLogger(__name__)


class CodelistEntry(BaseModel):
    uri: str = Field(description="URI of the SKOS concept")
    label: str = Field(description="Label of the concept")
    definition: str | None = Field(default=None, description="Optional skos:definition of the concept")


class Codelist(list[CodelistEntry]):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concept_scheme_uri = None

    @classmethod
    def from_uri(cls, concept_scheme_uri: str) -> 'Codelist':
        """Fetch all SKOS concepts from a concept scheme in the triplestore."""
        q = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?concept ?label ?definition
        WHERE {{
            ?concept skos:inScheme {sparql_escape_uri(concept_scheme_uri)} ;
                     skos:prefLabel ?label .
            FILTER(LANG(?label) = "en" || LANG(?label) = "")
            OPTIONAL {{
                ?concept skos:definition ?definition .
                FILTER(LANG(?definition) = "en" || LANG(?definition) = "")
            }}
        }}
        """

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])

        if not bindings:
            raise RuntimeError(f"No concepts found for concept scheme {concept_scheme_uri}")

        entries = [
            CodelistEntry(
                uri=b["concept"]["value"],
                label=b["label"]["value"],
                definition=b["definition"]["value"] if "definition" in b else None,
            )
            for b in bindings
            if "concept" in b and "label" in b
        ]

        logger.info("Fetched %d concepts from scheme %s", len(entries), concept_scheme_uri)
        instance = cls(entries)
        instance.concept_scheme_uri = concept_scheme_uri
        return instance

    def build_label_to_uri_map(self) -> dict[str, str]:
        """Build a label -> URI mapping for reverse lookup after LLM response."""
        mapping = {}
        for entry in self:
            mapping[entry.label] = entry.uri
            mapping[entry.label.lower()] = entry.uri
            mapping[entry.label.replace(" ", "_")] = entry.uri
            mapping[entry.label.replace(" ", "_").lower()] = entry.uri
        return mapping

    def build_uri_to_label_map(self) -> dict[str, str]:
        return {entry.uri: entry.label for entry in self}

    def get_labels(self) -> list[str]:
        return [entry.label for entry in self]

    def get_labels_with_definitions(self) -> str:
        """Return a prompt-ready string listing labels and, when available,
        a separate mapping of labels to their definitions.

        The labels are always listed first so the LLM knows which values
        to return.  Definitions are appended as supplementary context only
        when at least one entry has a ``skos:definition``.
        """
        labels = self.get_labels()
        definitions = {entry.label: entry.definition for entry in self if entry.definition}

        if definitions:
            return f"{labels}\n\nLabel descriptions:\n{definitions}"
        return str(labels)

    def resolve_label_to_uri(self, label: str, label_to_uri: dict[str, str]) -> str | None:
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
        for entry in self:
            if entry.label.lower().startswith(normalized):
                return entry.uri

        return None


class CodeListTask(DecisionTask, ABC):
    def fetch_codelist_uri_for_task(self) -> str:
        """Resolve the SKOS ConceptScheme URI from the Job linked to this task."""
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>

            SELECT ?codelist
            WHERE {
                GRAPH ?graph {
                    $task dct:isPartOf ?job .
                    ?job ext:codelist ?codelist .
                }
            }
            """
        ).substitute(
            task=sparql_escape_uri(self.task_uri)
        )
        
        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        
        self.logger = logging.getLogger(__name__)

        if not bindings:
            raise ValueError(
                f"No codelist URI found for task {self.task_uri}. "
                f"Ensure the job has ext:codelist set."
            )
        return bindings[0]["codelist"]["value"]

    def fetch_codelist(self) -> Codelist:
        codelist_uri = self.fetch_codelist_uri_for_task()
        return Codelist.from_uri(codelist_uri)

    def get_expressions_in_task_filter(self, varname = "?s") -> list[str]:
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            PREFIX eli: <http://data.europa.eu/eli/ontology#>
            SELECT DISTINCT ?expression WHERE {
                $task <http://redpencil.data.gift/vocabularies/tasks/inputContainer> ?input.
                ?input <http://redpencil.data.gift/vocabularies/tasks/hasResource> ?expression.
            }
            """
        ).substitute(
            task=sparql_escape_uri(self.task_uri)
        )
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if not bindings:
            # this means do all expressions, so no filter
            return "";
        expression_values = "\n".join([sparql_escape_uri(binding["expression"]["value"]) for binding in bindings])
        values = f"VALUES {varname} {{ {expression_values} }}"

        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            PREFIX eli: <http://data.europa.eu/eli/ontology#>
            SELECT DISTINCT $varname WHERE {
                $values
                ?input <http://redpencil.data.gift/vocabularies/tasks/hasResource> ?expression.
                FILTER NOT EXISTS {
                    $varname a eli:Expression .
                }   
            }
            """
        ).substitute(
            values=values,
            varname=varname
        )
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if bindings:
            non_expression_uris = ", ".join([b[varname.replace("?","")]["value"] for b in bindings])
            raise RuntimeError(f"The following uris were not found to be expressions: {non_expression_uris}")
        
        return values
    
    def get_target_graph(self) -> str | None:
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            SELECT ?graph WHERE {
                $task dct:isPartOf ?job .
                ?job ext:graphForTargets ?graph .
            }
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if not bindings:
            return None
        return bindings[0]["graph"]["value"]

    def fetch_shape_targets(self) -> tuple[list[str], list[str]]:
        """Fetch ext:shapeForTargets from the job and resolve sh:targetNode / sh:targetClass.

        Returns:
            Tuple of (target_nodes, target_classes) — lists of URIs.
            Both lists are empty when no shapes are configured on the job.
        """
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            PREFIX sh: <http://www.w3.org/ns/shacl#>

            SELECT ?targetNode ?targetClass WHERE {
                $task dct:isPartOf ?job .
                ?job ext:shapeForTargets ?shape .
                OPTIONAL { ?shape sh:targetNode ?targetNode . }
                OPTIONAL { ?shape sh:targetClass ?targetClass . }
            }
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        target_nodes: set[str] = set()
        target_classes: set[str] = set()

        for b in bindings:
            if "targetNode" in b:
                target_nodes.add(b["targetNode"]["value"])
            if "targetClass" in b:
                target_classes.add(b["targetClass"]["value"])

        return list(target_nodes), list(target_classes)

    def fetch_property_path_for_text(self) -> str | None:
        """Fetch ext:propertyPathForText from the job.

        Returns the property URI string, or None if not configured.
        Validates the SPARQL result to prevent injection:
          - Must be a URI type (not a literal)
          - Must start with http:// or https://
        """
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>

            SELECT ?propertyPath WHERE {
                $task dct:isPartOf ?job .
                ?job ext:propertyPathForText ?propertyPath .
            }
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        result = bindings[0]["propertyPath"]
        uri = result["value"]
        
        return uri