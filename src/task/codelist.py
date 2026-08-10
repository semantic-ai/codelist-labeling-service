import uuid
from abc import ABC
from string import Template
from decide_ai_service_base.task import DecisionTask
from pydantic import BaseModel, Field
from helpers import query, logger
from escape_helpers import sparql_escape_uri
from decide_ai_service_base.sparql_config import get_prefixes_for_query
from decide_ai_service_base.annotation import LinkingAnnotation
from decide_ai_service_base.util import get_agent_uri


from helpers import query, update, logger
from escape_helpers import sparql_escape_uri, sparql_escape_string

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import AGENT_TYPES, get_prefixes_for_query, GRAPHS


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

        SELECT DISTINCT ?concept ?code ?definition
        WHERE {{
            ?concept skos:inScheme {sparql_escape_uri(concept_scheme_uri)} ;
                     skos:prefLabel ?prefLabel .
            FILTER(LANG(?prefLabel) = "en" || LANG(?prefLabel) = "")
            OPTIONAL {{ ?concept skos:notation ?notation . }}
            BIND(COALESCE(STR(?notation), STR(?prefLabel)) AS ?code)
            OPTIONAL {{
                ?concept skos:definition ?rawDefinition .
                FILTER(LANG(?rawDefinition) = "en" || LANG(?rawDefinition) = "")
            }}
            BIND(STR(?rawDefinition) AS ?definition)
        }}
        """

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])

        if not bindings:
            raise RuntimeError(f"No concepts found for concept scheme {concept_scheme_uri}")

        entries = [
            CodelistEntry(
                uri=b["concept"]["value"],
                label=b["code"]["value"],
                definition=b["definition"]["value"] if "definition" in b else None,
            )
            for b in bindings
            if "concept" in b and "code" in b
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
    #------------------------------
    # Codelist resolution and SPARQL helpers
    #------------------------------
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
            logger.info(
                "get_expressions_in_task_filter: no inputContainer resources found for task %s — no filter applied",
                self.task_uri,
            )
            return "";
        expression_uris = [binding["expression"]["value"] for binding in bindings]
        # logger.info(
        #     "get_expressions_in_task_filter: found %d inputContainer resource(s) for task %s: %s",
        #     len(expression_uris),
        #     self.task_uri,
        #     expression_uris,
        # )
        expression_values = "\n".join([sparql_escape_uri(uri) for uri in expression_uris])
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


    #------------------------------
    # SPARQL query helpers for fetching expressions and their classes
    #------------------------------

    def fetch_expressions_without_annotations_with_text(self, target_graph: str) -> list[dict]:
        concept_scheme_uri = self.fetch_codelist_uri_for_task()
        expression_filter = self.get_expressions_in_task_filter()
        member_block = self.member_content_sparql_block("?s")
        q = Template(
            get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct", "skos", "schema") + """
            SELECT ?s ?title ?description ?decision_basis ?content ?title_code ?work_type
                   (GROUP_CONCAT(DISTINCT ?_member_text; separator="\\n\\n") AS ?member_content)
            WHERE {
                $expression_filter
                GRAPH $target_graph {
                    ?s rdf:type eli:Expression .
                    OPTIONAL { ?s eli:title ?title }
                    OPTIONAL { ?s eli:description ?description }
                    OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                    OPTIONAL { ?s epvoc:expressionContent ?content }
                    $member_block
                }
                FILTER NOT EXISTS {
                    VALUES ?g { $target_graph $ai_graph }
                    GRAPH ?g {
                        ?ann a oa:Annotation ;
                            oa:hasTarget ?s ;
                            oa:motivatedBy oa:classifying ;
                            oa:hasBody ?annotatedConcept .
                    }
                    GRAPH $public_graph {
                        ?annotatedConcept skos:inScheme $concept_scheme_uri .
                    }
                }
            }
            GROUP BY ?s ?title ?description ?decision_basis ?content ?title_code ?work_type
            """
        ).substitute(
            target_graph=sparql_escape_uri(target_graph),
            ai_graph=sparql_escape_uri(GRAPHS["ai"]),
            public_graph=sparql_escape_uri(GRAPHS["public"]),
            concept_scheme_uri=sparql_escape_uri(concept_scheme_uri),
            expression_filter=expression_filter,
            member_block=member_block,
        )

        response = query(q, sudo=True)
        results = []
        for b in response.get("results", {}).get("bindings", []):
            text = self.assemble_expression_text(b)
            if text:
                results.append({"uri": b["s"]["value"], "text": text})
        return results

    #------------------------------
    # SPARQL query helpers for fetching expression text and member content
    #------------------------------

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

    def member_content_sparql_block(
        self,
        expr_var: str = "?s",
        content_property: str = "epvoc:expressionContent",
    ) -> str:
        """Return SPARQL OPTIONAL blocks for work_type, title_code, and aggregated member content.

        The caller must include ``schema`` in ``get_prefixes_for_query`` and
        add a ``GROUP BY`` that covers ``?title_code`` and ``?work_type``
        plus a ``GROUP_CONCAT(DISTINCT ?_member_text; separator="\\n\\n") AS ?member_content``
        in the SELECT clause.
        """
        return f"""
        OPTIONAL {{
            ?_work eli:is_realized_by {expr_var} ;
                   eli:work_type ?work_type .
        }}
        OPTIONAL {{ {expr_var} schema:code ?title_code }}
        OPTIONAL {{
            ?_work eli:is_realized_by {expr_var} ;
                   eli:has_member ?_member_work .
            ?_member_work eli:is_realized_by ?_member_expr .
            ?_member_expr a eli:Expression .
            OPTIONAL {{ ?_member_expr eli:title ?_m_title }}
            OPTIONAL {{ ?_member_expr schema:code ?_m_code }}
            OPTIONAL {{ ?_member_expr eli:description ?_m_description }}
            OPTIONAL {{ ?_member_expr {content_property} ?_m_content }}

            BIND(CONCAT(
                COALESCE(STR(?_m_code), ""), "\\n",
                COALESCE(STR(?_m_title), ""), "\\n",
                COALESCE(STR(?_m_description), ""), "\\n",
                COALESCE(STR(?_m_content), "")
            ) AS ?_member_text)
        }}
        """

    def fetch_member_expression_mapping(self, source_uri: str) -> dict[str, str]:
        """Fetch a mapping of action code → expression URI for member expressions.

        For a single action (no members), returns {own_code: own_uri}.
        For an actieplan, returns {member_code: member_expression_uri} for each member.

        Returns:
            dict mapping code strings to expression URIs.
        """
        q = Template(
            get_prefixes_for_query("eli", "schema") +
            """
            SELECT ?code ?memberExpr WHERE {
                GRAPH ?graph {
                    ?work eli:is_realized_by $source ;
                          eli:has_member ?memberWork .
                    ?memberWork eli:is_realized_by ?memberExpr .
                    ?memberExpr a eli:Expression .
                    ?memberExpr schema:code ?code .
                }
            }
            """
        ).substitute(source=sparql_escape_uri(source_uri))

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        if bindings:
            return {
                b["code"]["value"]: b["memberExpr"]["value"]
                for b in bindings
                if "code" in b and "memberExpr" in b
            }

        # No members — single action, return self-mapping with own code
        q_self = Template(
            get_prefixes_for_query("schema") +
            """
            SELECT ?code WHERE {
                GRAPH ?graph {
                    $source schema:code ?code .
                }
            }
            """
        ).substitute(source=sparql_escape_uri(source_uri))

        res_self = query(q_self, sudo=True)
        self_bindings = res_self.get("results", {}).get("bindings", [])

        if self_bindings:
            code = self_bindings[0]["code"]["value"]
            return {code: source_uri}

        # Fallback: no code found, use URI fragment as key
        fallback_key = source_uri.rsplit("/", 1)[-1]
        return {fallback_key: source_uri}

    def assemble_expression_text(self, binding: dict) -> str:
        """Assemble classification input text from a SPARQL binding row.

        Produces a structured text block with an optional header
        (work_type + code), the core expression fields, and any
        aggregated member content. Exact repeated lines are emitted once;
        VMM ``expressionContent`` values can already contain the code/title
        metadata that is also available through separate RDF properties.
        """
        parts: list[str] = []

        code = binding.get("title_code", {}).get("value", "")
        work_type_raw = binding.get("work_type", {}).get("value", "")
        work_type = work_type_raw.rsplit("/", 1)[-1] if work_type_raw else ""

        if work_type or code:
            parts.append(f"{work_type} [{code}]" if code else work_type)

        for field in ("title", "description", "decision_basis", "content"):
            val = binding.get(field, {}).get("value", "")
            if val:
                parts.append(val)

        member_content = binding.get("member_content", {}).get("value", "")
        if member_content:
            parts.append(member_content)

        lines: list[str] = []
        seen_lines: set[str] = set()
        for part in parts:
            for raw_line in part.splitlines():
                line = raw_line.strip()
                if not line:
                    if lines and lines[-1]:
                        lines.append("")
                    continue

                normalized = " ".join(line.split()).casefold()
                if normalized in seen_lines:
                    continue

                seen_lines.add(normalized)
                lines.append(line)

        while lines and not lines[-1]:
            lines.pop()

        return "\n".join(lines)

    def fetch_text_with_property_path(self, property_uri: str) -> str:
        """Fetch text from the task source using the specified property URI.

        Uses sparql_escape_uri() to safely interpolate the property URI,
        preventing SPARQL injection.
        """
        if property_uri.startswith("<") and property_uri.endswith(">"):
            property_uri = property_uri[1:-1]

        property_term = sparql_escape_uri(property_uri)
        member_block = self.member_content_sparql_block(
            "?s", content_property=property_term
        )
        q = Template(
            get_prefixes_for_query("rdf", "eli", "eli-dl", "schema") +
            """
            SELECT ?s ?title ?description ?decision_basis ?content ?title_code ?work_type
                   (GROUP_CONCAT(DISTINCT ?_member_text; separator="\\n\\n") AS ?member_content)
            WHERE {
                GRAPH ?graph {
                    VALUES ?s { $source }
                    ?s rdf:type eli:Expression .
                    OPTIONAL { ?s eli:title ?title }
                    OPTIONAL { ?s eli:description ?description }
                    OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                    OPTIONAL { ?s $property ?content }
                    $member_block
                }
            }
            GROUP BY ?s ?title ?description ?decision_basis ?content ?title_code ?work_type
            """
        ).substitute(
            source=sparql_escape_uri(self.source),
            property=property_term,
            member_block=member_block,
        )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        texts = [self.assemble_expression_text(b) for b in bindings]
        return "\n\n".join(text for text in texts if text)


    #------------------------------
    # SPARQL query helpers for storing no-match-found annotations
    #------------------------------

    def _store_no_match_for_expression(self, expression_uri: str):
        """Store a no-match-found annotation for a specific expression."""
        uri = "http://mu.semte.ch/vocabularies/ext/no-match-found"
        try:
            annotation = LinkingAnnotation(
                self.task_uri,
                expression_uri,
                uri,
                get_agent_uri("model_annotator"),
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
        except Exception as e:
            logger.error("Failed to insert no-match-found for %s: %s", expression_uri, e)

    def store_no_match(self):
        uri = f"http://mu.semte.ch/vocabularies/ext/no-match-found"

        try:
            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                uri,
                get_agent_uri("model_annotator"),
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
            self.results_container_uris.append(self.create_output_container(self.source))
        except Exception as e:
            error_msg = f"Failed to insert no-match-found: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

            
    #------------------------------
    # Function to create an output data container for an annotation
    #------------------------------

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container for an annotation.

        Args:
            resource: String containing an annotation URI

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            """
            INSERT DATA {
                GRAPH $graph {
                    $container a nfo:DataContainer ;
                        mu:uuid $uuid ;
                        task:hasResource $resource .
                }
            }
            """
        ).substitute(
            graph=sparql_escape_uri(GRAPHS["data_containers"]),
            container=sparql_escape_uri(container_uri),
            uuid=sparql_escape_string(container_id),
            resource=sparql_escape_uri(resource)
        )

        update(q, sudo=True)
        return container_uri



