import logging
import os
import random
import time
import uuid
from string import Template
from helpers import query, update, logger
from escape_helpers import sparql_escape_uri, sparql_escape_string
import uuid 

from decide_ai_service_base.task import DecisionTask, Task
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, get_prefixes_for_query, \
    GRAPHS
from decide_ai_service_base.annotation import LinkingAnnotation

from ..llm_models.llm_model_clients import create_llm_client
from ..llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput
from .codelist import Codelist, CodelistEntry, CodeListTask
from ..config import get_config


class ModelAnnotatingTask(CodeListTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str = None,
                 codelist_entries: 'Codelist | None' = None,
                 property_path_for_text: str | None = None):
        super().__init__(task_uri)
        self.source = source

        if source is not None:
            self.source = source

        config = get_config()

        self._codelist_entries = codelist_entries if codelist_entries is not None else self.fetch_codelist()
        self._label_to_uri = self._codelist_entries.build_label_to_uri_map()
        self._property_path_for_text = property_path_for_text

        # LLM setup
        self._llm = create_llm_client(config.llm)
        self._provider = config.llm.provider

        prompt = config.get_codelist_prompt(self._codelist_entries.concept_scheme_uri)
        self._llm_system_message = prompt.system_message
        self._llm_user_message = prompt.user_message

    def fetch_text_with_property_path(self, property_uri: str) -> str:
        """Fetch text from the task source using the specified property URI.

        Uses sparql_escape_uri() to safely interpolate the property URI,
        preventing SPARQL injection.
        """
        q = Template(
            get_prefixes_for_query("eli") +
            """
            SELECT ?text WHERE {
                GRAPH ?graph {
                    VALUES ?s { $source }
                    ?s $property ?text .
                }
            }
            """
        ).substitute(
            source=sparql_escape_uri(self.source),
            property=property_uri
        )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        texts = [b["text"]["value"] for b in bindings if "text" in b]
        return "\n".join(texts)

    def process(self):
        if self._property_path_for_text:
            task_data = self.fetch_text_with_property_path(self._property_path_for_text)
        else:
            task_data = self.fetch_data()

        if not task_data.strip():
            logger.warning("No task data found; skipping model annotation.")
            return

        labels = self._codelist_entries.get_labels()
        if not labels:
            logger.error("No concepts found in codelist; skipping model annotation.")
            return

        labels_for_prompt = self._codelist_entries.get_labels_with_definitions()

        classes: list[str] = []
        if self._provider == "random":
            logger.warning("Using random label (provider=random).")
            classes = [random.choice(labels)]
        elif self._llm is None:
            logger.error("No LLM client available; skipping model annotation.")
            return
        else:
            max_retries = 3
            llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                     user_message=self._llm_user_message.format(
                                         code_list=labels_for_prompt, decision_text=task_data),
                                     assistant_message=None,
                                     output_format=EntityLinkingTaskOutput)


            for attempt in range(1, max_retries + 1):
                try:
                    response = self._llm(llm_input)
                    classes = response.designated_classes
                    
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        raise RuntimeError(f"LLM call failed after {max_retries} attempts ({exc}); skipping annotation.")
                    else:
                        logger.warning(f"LLM call attempt {attempt}/{max_retries} failed ({exc}); retrying.")
                        time.sleep(attempt)

        logger.warning(f"LLM returned classes: {classes}")

        for c in classes:
            concept_uri = self._codelist_entries.resolve_label_to_uri(c, self._label_to_uri)
            if not concept_uri:
                logger.warning(f"No URI found for class '{c}', skipping annotation.")
                continue

            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                concept_uri,
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
            logger.warning("Created SDG annotation")

        if classes:
            self.results_container_uris.append(self.create_output_container(self.source))
        else:
            self.store_no_match()

        rate_limit_delay = float(os.environ.get("RATE_LIMIT_DELAY_SECONDS", "0"))
        logger.warning(f"[RATE-LIMIT] Waiting for {rate_limit_delay} seconds to respect rate limits.")
        print(f"Waiting for {rate_limit_delay} seconds to respect rate limits.", flush=True)
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
    
    def store_no_match(self):
        uri = f"http://mu.semte.ch/vocabularies/ext/no-match-found"

        try:
            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                uri,
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
            self.results_container_uris.append(self.create_output_container(self.source))
        except Exception as e:
            error_msg = f"Failed to insert no-match-found: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            

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

class ModelBatchAnnotatingTask(CodeListTask):
    """Task that creates ModelAnnotatingTasks for all decisions that are not yet annotated."""

    __task_type__ = TASK_OPERATIONS["codelist_annotation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):        
        codelist_entries = self.fetch_codelist()
        target_graph = self.get_target_graph()
        target_nodes, target_classes = self.fetch_shape_targets()
        property_path_for_text = self.fetch_property_path_for_text()

        decision_uris = self.fetch_decisions_without_annotations(
            concept_scheme_uri=codelist_entries.concept_scheme_uri,
            target_graph=target_graph,
            target_nodes=target_nodes,
            target_classes=target_classes,
        )
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            task = ModelAnnotatingTask(
                self.task_uri,
                source=decision_uri,
                codelist_entries=codelist_entries,
                property_path_for_text=property_path_for_text,
            )
            task.process()
            self.results_container_uris.extend(task.results_container_uris)

            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    def fetch_decisions_without_annotations(
        self,
        concept_scheme_uri: str,
        target_graph: str | None = None,
        target_nodes: list[str] | None = None,
        target_classes: list[str] | None = None,
    ) -> list[str]:
        """Fetch decision URIs that have no classifying annotation for the given concept scheme.

        Uses ext:shapeForTargets to determine which decisions to consider:
          - target_nodes (from sh:targetNode): specific decision URIs
          - target_classes (from sh:targetClass): all instances of the given classes
          - Neither: defaults to eli:Expression

        target_graph is optional; when not set, searches across all graphs.
        """
        # Build the target pattern based on SHACL shape configuration
        if target_nodes and target_classes:
            node_values = " ".join(sparql_escape_uri(n) for n in target_nodes)
            class_values = " ".join(sparql_escape_uri(c) for c in target_classes)
            target_pattern = (
                f"{{ VALUES ?s {{ {node_values} }} }}\n"
                f"UNION\n"
                f"{{ ?s rdf:type ?targetClass . VALUES ?targetClass {{ {class_values} }} }}"
            )
        elif target_nodes:
            node_values = " ".join(sparql_escape_uri(n) for n in target_nodes)
            target_pattern = f"VALUES ?s {{ {node_values} }}"
        elif target_classes:
            class_values = " ".join(sparql_escape_uri(c) for c in target_classes)
            target_pattern = f"?s rdf:type ?targetClass . VALUES ?targetClass {{ {class_values} }}"
        else:
            target_pattern = "?s rdf:type eli:Expression ."

        # Build the graph wrapper — optional when target_graph is not set
        if target_graph:
            target_clause = f"GRAPH {sparql_escape_uri(target_graph)} {{ {target_pattern} }}"
        else:
            target_clause = target_pattern

        # Build the FILTER NOT EXISTS graphs to check
        if target_graph:
            filter_graph_values = f"VALUES ?g {{ {sparql_escape_uri(target_graph)} {sparql_escape_uri(GRAPHS['ai'])} }}"
        else:
            filter_graph_values = f"VALUES ?g {{ {sparql_escape_uri(GRAPHS['ai'])} }}"

        expression_filter = self.get_expressions_in_task_filter()
        q = Template(get_prefixes_for_query("rdf", "eli", "oa", "skos", "ext") + """
        SELECT DISTINCT ?s
        WHERE {
            $expression_filter
            $target_clause
            FILTER NOT EXISTS {
                $filter_graph_values
                GRAPH ?g {
                    ?ann a oa:Annotation ;
                         oa:hasTarget ?s ;
                         oa:motivatedBy oa:classifying ;
                         oa:hasBody ?concept .
                    ?concept skos:inScheme|ext:forConceptScheme $concept_scheme_uri .
                }
            }
        }
        """).substitute(
            expression_filter=expression_filter,
            target_graph=sparql_escape_uri(target_graph), 
            ai_graph=sparql_escape_uri(GRAPHS['ai']),
            concept_graph=sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public")),
            concept_scheme_uri=sparql_escape_uri(concept_scheme_uri),
            target_clause=target_clause,
            filter_graph_values=filter_graph_values
        )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        return [b["s"]["value"] for b in bindings if "s" in b]