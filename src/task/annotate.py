import logging
import random
import time
import uuid
from string import Template
from helpers import query, update
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

    def __init__(self, task_uri: str, source: str, codelist_entries: Codelist):
        super().__init__(task_uri)
        self.source = source

        config = get_config()

        self._codelist_entries = codelist_entries
        self._label_to_uri = self._codelist_entries.build_label_to_uri_map()

        # LLM setup
        self._llm = create_llm_client(config.llm)
        self._provider = config.llm.provider

        self._llm_system_message = "You are a juridical and administrative assistant that must determine the best matching codes from a list with a given text."
        self._llm_user_message = "Determine the best matching codes from the following list for the given public decision.\n\n" \
            "\"\"\"" \
            "CODE LIST:\n" \
            "{code_list}\n" \
            "\"\"\"\n\n" \
            "\"\"\"" \
            "DECISION TEXT:\n" \
            "{decision_text}\n" \
            "\"\"\"" \
            "Provide your answer as a list of strings representing the matching codes. Provide all matching codes (can be a single one), but only those that are truly matching and only from the given list! If none of the codes match, return an empty list."

    def process(self):
        # TODO: read ext:propertyPathForText from the job and pass it to fetch_data() instead of hardcoding epvoc:expressionContent
        task_data = self.fetch_data()
        self.logger.info(task_data)

        if not task_data.strip():
            self.logger.warning(
                "No task data found; skipping model annotation.")
            return

        labels = [entry.label for entry in self._codelist_entries]
        if not labels:
            self.logger.error("No concepts found in codelist; skipping model annotation.")
            return

        classes: list[str] = []
        if self._provider == "random":
            self.logger.warning("Using random label (provider=random).")
            classes = [random.choice(labels)]
        elif self._llm is None:
            self.logger.error("No LLM client available; skipping model annotation.")
            return
        else:
            max_retries = 3
            llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                     user_message=self._llm_user_message.format(
                                         code_list=labels, decision_text=task_data),
                                     assistant_message=None,
                                     output_format=EntityLinkingTaskOutput)

            for attempt in range(1, max_retries + 1):
                try:
                    response = self._llm(llm_input)
                    classes = response.designated_classes
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        self.logger.warning(
                            f"LLM call failed after {max_retries} attempts ({exc}); skipping annotation.")
                    else:
                        self.logger.warning(
                            f"LLM call attempt {attempt}/{max_retries} failed ({exc}); retrying.")
                        time.sleep(attempt)

        self.logger.warning(f"LLM returned classes: {classes}")

        for c in classes:
            concept_uri = self._codelist_entries.resolve_label_to_uri(c, self._label_to_uri)
            if not concept_uri:
                self.logger.warning(f"No URI found for class '{c}', skipping annotation.")
                continue

            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                concept_uri,
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
            self.logger.warning("Created SDG annotation")

        if classes:
            self.results_container_uris.append(self.create_output_container(self.source))
        else:
            self.store_no_match()

        
    
    def store_no_match(self):
        id = uuid.uuid4()
        uri = f"http://mu.semte.ch/vocabularies/ext/no-match-found/id/{id}"
        query_string = Template(get_prefixes_for_query("ext", "mu") +
        
        """
        INSERT DATA {
            GRAPH $graph {
                $uri a ext:NoMatchFound ;
                     mu:uuid $id ;
                     ext:forConceptScheme $concept_scheme .
            }
        }
        """
        ).substitute(
            graph=sparql_escape_uri(GRAPHS['ai']),
            uri=sparql_escape_uri(uri),
            id=sparql_escape_string(id),
            concept_scheme=sparql_escape_uri(self._codelist_entries.concept_scheme_uri)
        )

        try:
            update(query_string, sudo=True)

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
            self.logger.error(error_msg, exc_info=True)
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
            f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                $container a nfo:DataContainer ;
                    mu:uuid "$uuid" ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=container_id,
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
        decision_uris = self.fetch_decisions_without_annotations(target_graph, codelist_entries.concept_scheme_uri)
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            task = ModelAnnotatingTask(self.task_uri, decision_uri, codelist_entries=codelist_entries)
            task.process()
            self.results_container_uris.extend(task.results_container_uris)

            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)
    

    @staticmethod
    def fetch_decisions_without_annotations(target_graph: str, concept_scheme_uri: str) -> list[str]:
        # TODO: use ext:shapeForTargets (and optionally ext:graphForTargets) from the job to scope which decisions to fetch
        q = Template(get_prefixes_for_query("rdf", "eli", "oa", "skos", "ext") + """
        SELECT DISTINCT ?s
        WHERE {
            GRAPH $target_graph {
                ?s rdf:type eli:Expression .
            }
            FILTER NOT EXISTS {
                VALUES ?g { $target_graph $ai_graph }
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
            target_graph=sparql_escape_uri(target_graph), 
            ai_graph=sparql_escape_uri(GRAPHS['ai']),
            concept_graph=sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public")),
            concept_scheme_uri=sparql_escape_uri(concept_scheme_uri)
        )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        return [b["s"]["value"] for b in bindings if "s" in b]