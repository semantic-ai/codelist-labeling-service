import logging
import random
import time
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


class ModelAnnotatingTask(DecisionTask):
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

        if not classes:
            # Temporarily link to a "no match found" concept to avoid re-processing the same decision.
            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                "http://data.lblod.gift/id/concept/no-match-found",
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()
    
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
        except Exception as e:
            error_msg = f"Failed to insert no-match-found: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        else:
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


class ModelBatchAnnotatingTask(Task):
    """Task that creates ModelAnnotatingTasks for all decisions that are not yet annotated."""

    __task_type__ = TASK_OPERATIONS["codelist_annotation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def get_target_graph(self) -> str:
        q = f"""
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        SELECT ?graph WHERE {{
            <{self.task_uri}> dct:isPartOf ?job .
            ?job ext:graphForTargets ?graph .
        }}
        """
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if not bindings:
            raise RuntimeError(f"No ext:graphForTargets found for task {self.task_uri}")
        return bindings[0]["graph"]["value"]

    def get_codelist(self) -> str | None:
        q = f"""
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        SELECT ?codelist WHERE {{
            <{self.task_uri}> dct:isPartOf ?job .
            ?job ext:codelist ?codelist .
        }}
        """
        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["codelist"]["value"]
        return None

    def process(self):
        job_codelist = self.get_codelist()
        if not job_codelist:
            raise RuntimeError(f"No ext:codelist found for task {self.task_uri}")
        codelist_entries = Codelist.from_uri(job_codelist)
        target_graph = self.get_target_graph()
        decision_uris = self.fetch_decisions_without_annotations(target_graph)
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            ModelAnnotatingTask(self.task_uri, decision_uri, codelist_entries=codelist_entries).process()
            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)
    

    @staticmethod
    def fetch_decisions_without_annotations(self, target_graph: str) -> list[str]:
        # TODO this one should accept a codelist and filter on annotations to concepts of that codelist (or no match founds for that codelist, else you can only annotate with one codelist)
        # TODO: use ext:shapeForTargets (and optionally ext:graphForTargets) from the job to scope which decisions to fetch
        q = Template(get_prefixes_for_query("rdf", "eli", "oa") + """
        SELECT DISTINCT ?s
        WHERE {
            GRAPH <$target_graph> {
                ?s rdf:type eli:Expression .
            }
            FILTER NOT EXISTS {
                {
                    GRAPH <$target_graph> {
                        ?ann a oa:Annotation ;
                            oa:hasTarget ?s ;
                            oa:motivatedBy oa:classifying .
                    }
                } UNION {
                    GRAPH $ai_graph {
                        ?ann a oa:Annotation ;
                            oa:hasTarget ?s ;
                            oa:motivatedBy oa:classifying .
                    }
                }
            }
        }
        """).substitute(target_graph=target_graph, ai_graph=sparql_escape_uri(GRAPHS['ai']))

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        return [b["s"]["value"] for b in bindings if "s" in b]