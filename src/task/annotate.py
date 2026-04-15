import logging
import random
from string import Template
from helpers import query, update
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.task import DecisionTask, Task
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, get_prefixes_for_query, \
    GRAPHS
from decide_ai_service_base.annotation import LinkingAnnotation

from ..llm_models.llm_model_clients import create_llm_client
from ..llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput
from .codelist import CodelistEntry, CodeListTask
from ..config import get_config


class ModelAnnotatingTask(CodeListTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str | None = None, codelist_entries: list[CodelistEntry] | None = None):
        if source is None:
            super().__init__(task_uri)
        else:
            self.task_uri = task_uri
            self.logger = logging.getLogger(self.__class__.__name__)
            self.source = source

        config = get_config()

        # Codelist: use provided entries or resolve dynamically from the job
        self._codelist_entries = codelist_entries if codelist_entries is not None else self.fetch_codelist()
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
        if self._provider == "random" or self._llm is None:
            self.logger.warning("Using random label (provider=random).")
            classes = [random.choice(labels)]
        else:
            try:
                llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                         user_message=self._llm_user_message.format(
                                             code_list=labels, decision_text=task_data),
                                         assistant_message=None,
                                         output_format=EntityLinkingTaskOutput)

                response = self._llm(llm_input)
                classes = response.designated_classes
            except Exception as exc:
                self.logger.warning(
                    f"LLM call failed ({exc}); using random label as fallback.")
                classes = [random.choice(labels)]

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


class ModelBatchAnnotatingTask(CodeListTask):
    """Task that creates ModelAnnotatingTasks for all decisions that are not yet annotated."""

    __task_type__ = TASK_OPERATIONS["model_batch_annotation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        codelist_entries = self.fetch_codelist()
        decision_uris = self.fetch_decisions_without_annotations()
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            ModelAnnotatingTask(self.task_uri, decision_uri, codelist_entries=codelist_entries).process()
            print(f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    @staticmethod
    def fetch_decisions_without_annotations() -> list[str]:
        # TODO: use ext:shapeForTargets (and optionally ext:graphForTargets) from the job to scope which decisions to fetch
        q = Template(get_prefixes_for_query("rdf", "eli", "oa") + """
        SELECT DISTINCT ?s
        WHERE {
            GRAPH ?dataGraph {
                ?s rdf:type eli:Expression .
            }
            FILTER NOT EXISTS {
                GRAPH $graph {
                ?ann a oa:Annotation ;
                    oa:hasTarget ?s ;
                    oa:motivatedBy oa:classifying .
                }
            }
        }
        """).substitute(
            graph=sparql_escape_uri(GRAPHS['ai'])
        )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        decision_uris = [b["s"]["value"] for b in bindings if "s" in b]

        return decision_uris