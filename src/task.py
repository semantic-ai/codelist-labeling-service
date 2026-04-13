import logging
import random
from string import Template
from helpers import query, update

from decide_ai_service_base.task import DecisionTask, Task
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, get_prefixes_for_query
from decide_ai_service_base.annotation import LinkingAnnotation

from .llm_models.llm_model_clients import create_llm_client
from .llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput
from .classifier.train import train
from .codelist import CodelistEntry, fetch_codelist, fetch_codelist_uri_for_task, build_label_to_uri_map, resolve_label_to_uri
from .config import get_config


class ModelAnnotatingTask(DecisionTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str, codelist_entries: list[CodelistEntry]):
        super().__init__(task_uri)
        self.source = source

        config = get_config()

        self._codelist_entries = codelist_entries
        self._label_to_uri = build_label_to_uri_map(self._codelist_entries)

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
            concept_uri = resolve_label_to_uri(c, self._label_to_uri, self._codelist_entries)
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

    __task_type__ = "http://lblod.data.gift/id/jobs/concept/TaskOperation/codelist-matching/annotate"

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def get_target_graph(self) -> str | None:
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
        if bindings:
            return bindings[0]["graph"]["value"]
        return None

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
        codelist_entries = fetch_codelist(job_codelist)

        target_graph = self.get_target_graph()
        decision_uris = self.fetch_decisions_without_annotations(target_graph)
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            ModelAnnotatingTask(self.task_uri, decision_uri, codelist_entries=codelist_entries).process()
            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    def fetch_decisions_without_annotations(self, target_graph: str) -> list[str]:
        target_graph_pattern = f"GRAPH <{target_graph}>" if target_graph else "GRAPH ?dataGraph"

        q = Template(
            get_prefixes_for_query("rdf", "eli", "oa") + f"""
            SELECT DISTINCT ?s
            WHERE {{
                GRAPH <$graph> {{
                    ?s rdf:type eli:Expression .
                }}
                FILTER NOT EXISTS {{
                    GRAPH <$graph> {{
                    ?ann a oa:Annotation ;
                        oa:hasTarget ?s ;
                        oa:motivatedBy oa:classifying .
                    }}
                }}
            }}
            """
        ).substitute(graph=target_graph)

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        decision_uris = [b["s"]["value"] for b in bindings if "s" in b]

        return decision_uris


class ClassifierTrainingTask(Task):
    """Task that trains a classifier for the available annotations in the triple store."""

    __task_type__ = TASK_OPERATIONS["classifier_training"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        concept_scheme_uri = fetch_codelist_uri_for_task(self.task_uri)
        codelist_entries = fetch_codelist(concept_scheme_uri)
        labels = [entry.label for entry in codelist_entries]
        uri_to_label = {entry.uri: entry.label for entry in codelist_entries}

        decisions = self.fetch_decisions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions, uri_to_label)

        decisions = [d for d in decisions if d.get("classes")]
        if not decisions:
            print("No labeled decisions found; skipping training.", flush=True)
            return

        ml_config = get_config().ml_training

        print("Started training...", flush=True)
        train(
            decisions[:10],
            labels,
            ml_config.huggingface_output_model_id,
            transformer=ml_config.transformer,
            learning_rate=ml_config.learning_rate,
            epochs=ml_config.epochs,
            weight_decay=ml_config.weight_decay,
        )
        print("Done training!", flush=True)

    def convert_classes_to_original_names(self, decisions: list[dict[str, str | list[str]]], uri_to_label: dict[str, str]):
        for decision in decisions:
            decision["classes"] = [
                uri_to_label.get(c, c) for c in decision["classes"]
            ]
        return decisions

    def fetch_decisions_with_classes(self) -> list[dict[str, str | list[str]]]:
        q = get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct") + """
        SELECT ?decision ?title ?description ?decision_basis ?content ?classes
        WHERE {
        {
            SELECT ?decision (GROUP_CONCAT(DISTINCT STR(?body); separator="|") AS ?classes)
            WHERE {
                GRAPH <http://mu.semte.ch/graphs/ai> {
                    ?ann a oa:Annotation ;
                        oa:hasTarget ?decision ;
                        oa:motivatedBy oa:classifying ;
                        oa:hasBody ?body .
                }
            }
            GROUP BY ?decision
        }
            GRAPH ?dataGraph {
                ?decision rdf:type eli:Expression .
                OPTIONAL { ?decision eli:title ?title }
                OPTIONAL { ?decision eli:description ?description }
                OPTIONAL { ?decision eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?decision epvoc:expressionContent ?content }
                OPTIONAL { ?decision dct:language ?lang }
            }
        }
        """

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        results = []
        for b in bindings:
            decision = b["decision"]["value"]
            classes_concat = b.get("classes", {}).get("value", "")
            classes = [c for c in classes_concat.split("|") if c]
            title = b.get("title", {}).get("value", "")
            description = b.get("description", {}).get("value", "")
            decision_basis = b.get("decision_basis", {}).get("value", "")
            content = b.get("content", {}).get("value", "")

            text = "\n".join(
                [t for t in [title, description, decision_basis, content] if t])

            results.append({
                "decision": decision,
                "classes": classes,
                "text": text
            })

        return results
