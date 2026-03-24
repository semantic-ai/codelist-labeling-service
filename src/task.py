import logging
import random
from helpers import query, update

from decide_ai_service_base.task import DecisionTask, Task
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, get_prefixes_for_query
from decide_ai_service_base.annotation import LinkingAnnotation

from .llm_models.llm_model_clients import OpenAIModel
from .llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput
from .classifier.train import train
from .config import get_config


class ModelAnnotatingTask(DecisionTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str | None = None):
        if source is None:
            super().__init__(task_uri)
        else:
            self.task_uri = task_uri
            self.logger = logging.getLogger(self.__class__.__name__)
            self.source = source

        config = get_config()
        self._llm_config = {
            "model_name": config.llm.model_name,
            "temperature": config.llm.temperature,
        }

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
            "Provide your answer as a list of strings representing the matching codes. Provide all matching codes (can be a single one), but only those that are truly matching and only from the given list!"

        self._llm = OpenAIModel(self._llm_config)

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)

        if not task_data.strip():
            self.logger.warning(
                "No task data found; skipping model annotation.")
            return

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["SDG-01 No Poverty",
                "SDG-02 Zero Hunger",
                "SDG-03 Good Health and Well-Being",
                "SDG-04 Quality Education",
                "SDG-05 Gender Equality",
                "SDG-06 Clean Water and Sanitation",
                "SDG-07 Affordable and Clean Energy",
                "SDG-08 Decent Work and Economic Growth",
                "SDG-09 Industry, Innovation and Infrastructure",
                "SDG-10 Reduced Inequality",
                "SDG-11 Sustainable Cities and Communities",
                "SDG-12 Responsible Consumption and Production",
                "SDG-13 Climate Action",
                "SDG-14 Life Below Water",
                "SDG-15 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]

        classes: list[str] = []
        config = get_config()
        api_key = config.llm.api_key.get_secret_value() if config.llm.api_key else None
        if not api_key:
            self.logger.warning(
                "OpenAI API key missing (config.llm.api_key), using dummy SDG label for testing.")
            classes = [random.choice(sdgs).replace(" ", "_")]
        else:
            try:
                llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                         user_message=self._llm_user_message.format(
                                             code_list=sdgs, decision_text=task_data),
                                         assistant_message=None,
                                         output_format=EntityLinkingTaskOutput)

                response = self._llm(llm_input)
                classes = [designated_class.replace(
                    " ", "_") for designated_class in response.designated_classes]
            except Exception as exc:
                self.logger.warning(
                    f"LLM call failed ({exc}); using dummy SDG label for testing.")
                classes = [random.choice(sdgs).replace(" ", "_")]

        for c in classes:
            annotation = LinkingAnnotation(
                self.task_uri,
                self.source,
                # TO DO: CHANGE TO ACTUAL URI
                "http://example.org/" + c,
                AI_COMPONENTS["model_annotater"],
                AGENT_TYPES["ai_component"]
            )
            annotation.add_to_triplestore_if_not_exists()


class ModelBatchAnnotatingTask(Task):
    """Task that creates ModelAnnotatingTasks for all decisions that are not yet annotated."""

    __task_type__ = TASK_OPERATIONS["model_batch_annotation"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        decision_uris = self.fetch_decisions_without_annotations()
        print(f"{len(decision_uris)} decisions to process.", flush=True)

        for i, decision_uri in enumerate(decision_uris):
            ModelAnnotatingTask(self.task_uri, decision_uri).process()
            print(
                f"Processed decision {i+1}/{len(decision_uris)}: {decision_uri}", flush=True)

    def fetch_decisions_without_annotations(self) -> list[str]:
        q = get_prefixes_for_query("rdf", "eli", "oa") + """
        SELECT DISTINCT ?s
        WHERE {
            GRAPH ?dataGraph {
                ?s rdf:type eli:Expression .
            }
            FILTER NOT EXISTS {
                GRAPH <http://mu.semte.ch/graphs/ai> {
                ?ann a oa:Annotation ;
                    oa:hasTarget ?s ;
                    oa:motivatedBy oa:classifying .
                }
            }
        }
        """

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
        decisions = self.fetch_decisions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions)

        decisions = [d for d in decisions if d.get("classes")]
        if not decisions:
            print("No labeled decisions found; skipping training.", flush=True)
            return

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["SDG-01 No Poverty",
                "SDG-02 Zero Hunger",
                "SDG-03 Good Health and Well-Being",
                "SDG-04 Quality Education",
                "SDG-05 Gender Equality",
                "SDG-06 Clean Water and Sanitation",
                "SDG-07 Affordable and Clean Energy",
                "SDG-08 Decent Work and Economic Growth",
                "SDG-09 Industry, Innovation and Infrastructure",
                "SDG-10 Reduced Inequality",
                "SDG-11 Sustainable Cities and Communities",
                "SDG-12 Responsible Consumption and Production",
                "SDG-13 Climate Action",
                "SDG-14 Life Below Water",
                "SDG-15 Life on Land",
                "SDG-16 Peace, Justice and Strong Institutions",
                "SDG-17 Partnerships for the Goals"
                ]

        config = get_config()
        ml_config = config.ml_training

        print("Started training...", flush=True)
        train(
            decisions[:10],
            sdgs,
            ml_config.huggingface_output_model_id,
            transformer=ml_config.transformer,
            learning_rate=ml_config.learning_rate,
            epochs=ml_config.epochs,
            weight_decay=ml_config.weight_decay,
        )
        print("Done training!", flush=True)

    def convert_classes_to_original_names(self, decisions: list[dict[str, str | list[str]]]):
        for decision in decisions:
            decision["classes"] = [
                c.split("/")[-1].replace("_", " ") for c in decision["classes"]]

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
