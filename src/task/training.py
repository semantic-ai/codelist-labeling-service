from helpers import query, update

from decide_ai_service_base.task import Task
from decide_ai_service_base.sparql_config import TASK_OPERATIONS, get_prefixes_for_query

from ..classifier.train import train
from ..codelist import fetch_codelist
from ..config import get_config


class ClassifierTrainingTask(Task):
    """Task that trains a classifier for the available annotations in the triple store."""

    __task_type__ = TASK_OPERATIONS["classifier_training"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def process(self):
        config = get_config()
        codelist_entries = fetch_codelist(config.codelist.concept_scheme_uri)
        labels = [entry.label for entry in codelist_entries]
        uri_to_label = {entry.uri: entry.label for entry in codelist_entries}

        decisions = self.fetch_decisions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions, uri_to_label)

        decisions = [d for d in decisions if d.get("classes")]
        if not decisions:
            print("No labeled decisions found; skipping training.", flush=True)
            return

        ml_config = config.ml_training

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
