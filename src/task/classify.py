from string import Template
from helpers import query
from decide_ai_service_base.sparql_config import AGENT_TYPES, get_prefixes_for_query
from decide_ai_service_base.annotation import LinkingAnnotation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from escape_helpers import sparql_escape_uri

from .codelist import CodeListTask
from ..classifier.predict import predict as classifier_predict
from ..config import get_config


class ClassifierAnnotatingTask(CodeListTask):
    """Runs a trained HuggingFace classifier on unlabeled decisions to produce codelist annotations at scale."""

    __task_type__ = "http://lblod.data.gift/id/jobs/concept/TaskOperation/codelist-matching/classifier-annotate"

    # AI_COMPONENTS in decide-ai-service-base has no classifier entry yet;
    # hardcoded so classifier annotations are distinguishable from LLM annotations in provenance queries.
    _CLASSIFIER_AGENT_URI = "http://data.lblod.info/id/ai-components/classifier-annotation"

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
        return bindings[0]["graph"]["value"] if bindings else None

    def get_job_confidence_threshold(self) -> float | None:
        q = Template(
            """
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            SELECT ?threshold WHERE {
                $task dct:isPartOf ?job .
                ?job ext:confidenceThreshold ?threshold .
            }
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])
        if bindings:
            try:
                return float(bindings[0]["threshold"]["value"])
            except (KeyError, ValueError):
                pass
        return None

    def fetch_decisions_without_annotations_with_text(self, target_graph: str) -> list[dict]:
        q = Template(
            get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct") + """
            SELECT DISTINCT ?s ?title ?description ?decision_basis ?content
            WHERE {
                GRAPH $graph {
                    ?s rdf:type eli:Expression .
                    OPTIONAL { ?s eli:title ?title }
                    OPTIONAL { ?s eli:description ?description }
                    OPTIONAL { ?s eli-dl:decision_basis ?decision_basis }
                    OPTIONAL { ?s epvoc:expressionContent ?content }
                }
                FILTER NOT EXISTS {
                    GRAPH $graph {
                        ?ann a oa:Annotation ;
                             oa:hasTarget ?s ;
                             oa:motivatedBy oa:classifying .
                    }
                }
            }
            """
        ).substitute(graph=sparql_escape_uri(target_graph))

        response = query(q, sudo=True)
        results = []
        for b in response.get("results", {}).get("bindings", []):
            text = "\n".join(
                t for t in [
                    b.get("title", {}).get("value", ""),
                    b.get("description", {}).get("value", ""),
                    b.get("decision_basis", {}).get("value", ""),
                    b.get("content", {}).get("value", ""),
                ] if t
            )
            results.append({"uri": b["s"]["value"], "text": text})
        return results

    def process(self):
        config = get_config()
        inference_cfg = config.ml_inference

        if not inference_cfg.huggingface_model_id:
            raise RuntimeError(
                "ml_inference.huggingface_model_id is not set in config.json. "
                "Run ClassifierTrainingTask first and set the output model ID here."
            )

        target_graph = self.get_target_graph()
        if not target_graph:
            raise RuntimeError(f"No ext:graphForTargets found for task {self.task_uri}")

        job_threshold = self.get_job_confidence_threshold()
        confidence_threshold = (
            job_threshold if job_threshold is not None else inference_cfg.confidence_threshold
        )

        hf_token = (
            inference_cfg.huggingface_token.get_secret_value()
            if inference_cfg.huggingface_token
            else None
        )

        tokenizer = AutoTokenizer.from_pretrained(inference_cfg.huggingface_model_id, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(
            inference_cfg.huggingface_model_id, token=hf_token
        )
        model.eval()

        problem_type = getattr(model.config, "problem_type", None)
        if problem_type not in ("single_label_classification", "multi_label_classification"):
            raise RuntimeError(f"Unexpected problem_type={problem_type!r} on loaded model.")

        id2label: dict[int, str] = model.config.id2label

        codelist = self.fetch_codelist()
        label_to_uri = codelist.build_label_to_uri_map()

        decisions = self.fetch_decisions_without_annotations_with_text(target_graph)
        print(f"{len(decisions)} decisions to classify.", flush=True)

        for i, decision in enumerate(decisions):
            uri, text = decision["uri"], decision["text"]
            if not text.strip():
                self.logger.warning("Decision %s has no text; skipping.", uri)
                continue

            try:
                predictions = classifier_predict(
                    text, model, tokenizer, id2label, problem_type, confidence_threshold
                )
            except Exception as exc:
                self.logger.error("Inference failed for %s: %s", uri, exc, exc_info=True)
                continue

            if not predictions:
                continue

            for label, _conf in predictions:
                concept_uri = codelist.resolve_label_to_uri(label, label_to_uri)
                if not concept_uri:
                    self.logger.warning("No URI for label %r; skipping.", label)
                    continue
                annotation = LinkingAnnotation(
                    self.task_uri,
                    uri,
                    concept_uri,
                    self._CLASSIFIER_AGENT_URI,
                    AGENT_TYPES["ai_component"],
                )
                annotation.add_to_triplestore_if_not_exists()

            print(
                f"Classified {i+1}/{len(decisions)}: {uri} → {[l for l, _ in predictions]}",
                flush=True,
            )
