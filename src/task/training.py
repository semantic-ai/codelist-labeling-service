from helpers import query, update
from escape_helpers import sparql_escape_uri

from string import Template

from decide_ai_service_base.sparql_config import TASK_OPERATIONS, get_prefixes_for_query, GRAPHS

from ..classifier.train import train
from ..config import get_config
from .codelist import CodeListTask, Codelist


class ClassifierTrainingTask(CodeListTask):
    """Task that trains a classifier for the available annotations in the triple store."""

    __task_type__ = TASK_OPERATIONS["codelist_training"]

    def process(self):
        codelist_entries = self.fetch_codelist()

        decisions = self.fetch_decisions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions, codelist_entries)

        decisions = [d for d in decisions if d.get("classes")]
        if not decisions:
            print("No labeled decisions found; skipping training.", flush=True)
            return

        ml_config = get_config().ml_training

        print("Started training...", flush=True)
        train(
            decisions,
            codelist_entries.get_labels(),
            ml_config.huggingface_output_model_id,
            transformer=ml_config.transformer,
            learning_rate=ml_config.learning_rate,
            epochs=ml_config.epochs,
            weight_decay=ml_config.weight_decay,
        )
        print("Done training!", flush=True)

    @staticmethod
    def convert_classes_to_original_names(decisions: list[dict[str, str | list[str]]], codelist: Codelist):
        uri_to_label = codelist.build_uri_to_label_map()
        for decision in decisions:
            decision["classes"] = [
                uri_to_label.get(c, c) for c in decision["classes"]
            ]
        return decisions

    def fetch_decisions_with_classes(self) -> list[dict[str, str | list[str]]]:
        q = Template(get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct", "skos") + """
        SELECT ?decision ?title ?description ?decision_basis ?content ?classes
        WHERE {
        {
            SELECT ?decision (GROUP_CONCAT(DISTINCT STR(?body); separator="|") AS ?classes)
            WHERE {
                GRAPH $ai_graph {
                    ?ann a oa:Annotation ;
                        oa:hasTarget ?decision ;
                        oa:motivatedBy oa:classifying ;
                        oa:hasBody ?body .
                }
                
                GRAPH $public_graph {
                    ?body a skos:Concept ;
                          skos:inScheme ?scheme .
                  }
            
                  VALUES ?scheme {
                    $concept_scheme_uri
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
        """).substitute(
            ai_graph=sparql_escape_uri(GRAPHS['ai']),
            public_graph=sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public")),
            concept_scheme_uri=sparql_escape_uri(self.fetch_codelist_uri_for_task()),
        )

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
