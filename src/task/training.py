import json
from datetime import datetime
from pathlib import Path
from typing import Any

from helpers import query, update, logger
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

        decisions = self.fetch_actions_with_classes()
        decisions = self.convert_classes_to_original_names(decisions, codelist_entries)

        if not decisions:
            logger.warning(
                "No labeled decisions found for training task %s; skipping.",
                self.task_uri,
            )
            return

        ml_config = get_config().ml_training

        self.save_dataset_jsonl(decisions)

        logger.info(
            "Starting classifier training task %s with %d decisions and %d labels",
            self.task_uri,
            len(decisions),
            len(codelist_entries),
        )
        train(
            decisions,
            codelist_entries.get_labels(),
            ml_config.huggingface_output_model_id,
            concept_scheme_uri=codelist_entries.concept_scheme_uri,
            transformer=ml_config.model_name,
            learning_rate=ml_config.learning_rate,
            epochs=ml_config.num_train_epochs,
            weight_decay=ml_config.weight_decay,
        )
        logger.info("Completed classifier training task %s", self.task_uri)

    # Repo root (src/task/training.py -> src/task -> src -> repo root), so the path
    # is independent of the process's cwd (which may not be the app/repo dir).
    _REPO_ROOT = Path(__file__).resolve().parents[2]

    def save_dataset_jsonl(self, decisions: list[dict], output_dir: str | Path | None = None) -> Path:
        """Save training decisions to a JSONL file."""
        dir_path = Path(output_dir) if output_dir is not None else self._REPO_ROOT / "data"
        dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"training_dataset_{timestamp}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for decision in decisions:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")
        logger.info("Saved training dataset (%d samples) to %s", len(decisions), file_path)
        return file_path

    @staticmethod
    def convert_classes_to_original_names(decisions: list[dict], codelist: Codelist):
        uri_to_label = codelist.build_uri_to_label_map()
        for decision in decisions:
            decision["classes"] = [
                uri_to_label.get(c, c) for c in decision["classes"]
            ]
            if "label_votes" in decision:
                decision["label_votes"] = {
                    uri_to_label.get(class_uri, class_uri): votes
                    for class_uri, votes in decision["label_votes"].items()
                }
        return decisions
    
    @staticmethod
    def _assemble_action_text(binding: dict) -> str:
        """Assemble training text for an action: plan context + action content."""
        parts: list[str] = []

        # Plan context header
        plan_title = binding.get("plan_title", {}).get("value", "")
        plan_desc = binding.get("plan_description", {}).get("value", "")
        plan_code = binding.get("plan_code", {}).get("value", "")
        if plan_title or plan_code:
            header = f"[Plan: {plan_title}]" if plan_title else ""
            if plan_code:
                header = f"[Plan: {plan_code} {plan_title}]"
            parts.append(header)
        if plan_desc:
            parts.append(plan_desc)

        # Action's own text
        action_code = binding.get("action_code", {}).get("value", "")
        action_title = binding.get("action_title", {}).get("value", "")
        if action_code or action_title:
            parts.append(f"{action_code} {action_title}".strip())

        action_desc = binding.get("action_description", {}).get("value", "")
        if action_desc:
            parts.append(action_desc)

        action_content = binding.get("action_content", {}).get("value", "")
        if action_content:
            parts.append(action_content)

        return "\n".join(parts)

    @staticmethod
    def _extract_classes(binding: dict, excluded_classes: set[str] | None = None) -> list[str]:
        classes_concat = binding.get("classes", {}).get("value", "")
        excluded = excluded_classes or set()
        return [c for c in classes_concat.split("|") if c and c not in excluded]

    @staticmethod
    def _keep_approved_labels(sample: dict[str, Any]) -> dict[str, Any]:
        """Return a copy containing only labels with at least one approval vote."""

        approved_vote_uri = "http://mu.semte.ch/vocabularies/ext/annotation-review#approve"

        approved_classes = [
            class_uri
            for class_uri in sample["classes"]
            if approved_vote_uri in sample["label_votes"].get(class_uri, [])
        ]
        return {
            **sample,
            "classes": approved_classes,
            "label_votes": {
                class_uri: sample["label_votes"][class_uri]
                for class_uri in approved_classes
            },
        }


    def _build_training_sample(
        self,
        binding: dict,
        decision_key: str,
        text: str,
        excluded_classes: set[str] | None = None,
    ) -> dict[str, str | list[str]]:
        return {
            "decision": binding[decision_key]["value"],
            "classes": self._extract_classes(binding, excluded_classes),
            "text": text,
        }


    def fetch_actions_with_classes(self, include_siblings: bool = False) -> list[dict]:
        """Fetch annotated actions (member expressions) with parent actieplan context.

        Returns one sample per action expression that has classifying annotations,
        including the action's own text, parent plan context, and assessment votes
        associated with each label.
        """
        concept_scheme_uri = sparql_escape_uri(self.fetch_codelist_uri_for_task())
        ai_graph = sparql_escape_uri(GRAPHS['ai'])
        public_graph = sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public"))

        human_validation_graph = sparql_escape_uri(
            GRAPHS.get(
                "human_validation",
                "http://mu.semte.ch/graphs/public/human-validation",
            )
        )

        sibling_select = ""
        sibling_block = ""
        if include_siblings:
            sibling_select = "(GROUP_CONCAT(DISTINCT ?_sibling_summary; separator=\" | \") AS ?sibling_summaries)"
            sibling_block = """
                OPTIONAL {
                    ?planWork eli:has_member ?siblingWork .
                    ?siblingWork eli:is_realized_by ?siblingExpr .
                    ?siblingExpr a eli:Expression .
                    FILTER(?siblingExpr != ?action)
                    OPTIONAL { ?siblingExpr schema:code ?_sib_code }
                    OPTIONAL { ?siblingExpr eli:title ?_sib_title }
                    BIND(CONCAT(COALESCE(STR(?_sib_code), ""), " ", COALESCE(STR(?_sib_title), "")) AS ?_sibling_summary)
                }
            """

        no_match_uri = "http://mu.semte.ch/vocabularies/ext/no-match-found"

        q = get_prefixes_for_query("rdf", "eli", "oa", "epvoc", "skos", "schema") + f"""
        SELECT ?action ?body ?voteLabel
               ?action_code ?action_title ?action_description ?action_content
               ?plan_title ?plan_description ?plan_code
               {sibling_select}
        WHERE {{
            GRAPH {ai_graph} {{
                ?ann a oa:Annotation ;
                     oa:hasTarget ?action ;
                     oa:motivatedBy oa:classifying ;
                     oa:hasBody ?body .
            }}
            OPTIONAL {{
                GRAPH {human_validation_graph} {{
                    ?review a oa:Annotation ;
                            oa:motivatedBy oa:assessing ;
                            oa:hasTarget ?ann ;
                            oa:hasBody ?voteLabel .
                }}
            }}
            {{
                GRAPH {public_graph} {{
                    ?body a skos:Concept ;
                          skos:inScheme ?scheme .
                }}
                VALUES ?scheme {{ {concept_scheme_uri} }}
            }}
            UNION
            {{
                GRAPH {ai_graph} {{
                    ?ann oa:hasBody <{no_match_uri}> .
                    ?ann oa:hasTarget ?action .
                    FILTER NOT EXISTS {{
                        ?other_ann a oa:Annotation ;
                            oa:hasTarget ?action ;
                            oa:motivatedBy oa:classifying ;
                            oa:hasBody ?real_body .
                        GRAPH {public_graph} {{
                            ?real_body a skos:Concept ;
                                      skos:inScheme {concept_scheme_uri} .
                        }}
                    }}
                }}
            }}

            GRAPH ?g {{
                ?action a eli:Expression .

                # Walk up: action expr ← work ← plan work → plan expr
                ?actionWork eli:is_realized_by ?action .
                ?planWork eli:has_member ?actionWork ;
                          eli:is_realized_by ?planExpr .
                ?planExpr a eli:Expression .

                # Action's own fields
                OPTIONAL {{ ?action schema:code ?action_code }}
                OPTIONAL {{ ?action eli:title ?action_title }}
                OPTIONAL {{ ?action eli:description ?action_description }}
                OPTIONAL {{ ?action epvoc:expressionContent ?action_content }}

                # Parent plan context
                OPTIONAL {{ ?planExpr eli:title ?plan_title }}
                OPTIONAL {{ ?planExpr eli:description ?plan_description }}
                OPTIONAL {{ ?planExpr schema:code ?plan_code }}

                {sibling_block}
            }}
        }}
        GROUP BY ?action ?body ?voteLabel
                 ?action_code ?action_title ?action_description ?action_content
                 ?plan_title ?plan_description ?plan_code
        """

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        samples_by_action: dict[str, dict] = {}
        for binding in bindings:
            action_uri = binding["action"]["value"]
            sample = samples_by_action.setdefault(
                action_uri,
                {
                    "decision": action_uri,
                    "classes": [],
                    "label_votes": {},
                    "text": self._assemble_action_text(binding),
                },
            )

            class_uri = binding.get("body", {}).get("value", "")
            if not class_uri or class_uri == no_match_uri:
                continue

            if class_uri not in sample["classes"]:
                sample["classes"].append(class_uri)

            votes = sample["label_votes"].setdefault(class_uri, [])
            vote_uri = binding.get("voteLabel", {}).get("value", "")
            if vote_uri and vote_uri not in votes:
                votes.append(vote_uri)

        return list(samples_by_action.values())

    def fetch_decisions_with_classes(self) -> list[dict[str, str | list[str]]]:
        expression_filter = self.get_expressions_in_task_filter("?decision")
        member_block = self.member_content_sparql_block("?decision")
        q = Template(get_prefixes_for_query("rdf", "eli", "eli-dl", "oa", "epvoc", "dct", "skos", "schema") + """
        SELECT ?decision ?title ?description ?decision_basis ?content ?classes ?title_code ?work_type
               (GROUP_CONCAT(DISTINCT ?_member_text; separator="\\n\\n") AS ?member_content)
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
                {
                    GRAPH $public_graph {
                        ?body a skos:Concept ;
                              skos:inScheme ?scheme .
                    }
                    VALUES ?scheme {
                        $concept_scheme_uri
                    }
                }
                UNION
                {
                    GRAPH $ai_graph {
                        ?ann oa:hasBody <$no_match_uri> .
                        ?ann oa:hasTarget ?decision .
                        FILTER NOT EXISTS {
                            ?other_ann a oa:Annotation ;
                                oa:hasTarget ?decision ;
                                oa:motivatedBy oa:classifying ;
                                oa:hasBody ?real_body .
                            GRAPH $public_graph {
                                ?real_body a skos:Concept ;
                                          skos:inScheme $concept_scheme_uri .
                            }
                        }
                    }
                }
            }
            GROUP BY ?decision
        }
            GRAPH ?dataGraph {
                $expression_filter
                ?decision rdf:type eli:Expression .
                OPTIONAL { ?decision eli:title ?title }
                OPTIONAL { ?decision eli:description ?description }
                OPTIONAL { ?decision eli-dl:decision_basis ?decision_basis }
                OPTIONAL { ?decision epvoc:expressionContent ?content }
                $member_block
            }
        }
        GROUP BY ?decision ?title ?description ?decision_basis ?content ?classes ?title_code ?work_type
        """).substitute(
            expression_filter=expression_filter,
            ai_graph=sparql_escape_uri(GRAPHS['ai']),
            public_graph=sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public")),
            concept_scheme_uri=sparql_escape_uri(self.fetch_codelist_uri_for_task()),
            member_block=member_block,
            no_match_uri="http://mu.semte.ch/vocabularies/ext/no-match-found",
        )

        res = query(q, sudo=True)
        bindings = res.get("results", {}).get("bindings", [])

        no_match_uri = "http://mu.semte.ch/vocabularies/ext/no-match-found"
        excluded_classes = {no_match_uri}
        return [
            self._build_training_sample(
                b,
                decision_key="decision",
                text=self.assemble_expression_text(b),
                excluded_classes=excluded_classes,
            )
            for b in bindings
        ]