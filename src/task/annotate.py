import os
import random
import time
from string import Template
from helpers import query, logger
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import TASK_OPERATIONS, AGENT_TYPES, get_prefixes_for_query, GRAPHS
from decide_ai_service_base.annotation import LinkingAnnotation
from decide_ai_service_base.util import get_agent_uri

from ..llm_models.llm_model_clients import create_llm_client
from ..llm_models.llm_task_models import LlmTaskInput
from .codelist import Codelist, CodeListTask
from ..config import get_config


class ModelAnnotatingTask(CodeListTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["model_annotation"]

    def __init__(self, task_uri: str, source: str = None,
                 codelist_entries: 'Codelist | None' = None,
                 property_path_for_text: str | None = None,
                 annotate_actionplan: bool = False):
        
        super().__init__(task_uri)
        self.source = source
        self.annotate_actionplan = annotate_actionplan

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

    def process(self):
        # Fetch the text to be annotated, either from a specific property path or from the default data source
        if self._property_path_for_text:
            task_data = self.fetch_text_with_property_path(self._property_path_for_text)
        else:
            task_data = self.fetch_data()

        # Check if task data is empty or only whitespace
        if not task_data.strip():
            raise RuntimeError(f"No task data found for decision {self.source}; cannot annotate.")

        labels = self._codelist_entries.get_labels()
        if not labels:
            raise RuntimeError(f"No concepts found in codelist for decision {self.source}; cannot annotate.")

        # Prepare labels for the prompt
        labels_for_prompt = self._codelist_entries.get_labels_with_definitions()

        # Fetch code → expression URI mapping (single action or actieplan members)
        member_mapping = self.fetch_member_expression_mapping(self.source)
        action_codes = list(member_mapping.keys())

        classifications: dict[str, list[str]] = {}
        if self._provider == "random":
            logger.warning("Using random label (provider=random).")
            classifications = {code: [random.choice(labels)] for code in action_codes}
        elif self._llm is None:
            raise RuntimeError("No LLM client available; cannot annotate.")
        else:
            max_retries = 3
            user_message = self._llm_user_message.format(
                code_list=labels_for_prompt,
                decision_text=task_data,
                action_codes=action_codes,
            )
            llm_input = LlmTaskInput(
                system_message=self._llm_system_message,
                user_message=user_message,
                assistant_message=None,
                output_format=dict[str, list[str]],
            )
            description_count = sum(
                bool(entry.definition) for entry in self._codelist_entries
            )
            logger.info(
                "Calling LLM for decision %s "
                "(text_chars=%d, prompt_chars=%d, codes=%d, descriptions=%d, actions=%d)",
                self.source,
                len(task_data),
                len(user_message),
                len(labels),
                description_count,
                len(action_codes),
            )

            for attempt in range(1, max_retries + 1):
                try:
                    raw_response = self._llm(llm_input)
                    classifications = self._normalize_llm_response(
                        raw_response, action_codes
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"LLM call failed after {max_retries} attempts "
                            f"({exc}); skipping annotation."
                        ) from exc
                    logger.warning(
                        "LLM call for decision %s failed on attempt %d/%d "
                        "(%s); retrying.",
                        self.source,
                        attempt,
                        max_retries,
                        exc,
                    )
                    time.sleep(attempt)

        logger.info(
            "[(check: %s) num_classifications=%d vs num_action_codes=%d ] Response=%s",
            len(classifications) == len(action_codes),
            len(classifications),
            len(action_codes),
            classifications,
        )



        any_annotations = False
        all_detected_concepts = set()

        # process all underlying action codes and their corresponding class labels

        for action_code, class_labels in classifications.items():
            expression_uri = self._resolve_action_code(action_code, member_mapping)
            if not expression_uri:
                logger.warning(
                    "No expression URI found for action code %r on decision %s; "
                    "skipping.",
                    action_code,
                    self.source,
                )
                continue

            for c in class_labels:
                concept_uri = self._codelist_entries.resolve_label_to_uri(c, self._label_to_uri)
                if not concept_uri:
                    logger.warning(
                        "No concept URI found for returned code %r (action %r) on decision %s; "
                        "skipping annotation.",
                        c,
                        action_code,
                        self.source,
                    )
                    continue

                annotation = LinkingAnnotation(
                    self.task_uri,
                    expression_uri,
                    concept_uri,
                    get_agent_uri("model_annotator"),
                    AGENT_TYPES["ai_component"]
                )
                annotation.add_to_triplestore_if_not_exists()
                any_annotations = True
                all_detected_concepts.add(concept_uri)

                logger.info(
                    "Stored model annotation for decision %s, action %s: code=%s, concept=%s",
                    self.source,
                    action_code,
                    c,
                    concept_uri,
                )

            if not class_labels:
                self._store_no_match_for_expression(expression_uri)

        # Annotate the actieplan (source) itself with all detected concepts
        # from the underlying action codes, if enabled and source is not a single action.
        is_actionplan = self.source and self.source not in member_mapping.values()
        if self.annotate_actionplan and is_actionplan and all_detected_concepts:
            for concept_uri in all_detected_concepts:
                annotation = LinkingAnnotation(
                    self.task_uri,
                    self.source,
                    concept_uri,
                    get_agent_uri("model_annotator"),
                    AGENT_TYPES["ai_component"]
                )
                annotation.add_to_triplestore_if_not_exists()
                logger.info(
                    "Stored actionplan annotation for %s: concept=%s",
                    self.source,
                    concept_uri,
                )

        if any_annotations:
            self.results_container_uris.append(self.create_output_container(self.source))
        elif not is_actionplan:
            self.store_no_match()

        rate_limit_delay = float(os.environ.get("RATE_LIMIT_DELAY_SECONDS", "0"))
        if rate_limit_delay > 0:
            logger.info(
                "Rate-limit delay for decision %s: %.1f seconds",
                self.source,
                rate_limit_delay,
            )
            time.sleep(rate_limit_delay)

    @staticmethod
    def _normalize_llm_response(
        raw_response: dict[str, list[str]] | list[str],
        action_codes: list[str],
    ) -> dict[str, list[str]]:
        """Normalize LLM response into a code → labels dict.

        Handles:
        - Correct dict[str, list[str]] responses
        - Flat list[str] fallback (assigns all labels to first action code)
        - Whitespace/case normalization on keys
        - Keys that don't match action_codes (attempts fuzzy match)
        """
        # Handle flat list fallback (LLM ignored structure instruction)
        if isinstance(raw_response, list):
            logger.warning(
                "LLM returned flat list instead of dict; assigning all labels "
                "to first action code %r.",
                action_codes[0] if action_codes else "?",
            )
            if action_codes:
                return {action_codes[0]: raw_response}
            return {}

        if not isinstance(raw_response, dict):
            logger.warning("LLM returned unexpected type %s; treating as empty.", type(raw_response))
            return {}

        # Build case-insensitive lookup for action codes
        code_lookup = {code.strip().lower(): code for code in action_codes}

        normalized: dict[str, list[str]] = {}
        for key, labels in raw_response.items():
            normalized_key = key.strip()
            # Exact match first
            if normalized_key in action_codes:
                resolved_key = normalized_key
            else:
                # Case-insensitive match
                resolved_key = code_lookup.get(normalized_key.lower())

            if not resolved_key:
                logger.warning(
                    "LLM returned unknown action code %r; attempting prefix match.",
                    key,
                )
                # Try prefix match
                lower_key = normalized_key.lower()
                for ac in action_codes:
                    if ac.lower().startswith(lower_key) or lower_key.startswith(ac.lower()):
                        resolved_key = ac
                        break

            if resolved_key:
                # Ensure labels is a list of strings
                if isinstance(labels, str):
                    labels = [labels]
                elif not isinstance(labels, list):
                    labels = []
                normalized[resolved_key] = [str(l).strip() for l in labels if l]
            else:
                logger.warning(
                    "Could not resolve LLM action code %r to any known action; skipping.",
                    key,
                )

        return normalized

    @staticmethod
    def _resolve_action_code(
        action_code: str, member_mapping: dict[str, str]
    ) -> str | None:
        """Resolve an action code to its expression URI with fallback matching."""
        # Exact match
        if action_code in member_mapping:
            return member_mapping[action_code]

        # Case-insensitive
        lower_code = action_code.strip().lower()
        for code, uri in member_mapping.items():
            if code.lower() == lower_code:
                return uri

        return None


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
        # logger.info(
        #     "Model annotation batch %s contains %d decisions",
        #     self.task_uri,
        #     len(decision_uris),
        # )

        for i, decision_uri in enumerate(decision_uris):

            task = ModelAnnotatingTask(
                self.task_uri,
                source=decision_uri,
                codelist_entries=codelist_entries,
                property_path_for_text=property_path_for_text,
            )
            task.process()
            self.results_container_uris.extend(task.results_container_uris)

            logger.info(
                "Processed model annotation %d/%d: %s",
                i + 1,
                len(decision_uris),
                decision_uri,
            )

    def fetch_decisions_without_annotations(
        self,
        concept_scheme_uri: str,
        target_graph: str | None = None,
        target_nodes: list[str] | None = None,
        target_classes: list[str] | None = None,
        annotate_actions: bool = False,
    ) -> list[str]:
        """Fetch expression URIs that still need annotation for the given concept scheme.

        Two modes controlled by ``annotate_actions``:

        - ``annotate_actions=False`` (default, actieplannen mode):
          Returns actieplan expressions (``eli:work_type vmm:Actieplan``) where
          at least one underlying actie member has no classifying annotation yet
          (i.e. annotation is incomplete).

        - ``annotate_actions=True`` (acties mode):
          Returns standalone expression URIs that have no classifying annotation
          and are NOT members of any actieplan (to prevent double-processing).

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

        # Build the FILTER NOT EXISTS graphs to check for annotations
        if target_graph:
            filter_graph_values = f"VALUES ?g {{ {sparql_escape_uri(target_graph)} {sparql_escape_uri(GRAPHS['ai'])} }}"
        else:
            filter_graph_values = f"VALUES ?g {{ {sparql_escape_uri(GRAPHS['ai'])} }}"

        # Annotation check subpattern (reused in both modes)
        annotation_check = f"""
                {filter_graph_values}
                GRAPH ?g {{
                    ?ann a oa:Annotation ;
                         oa:hasTarget ${{target_var}} ;
                         oa:motivatedBy oa:classifying ;
                         oa:hasBody ?concept .
                    ?concept skos:inScheme|ext:forConceptScheme {sparql_escape_uri(concept_scheme_uri)} .
                }}
        """
        annotation_check_s = annotation_check.replace("${target_var}", "?s")
        annotation_check_member = annotation_check.replace("${target_var}", "?checkExpr")

        expression_filter = self.get_expressions_in_task_filter()

        vmm_prefix = "PREFIX vmm: <http://lblod.data.gift/vocabularies/vmm/>"

        if annotate_actions:
            # Acties mode: standalone expressions without annotations,
            # excluding members of actieplannen (to prevent double-processing)
            q = Template(get_prefixes_for_query("rdf", "eli", "oa", "skos", "ext") + f"""
            {vmm_prefix}
            SELECT DISTINCT ?s
            WHERE {{
                $expression_filter
                $target_clause
                FILTER NOT EXISTS {{
                    {annotation_check_s}
                }}
                FILTER NOT EXISTS {{
                    ?parentWork eli:has_member ?childWork .
                    ?childWork eli:is_realized_by ?s .
                }}
            }}
            """).substitute(
                expression_filter=expression_filter,
                target_clause=target_clause,
            )
        else:
            # Actieplannen mode: actieplan expressions where at least one
            # member actie still has no annotation
            q = Template(get_prefixes_for_query("rdf", "eli", "oa", "skos", "ext") + f"""
            {vmm_prefix}
            SELECT DISTINCT ?s
            WHERE {{
                $expression_filter
                $target_clause
                ?work eli:is_realized_by ?s ;
                      eli:work_type vmm:Actieplan .
                FILTER EXISTS {{
                    ?work eli:has_member ?checkWork .
                    ?checkWork eli:is_realized_by ?checkExpr .
                    ?checkExpr a eli:Expression .
                    FILTER NOT EXISTS {{
                        {annotation_check_member}
                    }}
                }}
            }}
            """).substitute(
                expression_filter=expression_filter,
                target_clause=target_clause,
            )

        response = query(q, sudo=True)
        bindings = response.get("results", {}).get("bindings", [])
        return [b["s"]["value"] for b in bindings if "s" in b]
