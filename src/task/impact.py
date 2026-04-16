from helpers import query, update
from escape_helpers import sparql_escape_uri, sparql_escape_string
from string import Template

from decide_ai_service_base.sparql_config import TASK_OPERATIONS, GRAPHS, get_prefixes_for_query
from .codelist import CodeListTask
from ..llm_models.llm_model_clients import create_llm_model
from ..config import get_config
from langchain_core.messages import HumanMessage, SystemMessage


from pydantic import BaseModel, Field
from enum import Enum


class ImpactDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactAssessment(BaseModel):
    label: str = Field(description="The classification label being assessed")
    impact_direction: ImpactDirection = Field(description="Overall direction of the policy impact")
    confidence: ConfidenceLevel = Field(description="Confidence level in this assessment")
    reasoning: str = Field(description="Step-by-step reasoning behind the assessment")
    direct_effects: list[str] = Field(description="Direct effects of the policy on the label domain")
    second_order_effects: list[str] = Field(description="Indirect or downstream effects")
    key_uncertainties: list[str] = Field(description="Main factors that could change the assessment")
    summary: str = Field(description="One-sentence summary of the impact")


class ProcessItem(BaseModel):
    expression_uri: str
    expression_content: str
    language: str
    work_uri: str


class PolicyLabel(BaseModel):
    annotation_uri: str
    policy_concept_uri: str
    policy_label: str


class ImpactAssessmentTask(CodeListTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["impact_assessment"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)
        config = get_config()
        self.llm = create_llm_model(config.llm).with_structured_output(ImpactAssessment)
        self.provider = config.llm.provider

    def fetch_eli_expressions(self) -> list[ProcessItem]:
        """
        Retrieve ELI expressions, their epvoc:expressionContent,
        language, and corresponding ELI work URI from the task's input container.

        Returns:
            Dictionary containing:
                - "expression_uris": list containing the expression URIs
                - "expression_contents": list containing the expression contents
                - "languages": list containing the language URIs of the expressions
                - "work_uris": list containing the work URIs of the expressions
        """
        q = Template(
            get_prefixes_for_query("task", "epvoc", "eli") +
            f"""
            SELECT ?expression ?content ?lang ?work WHERE {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                $task task:inputContainer ?container .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container task:hasResource ?expression .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                ?expression a eli:Expression ;
                            epvoc:expressionContent ?content ;
                            eli:language ?lang .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["works"])} {{
                ?work a eli:Work ;
                    eli:is_realized_by ?expression .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            self.logger.warning(
                f"No expressions found in input container for task {self.task_uri}")
            return []

        return [
            ProcessItem(
                expression_uri=t[0],
                expression_content=t[1],
                language=t[2],
                work_uri=t[3]
            )
            for t
            in zip(
                [b["expression"]["value"] for b in bindings],
                [b["content"]["value"] for b in bindings],
                [b["lang"]["value"] for b in bindings],
                [b["work"]["value"] for b in bindings]
            )
        ]

    def fetch_policy_labels(self, expression_uri: str) -> list[PolicyLabel]:
        concept_scheme_uri = self.fetch_codelist_uri_for_task()
        q = Template(
            get_prefixes_for_query("oa", "skos") +
            """
            SELECT ?annotation ?concept ?label
            WHERE {
              GRAPH $annotation_graph {
                ?annotation a oa:Annotation ;
                            oa:motivatedBy oa:classifying ;
                            oa:hasTarget $expression_uri ;
                            oa:hasBody ?concept .
              }
              GRAPH $concept_graph {
                ?concept a skos:Concept ;
                         skos:inScheme $concept_scheme_uri ;
                         skos:prefLabel ?label .
              }
            }
            """
        ).substitute(
            annotation_graph=sparql_escape_uri(GRAPHS['ai']),
            concept_graph=sparql_escape_uri(GRAPHS.get("public", "http://mu.semte.ch/graphs/public")),
            expression_uri=sparql_escape_uri(expression_uri),
            concept_scheme_uri=sparql_escape_uri(concept_scheme_uri)
        )
        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            self.logger.warning(
                f"No expressions found in input container for task {self.task_uri}")
            return []

        return [
            PolicyLabel(
                annotation_uri=t[0],
                policy_concept_uri=t[1],
                policy_label=t[2]
            )
            for t
            in zip(
                [b["annotation"]["value"] for b in bindings],
                [b["concept"]["value"] for b in bindings],
                [b["label"]["value"] for b in bindings]
            )
        ]

    def _process_single(self, process_item: ProcessItem, policy_label: PolicyLabel) -> ImpactAssessment:
        SYSTEM_PROMPT = """
        You are a policy impact analyst specializing in sustainable development and governance.

        You will be given:
        1. A **policy text** — a description of a decision, regulation, or initiative
        2. A **label** — a classification (e.g. an SDG goal, a thematic domain) that has already been assigned to this policy

        Your task is to assess whether the impact of this policy on the given label's domain is **positive**, **negative**, or **uncertain**.

        Follow this reasoning process:
        1. Identify the core intent and mechanisms of the policy
        2. Consider the specific scope and targets of the given label
        3. Assess direct effects first, then second-order effects
        4. Weigh both short-term and long-term consequences
        5. Conclude with an overall impact direction and confidence level (**low**, **medium** or **high**)

        Be precise, grounded, and concise. Avoid generic praise or criticism.
        """

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Policy text: {process_item.expression_content}\nLabel: {policy_label.policy_label}\n\nProvide a structured impact assessment."),
        ]

        return self.llm.invoke(messages)


    def store(self, annotation_uri: str, assessment: ImpactAssessment):
        query_string = Template(get_prefixes_for_query("oa", "ext", "xsd") +
        """
        INSERT {
            GRAPH $graph {
                $annotation_uri ext:has_impact $assessment .
            }
        }
        WHERE {
            GRAPH $graph {
                $annotation_uri a oa:Annotation .
                FILTER NOT EXISTS { $annotation_uri ext:has_impact ?anyImpact . }
            }
        }
        """
        ).substitute(
            graph=sparql_escape_uri(GRAPHS['ai']),
            annotation_uri=sparql_escape_uri(annotation_uri),
            assessment=sparql_escape_string(assessment.impact_direction.value)
        )
        try:
            update(query_string, sudo=True)
        except Exception as e:
            error_msg = f"Failed to insert impact assesment to triplestore for annotation {annotation_uri}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


    def process(self):
        for process_item in self.fetch_eli_expressions():
            for policy_label in self.fetch_policy_labels(process_item.expression_uri):
                assessment = self._process_single(process_item, policy_label)
                self.store(policy_label.annotation_uri, assessment)
