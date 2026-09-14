import hashlib
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

import pytz
from escape_helpers import sparql_escape_uri, sparql_escape_string
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS


INSTANCE_BASE = "http://lblod.data.gift/id"
AI_MODEL_BASE = f"{INSTANCE_BASE}/ai-models"
SOURCE_CODE_BASE = f"{INSTANCE_BASE}/source-codes"
INPUT_BASE = f"{INSTANCE_BASE}/inputs"
OUTPUT_BASE = f"{INSTANCE_BASE}/outputs"
QUALITY_MEASUREMENT_BASE = f"{INSTANCE_BASE}/quality-measurements"
METRIC_BASE = f"{INSTANCE_BASE}/metrics"
DIMENSION_BASE = f"{INSTANCE_BASE}/dims"
CONCEPT_SCHEME_BASE = f"{INSTANCE_BASE}/conceptschemes"

METRIC_URIS = {
    "eval_accuracy": f"{METRIC_BASE}/AccuracyMetric",
    "eval_precision": f"{METRIC_BASE}/PrecisionMetric",
    "eval_recall": f"{METRIC_BASE}/RecallMetric",
    "eval_f1": f"{METRIC_BASE}/F1ScoreMetric",
}


def _to_finite_decimal(metric_name: str, metric_value: object) -> Decimal:
    if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float, Decimal)):
        raise ValueError(f"Invalid value for {metric_name}: {metric_value!r}")

    try:
        decimal_value = Decimal(str(metric_value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid value for {metric_name}: {metric_value!r}") from None

    if not decimal_value.is_finite():
        raise ValueError(f"Invalid value for {metric_name}: {metric_value!r}")
    return decimal_value


def _quote_path_segments(value: str) -> str:
    return "/".join(quote(segment, safe="") for segment in value.split("/") if segment)


def _build_derived_concept_scheme_uri(
    concept_scheme_uri: str,
    supported_label_uris: list[str],
) -> str:
    if not isinstance(concept_scheme_uri, str) or not concept_scheme_uri:
        raise ValueError("concept_scheme_uri must be a non-empty URI")
    if not supported_label_uris or not all(
        isinstance(uri, str) and uri for uri in supported_label_uris
    ):
        raise ValueError("supported_label_uris must contain at least one non-empty URI")
    supported_uris = sorted(set(supported_label_uris))

    identity = "\n".join([concept_scheme_uri, *supported_uris]).encode("utf-8")
    subset_digest = hashlib.sha256(identity).hexdigest()
    return f"{CONCEPT_SCHEME_BASE}/{subset_digest}"


def build_airo_model_insert_query(
    hub_model_id: str,
    commit_oid: str,
    hf_repo_url: str,
    results: dict,
    concept_scheme_uri: str,
    supported_label_uris: list[str],
) -> str:
    prefixes = get_prefixes_for_query(
        "dcterms", "dqv", "sd", "airo", "schema", "xsd", "rdf", "rdfs", "skos", "ext")

    model_slug = _quote_path_segments(hub_model_id)
    version_slug = quote(commit_oid, safe="")
    resource_slug = f"{model_slug}/{version_slug}"
    model_uri = f"{AI_MODEL_BASE}/{resource_slug}"
    source_code_uri = f"{SOURCE_CODE_BASE}/{resource_slug}"
    output_uri = f"{OUTPUT_BASE}/{resource_slug}"
    graph_uri = sparql_escape_uri(GRAPHS["ai"])
    public_graph_uri = sparql_escape_uri(GRAPHS["public"])
    derived_concept_scheme_uri = _build_derived_concept_scheme_uri(
        concept_scheme_uri,
        supported_label_uris,
    )
    derived_membership = "\n".join(
        f"    {sparql_escape_uri(concept_uri)} skos:inScheme "
        f"{sparql_escape_uri(derived_concept_scheme_uri)} ."
        for concept_uri in sorted(set(supported_label_uris))
    )

    published_date = datetime.now(tz=pytz.timezone("Europe/Brussels")).date().isoformat()
    published_literal = f'"{published_date}"^^xsd:date'

    qm_uris = []
    qm_nodes_parts = []
    for metric_name, metric_uri in METRIC_URIS.items():
        if metric_name not in results:
            continue
        metric_value = results[metric_name]
        decimal_value = _to_finite_decimal(metric_name, metric_value)

        qm_uri = f"{QUALITY_MEASUREMENT_BASE}/{uuid.uuid4()}"
        qm_uris.append(sparql_escape_uri(qm_uri))
        qm_nodes_parts.append(f"""
  {sparql_escape_uri(qm_uri)} a dqv:QualityMeasurement ;
      dqv:isMeasurementOf {sparql_escape_uri(metric_uri)} ;
      dqv:value \"{format(decimal_value, 'f')}\"^^xsd:decimal .""")

    qm_line = f"dqv:hasQualityMeasurement {', '.join(qm_uris)} ;" if qm_uris else ""
    qm_nodes = "".join(qm_nodes_parts)

    metric_definitions = {
        "eval_accuracy": ("Accuracy", "The aggregate accuracy of the codelist classifier.", "Accuracy"),
        "eval_precision": ("Precision", "The aggregate precision of the codelist classifier.", "Precision"),
        "eval_recall": ("Recall", "The aggregate recall of the codelist classifier.", "Recall"),
        "eval_f1": ("F1 score", "The aggregate F1 score of the codelist classifier.", "F1Score"),
    }
    metric_definition = "\n".join(
        f"""
    {sparql_escape_uri(metric_uri)} a dqv:Metric ;
        rdfs:label \"{label}\" ;
        rdfs:comment \"{comment}\" ;
        dqv:expectedDataType xsd:decimal ;
        dqv:inDimension {sparql_escape_uri(f'{DIMENSION_BASE}/{dimension_name}')} .
"""
        for metric_name, metric_uri in METRIC_URIS.items()
        for label, comment, dimension_name in [metric_definitions[metric_name]]
    )

    return prefixes + f"""
INSERT DATA {{
    GRAPH {graph_uri} {{
    {sparql_escape_uri(model_uri)} a airo:AIModel, sd:SoftwareVersion ;
        ext:classificationLevel {sparql_escape_uri(f'http://lblod.data.gift/id/concept/ai-model-level/small-trained-model')} ;
        schema:datePublished {published_literal} ;
        sd:hasSourceCode {sparql_escape_uri(source_code_uri)} ;
        sd:hasVersionId {sparql_escape_string(commit_oid)} ;
        airo:hasInput {sparql_escape_uri(f'http://lblod.data.gift/id/inputs/text-input')} ;
        airo:producesOutput {sparql_escape_uri(output_uri)} ;
        {qm_line}
        .

    {sparql_escape_uri('http://lblod.data.gift/id/components/codelist-classifier/v1.0.0')} airo:hasModel {sparql_escape_uri(model_uri)} .

    {sparql_escape_uri(output_uri)} a airo:Output, skos:ConceptScheme;
        rdfs:label \"Codelist classifications\" ;
        ext:forConceptScheme {sparql_escape_uri(derived_concept_scheme_uri)} .

    {sparql_escape_uri(source_code_uri)} a sd:SourceCode ;
        schema:codeRepository {sparql_escape_uri(hf_repo_url)} .

{qm_nodes}
    }}
    GRAPH {public_graph_uri} {{
    {sparql_escape_uri(derived_concept_scheme_uri)} a skos:ConceptScheme ;
        rdfs:label \"Codelist classifications\" ;
        ext:forConceptScheme {sparql_escape_uri(concept_scheme_uri)} .

{derived_membership}
    }}
}}
"""
