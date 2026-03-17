import uuid
import pytz
from string import Template
from datetime import datetime
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_float, sparql_escape_datetime
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS


def build_airo_model_insert_query(
    hub_model_id: str,
    commit_oid: str,
    code_git_sha: str,
    hf_repo_url: str,
    hf_tree_url: str,
    source_repo_url: str,
    results: dict,
    base: str = "http://example.com",
) -> str:
    prefixes = get_prefixes_for_query(
        "dcterms", "ns1", "ns2", "ns3", "schema", "xsd", "rdf")

    model_uri = f"{base}/model/{hub_model_id}"
    version_uri = f"{base}/version/{commit_oid}"
    code_uri = f"{base}/code/{commit_oid}"
    input_uri = f"{base}/modelinput/text"
    modelfiles_uri = f"{base}/modelfiles/{hub_model_id}"
    source_code_uri = f"{base}/sourcecode/lblod-text-classifier"

    published_literal = sparql_escape_datetime(
        datetime.now(tz=pytz.timezone("Europe/Brussels"))
    )

    qm_uris = []
    qm_nodes_parts = []
    for metric_name, metric_value in results.items():
        qm_uri = f"{base}/qualitymeasurement/{uuid.uuid4()}"
        metric_uri = f"{base}/metric/{metric_name}"
        qm_uris.append(sparql_escape_uri(qm_uri))
        qm_nodes_parts.append(f"""
  {sparql_escape_uri(qm_uri)} ns1:isMeasurementOf {sparql_escape_uri(metric_uri)} ;
      ns1:value {sparql_escape_float(metric_value)} .""")

    qm_list = ",\n        ".join(qm_uris) if qm_uris else ""
    qm_nodes = "".join(qm_nodes_parts)

    tpl = Template(prefixes + """
INSERT DATA {
  GRAPH ${graph_uri} {  
    ${model_uri} a ns3:AIModel ;
        dcterms:source ${hf_tree_anyuri} ;
        dcterms:title ${hub_model_id_str} ;
        ${qm_line}
        ns3:hasInput ${input_uri} ;
        ns3:hasVersion ${version_uri} ;
        ns2:dataPublished ${published_literal} ;
        ns2:hasVersion ${code_uri} .

    ${code_uri} a ns2:SoftwareVersion ;
        ns2:hasSourceCode ${source_code_uri} ;
        ns2:hasVersionId ${code_git_sha_str} .

    ${modelfiles_uri} a ns2:SourceCode ;
        schema:codeRepository ${hf_repo_anyuri} .

    ${input_uri} dcterms:type ${input_type_str} .

    ${source_code_uri} a ns2:SourceCode ;
        schema:codeRepository ${source_repo_anyuri} .

    ${version_uri} a ns3:Version ;
        ns2:hasSourceCode ${modelfiles_uri} ;
        ns2:hasVersionId ${commit_oid_str} .

${qm_nodes}
  }
}
""")

    query = tpl.substitute(
        graph_uri=sparql_escape_uri(GRAPHS["ai"]),
        model_uri=sparql_escape_uri(model_uri),
        hf_tree_anyuri=sparql_escape_uri(hf_tree_url),
        hub_model_id_str=sparql_escape_string(hub_model_id),
        qm_line=(f"ns1:QualityMeasurement {qm_list} ;" if qm_list else ""),
        input_uri=sparql_escape_uri(input_uri),
        version_uri=sparql_escape_uri(version_uri),
        published_literal=published_literal,
        code_uri=sparql_escape_uri(code_uri),
        source_code_uri=sparql_escape_uri(source_code_uri),
        code_git_sha_str=sparql_escape_string(code_git_sha),
        modelfiles_uri=sparql_escape_uri(modelfiles_uri),
        hf_repo_anyuri=sparql_escape_uri(hf_repo_url),
        input_type_str=sparql_escape_string("string"),
        source_repo_anyuri=sparql_escape_uri(source_repo_url),
        commit_oid_str=sparql_escape_string(commit_oid),
        qm_nodes=qm_nodes
    )

    return query
