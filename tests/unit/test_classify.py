"""
Unit tests for ClassifierAnnotatingTask (src/task/classify.py) and predict()
(src/classifier/predict.py).

The Hugging Face tokenizer and model are replaced by stand-ins whose logits are
chosen per decision text.  Everything else runs for real against Virtuoso.
"""

import logging
from types import SimpleNamespace

import helpers
import pytest
import torch
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import GRAPHS
from src.classifier.predict import predict
from src.config import AppConfig, MLInferenceConfig
from src.task.classify import CLASSIFY_COMPONENT, ClassifierAnnotatingTask

from tests.unit.conftest import (
    CONCEPT_SCHEME_URI,
    CONCEPT_URI,
    CONCEPT_URI_2,
    EXPRESSION_CONTENT,
    EXPRESSION_CONTENT_2,
    EXPRESSION_URI,
    EXPRESSION_URI_2,
    JOB_URI,
    NS,
    TASK_URI,
    sparql_ask,
    sparql_update,
)

LABEL_1 = "Affordable and Clean Energy"
LABEL_2 = "Climate Action"
ID2LABEL = {0: LABEL_1, 1: LABEL_2}


MODEL_ID = "test-org/test-classifier"


CLASSIFIER_AGENT = f"http://test.example.org/agents/{CLASSIFY_COMPONENT}"


SINGLE = "single_label_classification"


MULTI = "multi_label_classification"

TEXT_1 = EXPRESSION_CONTENT
TEXT_2 = EXPRESSION_CONTENT_2

# softmax([4, 0])[0] is 0.982; softmax([0.4, 0])[0] is 0.599


CONFIDENT = [4.0, 0.0]


HESITANT = [0.4, 0.0]


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"text": text}


class FakeModel:
    """Returns logits looked up by the decision text the tokenizer passed on."""

    def __init__(self, logits_by_text, problem_type, id2label):
        self._logits_by_text = logits_by_text
        self.config = SimpleNamespace(problem_type=problem_type, id2label=id2label)

    def eval(self):
        return self

    def __call__(self, text):
        return SimpleNamespace(logits=torch.tensor([self._logits_by_text[text]]))


@pytest.fixture
def classify_task() -> ClassifierAnnotatingTask:
    task = object.__new__(ClassifierAnnotatingTask)
    task.task_uri = TASK_URI
    task.logger = logging.getLogger("test.classify_task")
    task.source = None
    task.source_graph = None
    task.results_container_uris = []
    return task


@pytest.fixture
def codelist_concepts():
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["public"])} {{
                {sparql_escape_uri(CONCEPT_URI)}
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "{LABEL_1}" .
                {sparql_escape_uri(CONCEPT_URI_2)}
                    {sparql_escape_uri(NS["skos"] + "inScheme")} {sparql_escape_uri(CONCEPT_SCHEME_URI)} ;
                    {sparql_escape_uri(NS["skos"] + "prefLabel")} "{LABEL_2}" .
            }}
        }}
    """)
    yield


@pytest.fixture
def job_targeting_expressions(task_in_job_with_codelist):
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                {sparql_escape_uri(JOB_URI)}
                    {sparql_escape_uri(NS["ext"] + "graphForTargets")} {sparql_escape_uri(GRAPHS["expressions"])} .
            }}
        }}
    """)
    yield


def _set_job_threshold(value: float) -> None:
    sparql_update(f"""
        INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                {sparql_escape_uri(JOB_URI)}
                    {sparql_escape_uri(NS["ext"] + "confidenceThreshold")} "{value}" .
            }}
        }}
    """)


def _classify(
    task,
    mocker,
    logits_by_text,
    *,
    problem_type=SINGLE,
    config_threshold=0.5,
    model_id=MODEL_ID,
):
    config = AppConfig(
        ml_inference=MLInferenceConfig(
            huggingface_model_id=model_id,
            confidence_threshold=config_threshold,
        )
    )
    mocker.patch("src.task.classify.get_config", return_value=config)
    mocker.patch(
        "src.task.classify.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()
    )
    mocker.patch(
        "src.task.classify.AutoModelForSequenceClassification.from_pretrained",
        return_value=FakeModel(logits_by_text, problem_type, ID2LABEL),
    )
    task.process()


def _linking_annotations() -> set[tuple[str, str, str]]:
    result = helpers.query(f"""
        SELECT ?target ?body ?agent WHERE {{
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                ?ann a {sparql_escape_uri(NS["oa"] + "Annotation")} ;
                    {sparql_escape_uri(NS["oa"] + "motivatedBy")} {sparql_escape_uri(NS["oa"] + "classifying")} ;
                    {sparql_escape_uri(NS["oa"] + "hasTarget")} ?target ;
                    {sparql_escape_uri(NS["oa"] + "hasBody")} ?body .
                ?activity {sparql_escape_uri(NS["prov"] + "generated")} ?ann ;
                    {sparql_escape_uri(NS["prov"] + "wasAssociatedWith")} ?agent .
            }}
        }}
    """)
    return {
        (b["target"]["value"], b["body"]["value"], b["agent"]["value"])
        for b in result["results"]["bindings"]
    }


def _container_holds(resource_uri: str) -> bool:
    return sparql_ask(f"""
        ASK {{
            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container {sparql_escape_uri(NS["task"] + "hasResource")} {sparql_escape_uri(resource_uri)} .
            }}
        }}
    """)


@pytest.mark.usefixtures("codelist_concepts", "job_targeting_expressions", "unannotated_expressions")
class TestProcessThresholds:

    def test_job_threshold_overrides_stricter_config_threshold(self, classify_task, mocker):
        _set_job_threshold(0.1)

        _classify(
            classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: HESITANT}, config_threshold=0.99
        )

        assert _linking_annotations() == {
            (EXPRESSION_URI, CONCEPT_URI, CLASSIFIER_AGENT),
            (EXPRESSION_URI_2, CONCEPT_URI, CLASSIFIER_AGENT),
        }

    def test_job_threshold_overrides_looser_config_threshold(self, classify_task, mocker):
        _set_job_threshold(0.9)

        _classify(
            classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: HESITANT}, config_threshold=0.1
        )

        assert _linking_annotations() == {(EXPRESSION_URI, CONCEPT_URI, CLASSIFIER_AGENT)}

    def test_config_threshold_applies_without_job_threshold(self, classify_task, mocker):
        _classify(
            classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: HESITANT}, config_threshold=0.9
        )

        assert _linking_annotations() == {(EXPRESSION_URI, CONCEPT_URI, CLASSIFIER_AGENT)}


@pytest.mark.usefixtures("codelist_concepts", "job_targeting_expressions", "unannotated_expressions")
class TestProcessAnnotations:

    def test_annotation_links_decision_to_the_concept_of_the_predicted_label(
        self, classify_task, mocker
    ):
        _classify(classify_task, mocker, {TEXT_1: [0.0, 4.0], TEXT_2: CONFIDENT})

        assert _linking_annotations() == {
            (EXPRESSION_URI, CONCEPT_URI_2, CLASSIFIER_AGENT),
            (EXPRESSION_URI_2, CONCEPT_URI, CLASSIFIER_AGENT),
        }

    def test_multi_label_writes_one_annotation_per_label_above_threshold(
        self, classify_task, mocker
    ):
        _classify(
            classify_task,
            mocker,
            {TEXT_1: [4.0, 4.0], TEXT_2: [4.0, -4.0]},
            problem_type=MULTI,
        )

        assert _linking_annotations() == {
            (EXPRESSION_URI, CONCEPT_URI, CLASSIFIER_AGENT),
            (EXPRESSION_URI, CONCEPT_URI_2, CLASSIFIER_AGENT),
            (EXPRESSION_URI_2, CONCEPT_URI, CLASSIFIER_AGENT),
        }

    def test_classified_decision_and_its_annotation_go_into_output_containers(
        self, classify_task, mocker
    ):
        _set_job_threshold(0.9)

        _classify(classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: HESITANT})

        assert len(classify_task.results_container_uris) == 2
        assert _container_holds(EXPRESSION_URI)
        assert not _container_holds(EXPRESSION_URI_2)

    def test_rejects_a_model_with_an_unsupported_problem_type(self, classify_task, mocker):
        with pytest.raises(RuntimeError, match="Unexpected problem_type='regression'"):
            _classify(
                classify_task,
                mocker,
                {TEXT_1: CONFIDENT, TEXT_2: CONFIDENT},
                problem_type="regression",
            )

        assert _linking_annotations() == set()


@pytest.mark.usefixtures("codelist_concepts", "job_targeting_expressions")
class TestProcessSkipsAnnotated:

    @pytest.mark.usefixtures("one_annotated_one_plain_expression")
    def test_decision_with_a_classifying_annotation_for_the_codelist_is_not_classified_again(
        self, classify_task, mocker
    ):
        _classify(classify_task, mocker, {TEXT_1: [0.0, 4.0], TEXT_2: [0.0, 4.0]})

        targets_and_bodies = {(t, b) for t, b, _ in _linking_annotations()}
        assert targets_and_bodies == {
            (EXPRESSION_URI, CONCEPT_URI_2),
            (EXPRESSION_URI_2, CONCEPT_URI),
        }

    @pytest.mark.usefixtures("two_annotated_expressions")
    def test_fails_when_every_decision_is_already_annotated(self, classify_task, mocker):
        before = _linking_annotations()

        with pytest.raises(RuntimeError, match="nothing to classify"):
            _classify(classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: CONFIDENT})

        assert _linking_annotations() == before
        assert classify_task.results_container_uris == []


class TestProcessFailures:

    @pytest.mark.usefixtures("codelist_concepts", "job_targeting_expressions", "unannotated_expressions")
    def test_fails_when_no_model_id_is_configured(self, classify_task, mocker):
        with pytest.raises(RuntimeError, match="huggingface_model_id is not set"):
            _classify(
                classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: CONFIDENT}, model_id=None
            )

        assert _linking_annotations() == set()

    @pytest.mark.usefixtures("codelist_concepts", "task_in_job_with_codelist", "unannotated_expressions")
    def test_fails_when_the_job_has_no_graph_for_targets(self, classify_task, mocker):
        with pytest.raises(RuntimeError, match="No ext:graphForTargets found for task"):
            _classify(classify_task, mocker, {TEXT_1: CONFIDENT, TEXT_2: CONFIDENT})

        assert _linking_annotations() == set()

    @pytest.mark.usefixtures("codelist_concepts", "job_targeting_expressions")
    def test_fails_when_the_target_graph_holds_no_decisions(self, classify_task, mocker):
        with pytest.raises(RuntimeError, match="nothing to classify"):
            _classify(classify_task, mocker, {})

        assert _linking_annotations() == set()


LABELS = {0: "a", 1: "b", 2: "c"}


def _predict(logits, problem_type, threshold=0.5):
    model = FakeModel({"t": logits}, problem_type, LABELS)
    return predict("t", model, FakeTokenizer(), LABELS, problem_type, threshold)


class TestPredict:

    def test_single_label_returns_only_the_argmax_label_with_its_softmax_confidence(self):
        result = _predict([0.0, 3.0, 1.0], SINGLE)

        expected = torch.softmax(torch.tensor([0.0, 3.0, 1.0]), dim=-1)[1].item()
        assert [label for label, _ in result] == ["b"]
        assert result[0][1] == pytest.approx(expected)

    def test_single_label_returns_nothing_when_top_label_is_below_threshold(self):
        assert _predict([0.0, 0.5, 0.4], SINGLE) == []

    def test_multi_label_returns_every_label_whose_sigmoid_clears_threshold(self):
        result = _predict([3.0, -3.0, 2.0], MULTI)

        assert [label for label, _ in result] == ["a", "c"]
        assert result[0][1] == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
        assert result[1][1] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())

    def test_unknown_problem_type_raises(self):
        with pytest.raises(ValueError, match="Unknown problem_type: 'regression'"):
            _predict([1.0, 2.0, 3.0], "regression")
