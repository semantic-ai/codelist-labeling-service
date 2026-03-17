from datasets import Dataset
import numpy as np
from abc import ABC, abstractmethod


class LabeledData(ABC):
    """Base class for labeled data handling.

    Args:
        decisions (list[dict[str, str | list[str]]]): List of decision dictionaries containing
            the decision URI (key: decision), the classes (key: classes) and text (key: text).
        labels (list[str]): List of strings representing the unique labels.

    Attributes:
        data (list[dict[str, str | list[str]]]): List of decision dictionaries containing
            the decision URI (key: decision), the classes (key: classes) and text (key: text).
        labels (list[str]): List of strings representing the unique labels.
        label2id (dict): Mapping from label names to their corresponding IDs.
        id2label (dict): Mapping from label IDs to their corresponding names.

    """

    def __init__(self, decisions: list[dict[str, str | list[str]]], labels: list[str]):
        super().__init__()
        self.data = decisions
        self.labels = labels
        self.label2id, self.id2label = self.generate_label_map()

    def generate_label_map(self):
        id2label = {idx: label for idx, label in enumerate(self.labels)}
        label2id = {label: idx for idx, label in enumerate(self.labels)}
        return label2id, id2label

    @abstractmethod
    def format(self) -> Dataset:
        """Format the data into a Hugging Face Dataset."""
        pass


class SingleLabelData(LabeledData):
    def format(self) -> Dataset:
        """Format the data into a Hugging Face Dataset for single-label classification."""
        ds = Dataset.from_list([
            {
                "text": decision["text"],
                "label": self.label2id[decision["classes"][0]]
            }
            for decision in self.data
        ]).class_encode_column("label")
        try:
            return ds.train_test_split(test_size=0.1, stratify_by_column="label")
        except ValueError:
            # Fallback when classes are too small to stratify
            return ds.train_test_split(test_size=0.1, shuffle=True)


class MultiLabelData(LabeledData):
    def format(self) -> Dataset:
        """ Format the data into a Hugging Face Dataset for multi-label classification. """
        return Dataset.from_list([
            {
                "text": decision["text"],
                "label": np.isin(
                    np.arange(len(self.label2id)),
                    [self.label2id[l] for l in decision["classes"]]
                ).astype(float).tolist()
            }
            for decision in self.data
        ]).train_test_split(test_size=0.1)


def get_dataset_cls(problem_type: str):
    if problem_type == 'single_label_classification':
        return SingleLabelData
    elif problem_type == 'multi_label_classification':
        return MultiLabelData
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")