import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def sigmoid(x):
   return 1/(1 + np.exp(-x))


class Metrics(ABC):

    @abstractmethod
    def map_predictions(self, predictions):
        pass

    def compute(self, eval_pred):
        preds, labels = eval_pred
        preds = self.map_predictions(preds)
        accuracy = accuracy_score(labels.astype(int), preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels.astype(int), preds, average='weighted')
        recall = recall_score(labels.astype(int), preds, average='weighted')
        f1 = f1_score(labels.astype(int), preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class SingleLabelMetrics(Metrics):
    def map_predictions(self, predictions):
        return np.argmax(predictions, axis=1).astype(int)


class MultiLabelMetrics(Metrics):
    def map_predictions(self, predictions):
        predictions = sigmoid(predictions)
        return (predictions > 0.5).astype(int)


def get_metric_cls(problem_type: str):
    if problem_type == 'single_label_classification':
        return SingleLabelMetrics
    elif problem_type == 'multi_label_classification':
        return MultiLabelMetrics
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")