import os,sys

from script.entity.artifact_enitity import ClassificationMetricArtifact
from script.exception import MoneyLaunderingException
from sklearn.metrics import f1_score,precision_score,recall_score


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Method Name: get_classification_score
    Description: Calculate and return classification metrics.
    
    Input:
    - y_true: True labels
    - y_pred: Predicted labels
    
    Output: ClassificationMetricArtifact containing f1_score, precision_score, and recall_score
    
    On Failure: Writes an exception log and raises an exception.
    
    Version: 1.0
    """
    try:
        model_f1_score = f1_score(y_true, y_pred, average="weighted")
        model_recall_score = recall_score(y_true, y_pred, average="weighted")
        model_precision_score = precision_score(y_true, y_pred, average="weighted")

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise MoneyLaunderingException(e, sys)
