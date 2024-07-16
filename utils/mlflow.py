import mlflow
import numpy as np


def log_evaluation_metrics(
    dataset_type: str,
    metrics: dict,
    confusion_matrix: np.ndarray,
) -> None:
    mlflow.log_metric(f"{dataset_type} Accuracy", metrics["accuracy"])
    mlflow.log_metric(f"{dataset_type} Precision", metrics["precision"])
    mlflow.log_metric(f"{dataset_type} Recall", metrics["recall"])
    mlflow.log_metric(f"{dataset_type} F1 Score", metrics["f1_score"])

    mlflow.log_metric(f"{dataset_type} True Positives", confusion_matrix[1, 1])
    mlflow.log_metric(f"{dataset_type} False Positives", confusion_matrix[0, 1])
    mlflow.log_metric(f"{dataset_type} False Negatives", confusion_matrix[1, 0])
    mlflow.log_metric(f"{dataset_type} True Negatives", confusion_matrix[0, 0])
