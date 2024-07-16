import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List


def plot_train_and_val_losses(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
) -> None:
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_evaluation_result(
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_confusion_matrix: np.ndarray,
    test_confusion_matrix: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "Validation": [
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1_score"],
            ],
            "Test": [
                test_metrics["accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1_score"],
            ],
        },
        index=["Accuracy", "Precision", "Recall", "F1-Score"],
    )
    print(df)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        val_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Tumor", "Tumor"],
        yticklabels=["No Tumor", "Tumor"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        test_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Tumor", "Tumor"],
        yticklabels=["No Tumor", "Tumor"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")

    plt.show()
