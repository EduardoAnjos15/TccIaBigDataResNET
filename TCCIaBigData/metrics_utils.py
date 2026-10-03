"""Métricas compartilhadas pelas baselines e pelos modelos neurais."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def metrics_from_confusion(matrix: torch.Tensor, classes: list[str]) -> dict:
    matrix = matrix.to(dtype=torch.int64, device="cpu")
    total = matrix.sum().item()
    per_class = []
    f1_values = []
    for index, name in enumerate(classes):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        support = matrix[index, :].sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": matrix.diag().sum().item() / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def metrics_from_predictions(
    truth: list[int] | torch.Tensor,
    predictions: list[int] | torch.Tensor,
    classes: list[str],
) -> dict:
    truth_tensor = torch.as_tensor(truth, dtype=torch.int64)
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.int64)
    if truth_tensor.shape != prediction_tensor.shape:
        raise ValueError("Rótulos e predições devem ter o mesmo tamanho.")
    encoded = truth_tensor * len(classes) + prediction_tensor
    matrix = torch.bincount(encoded, minlength=len(classes) ** 2).reshape(
        len(classes), len(classes)
    )
    return metrics_from_confusion(matrix, classes)


def save_confusion_matrix(
    matrix: list[list[int]], classes: list[str], output_path: Path, title: str
) -> None:
    values = torch.as_tensor(matrix).numpy()
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe predita",
        ylabel="Classe real",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

