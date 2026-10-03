"""Gera gráficos e relatório final usando apenas métricas de teste já salvas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from generate_plots import (
    CLASS_LABELS_PT,
    configure_style,
    plot_normalized_confusion,
    plot_per_class_f1,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera relatório final a partir das métricas já salvas"
    )
    parser.add_argument(
        "--test-metrics",
        type=Path,
        default=Path("outputs/final_resnet18_test/test_metrics.json"),
    )
    parser.add_argument(
        "--validation-metrics",
        type=Path,
        default=Path("outputs/baseline_resnet18/metrics.json"),
    )
    parser.add_argument(
        "--selection-decision",
        type=Path,
        default=Path("outputs/model_selection/selection_decision.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/final_resnet18_test")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_and_validate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    for path in (
        args.test_metrics,
        args.validation_metrics,
        args.selection_decision,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo necessário ausente: {path}")
    test_payload = json.loads(args.test_metrics.read_text(encoding="utf-8"))
    validation_payload = json.loads(
        args.validation_metrics.read_text(encoding="utf-8")
    )
    selection_payload = json.loads(
        args.selection_decision.read_text(encoding="utf-8")
    )

    selected_architecture = selection_payload["decision"]["selected_architecture"]
    selected_checkpoint = Path(
        selection_payload["decision"]["selected_checkpoint"]
    ).resolve()
    evaluated_checkpoint = Path(test_payload["checkpoint"]).resolve()
    if test_payload["architecture"] != selected_architecture:
        raise ValueError("A arquitetura avaliada difere da seleção prévia.")
    if evaluated_checkpoint != selected_checkpoint:
        raise ValueError("O checkpoint avaliado difere da seleção prévia.")
    if validation_payload["config"]["architecture"] != selected_architecture:
        raise ValueError("As métricas de validação pertencem a outra arquitetura.")
    if int(validation_payload["config"]["seed"]) != int(
        selection_payload["decision"]["selected_training_seed"]
    ):
        raise ValueError("A validação e a seleção usam seeds diferentes.")
    if Path(test_payload["split_file"]).resolve() != Path(
        selection_payload["split_file"]
    ).resolve():
        raise ValueError("O teste e a seleção usam splits diferentes.")
    return test_payload, validation_payload, selection_payload


def per_class_map(metrics: dict) -> dict[str, dict]:
    return {row["class"]: row for row in metrics["per_class"]}


def top_confusions(metrics: dict, classes: list[str], limit: int = 8) -> list[dict]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    rows = []
    for truth_index, truth_name in enumerate(classes):
        support = int(matrix[truth_index].sum())
        for predicted_index, predicted_name in enumerate(classes):
            if truth_index == predicted_index:
                continue
            count = int(matrix[truth_index, predicted_index])
            if count:
                rows.append(
                    {
                        "true_class": truth_name,
                        "predicted_class": predicted_name,
                        "count": count,
                        "percentage_of_true_class": count / support,
                    }
                )
    return sorted(rows, key=lambda row: row["count"], reverse=True)[:limit]


def plot_validation_vs_test(
    validation: dict, test: dict, output_path: Path
) -> list[Path]:
    labels = ["Acurácia", "Macro-F1"]
    validation_values = [validation["accuracy"], validation["macro_f1"]]
    test_values = [test["accuracy"], test["macro_f1"]]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8, 6))
    val_bars = axis.bar(
        x - width / 2,
        validation_values,
        width,
        label="Validação",
        color="#3568A8",
    )
    test_bars = axis.bar(
        x + width / 2,
        test_values,
        width,
        label="Teste final",
        color="#4C956C",
    )
    lower = min(validation_values + test_values) - 0.006
    upper = max(validation_values + test_values) + 0.004
    axis.set_ylim(lower, upper)
    axis.set_title("Generalização da ResNet18 selecionada")
    axis.set_xlabel("Métrica")
    axis.set_ylabel("Desempenho (eixo ampliado)")
    axis.set_xticks(x, labels)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axis.grid(axis="y")
    axis.legend()
    axis.bar_label(
        val_bars,
        labels=[f"{value:.2%}" for value in validation_values],
        padding=3,
    )
    axis.bar_label(
        test_bars,
        labels=[f"{value:.2%}" for value in test_values],
        padding=3,
    )
    figure.text(
        0.01,
        0.01,
        "Comparação do mesmo checkpoint; o eixo vertical está ampliado.",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    return save_figure(figure, output_path)


def plot_per_class_comparison(
    validation: dict,
    test: dict,
    classes: list[str],
    output_path: Path,
) -> list[Path]:
    val_by_class = per_class_map(validation)
    test_by_class = per_class_map(test)
    labels = [CLASS_LABELS_PT.get(name, name) for name in classes]
    val_values = [float(val_by_class[name]["f1"]) for name in classes]
    test_values = [float(test_by_class[name]["f1"]) for name in classes]
    y = np.arange(len(classes))
    height = 0.36
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(
        y - height / 2,
        val_values,
        height,
        label="Validação",
        color="#3568A8",
    )
    axis.barh(
        y + height / 2,
        test_values,
        height,
        label="Teste final",
        color="#4C956C",
    )
    lower = min(val_values + test_values) - 0.015
    axis.set_xlim(max(0.0, lower), 1.002)
    axis.set_title("F1-score por classe — validação × teste")
    axis.set_xlabel("F1-score (eixo ampliado)")
    axis.set_ylabel("Classe")
    axis.set_yticks(y, labels)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.grid(axis="x")
    axis.legend(loc="lower right")
    axis.invert_yaxis()
    figure.text(
        0.01,
        0.01,
        "Comparação do mesmo checkpoint; o eixo horizontal está ampliado.",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    return save_figure(figure, output_path)


def write_per_class_csv(
    validation: dict, test: dict, classes: list[str], output_path: Path
) -> None:
    val_by_class = per_class_map(validation)
    test_by_class = per_class_map(test)
    fields = [
        "class",
        "class_pt",
        "validation_support",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_support",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_minus_validation_f1_percentage_points",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_name in classes:
            val_row = val_by_class[class_name]
            test_row = test_by_class[class_name]
            writer.writerow(
                {
                    "class": class_name,
                    "class_pt": CLASS_LABELS_PT.get(class_name, class_name),
                    "validation_support": val_row["support"],
                    "validation_precision": val_row["precision"],
                    "validation_recall": val_row["recall"],
                    "validation_f1": val_row["f1"],
                    "test_support": test_row["support"],
                    "test_precision": test_row["precision"],
                    "test_recall": test_row["recall"],
                    "test_f1": test_row["f1"],
                    "test_minus_validation_f1_percentage_points": (
                        float(test_row["f1"]) - float(val_row["f1"])
                    )
                    * 100.0,
                }
            )


def write_report(
    test_payload: dict,
    validation_payload: dict,
    classes: list[str],
    confusions: list[dict],
    output_path: Path,
) -> None:
    test = test_payload["test"]
    validation = validation_payload["validation"]
    test_by_class = per_class_map(test)
    ordered_classes = sorted(
        classes, key=lambda name: float(test_by_class[name]["f1"])
    )
    accuracy_gap = (float(test["accuracy"]) - float(validation["accuracy"])) * 100
    macro_gap = (float(test["macro_f1"]) - float(validation["macro_f1"])) * 100
    lines = [
        "# Relatório da avaliação final",
        "",
        "Modelo: ResNet18 pré-treinada no ImageNet com ajuste completo.",
        "",
        f"Checkpoint: `{test_payload['checkpoint']}`.",
        "",
        f"Imagens no teste: {test_payload['test_size']}.",
        "",
        "## Resultado principal",
        "",
        f"- Acurácia: **{float(test['accuracy']):.4%}**;",
        f"- macro-F1: **{float(test['macro_f1']):.4%}**;",
        f"- diferença de acurácia entre teste e validação: {accuracy_gap:+.3f} ponto percentual;",
        f"- diferença de macro-F1 entre teste e validação: {macro_gap:+.3f} ponto percentual.",
        "",
        "O teste ficou muito próximo e ligeiramente acima da validação do mesmo checkpoint, indicando boa capacidade de generalização e ausência de queda relevante fora dos dados usados para seleção.",
        "",
        "## F1 por classe no teste",
        "",
        "| Classe | F1 | Precisão | Recall | Suporte |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in ordered_classes:
        row = test_by_class[class_name]
        label = CLASS_LABELS_PT.get(class_name, class_name)
        lines.append(
            f"| {label} | {float(row['f1']):.2%} | {float(row['precision']):.2%} | {float(row['recall']):.2%} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Principais confusões dirigidas",
            "",
            "| Classe real | Classe predita | Imagens | Percentual da classe real |",
            "|---|---|---:|---:|",
        ]
    )
    for row in confusions:
        truth = CLASS_LABELS_PT.get(row["true_class"], row["true_class"])
        prediction = CLASS_LABELS_PT.get(
            row["predicted_class"], row["predicted_class"]
        )
        lines.append(
            f"| {truth} | {prediction} | {row['count']} | {row['percentage_of_true_class']:.2%} |"
        )
    lines.extend(
        [
            "",
            "A classe com menor F1 foi cultivo permanente. Isso é coerente com a semelhança visual entre diferentes tipos de áreas agrícolas. Mar/lago e floresta apresentaram os maiores valores de F1.",
            "",
            "Este relatório foi produzido somente a partir do `test_metrics.json` já salvo. Nenhuma nova inferência, seleção de modelo ou alteração de hiperparâmetro foi realizada.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "final_test_report.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"O relatório {report_path} já existe. Use --overwrite para regenerá-lo."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_payload, validation_payload, _ = load_and_validate(args)
    test = test_payload["test"]
    validation = validation_payload["validation"]
    classes = [row["class"] for row in test["per_class"]]
    if set(classes) != {row["class"] for row in validation["per_class"]}:
        raise ValueError("As classes de validação e teste diferem.")

    configure_style()
    generated = []
    generated.extend(
        plot_normalized_confusion(
            test,
            classes,
            args.output_dir / "test_confusion_matrix_normalized.png",
            "Matriz de confusão normalizada — teste final",
        )
    )
    generated.extend(
        plot_per_class_f1(
            test,
            classes,
            args.output_dir / "test_f1_per_class.png",
            "F1-score por classe — teste final",
        )
    )
    generated.extend(
        plot_validation_vs_test(
            validation,
            test,
            args.output_dir / "validation_vs_test.png",
        )
    )
    generated.extend(
        plot_per_class_comparison(
            validation,
            test,
            classes,
            args.output_dir / "validation_vs_test_f1_per_class.png",
        )
    )
    csv_path = args.output_dir / "final_test_per_class.csv"
    write_per_class_csv(validation, test, classes, csv_path)
    confusions = top_confusions(test, classes)
    confusions_path = args.output_dir / "top_confusions.json"
    confusions_path.write_text(
        json.dumps(confusions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(test_payload, validation_payload, classes, confusions, report_path)

    print(f"Relatório: {report_path.resolve()}")
    print(f"CSV por classe: {csv_path.resolve()}")
    print(f"Confusões: {confusions_path.resolve()}")
    print("Gráficos:")
    for path in generated:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
