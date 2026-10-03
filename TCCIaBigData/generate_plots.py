"""Gera gráficos acadêmicos a partir dos resultados reais dos experimentos."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from data_utils import default_split_file, load_or_create_split


CLASS_LABELS_PT = {
    "AnnualCrop": "Cultivo anual",
    "Forest": "Floresta",
    "HerbaceousVegetation": "Vegetação herbácea",
    "Highway": "Rodovia",
    "Industrial": "Área industrial",
    "Pasture": "Pastagem",
    "PermanentCrop": "Cultivo permanente",
    "Residential": "Área residencial",
    "River": "Rio",
    "SeaLake": "Mar/Lago",
}

COLORS = {
    "train": "#3568A8",
    "val": "#E68632",
    "test": "#4C956C",
    "accuracy": "#3568A8",
    "macro_f1": "#E68632",
    "bar": "#3568A8",
    "point": "#7A5195",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.6,
            "legend.frameon": False,
        }
    )


def save_figure(figure: plt.Figure, png_path: Path) -> list[Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png_path, pdf_path]


def translated_classes(classes: list[str]) -> list[str]:
    return [CLASS_LABELS_PT.get(name, name) for name in classes]


def plot_class_distribution(
    dataset,
    split_indices: dict[str, list[int]],
    output_path: Path,
) -> list[Path]:
    class_names = translated_classes(list(dataset.classes))
    x = np.arange(len(class_names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(14, 7))
    for offset, split_name in enumerate(("train", "val", "test")):
        counts = Counter(dataset.targets[index] for index in split_indices[split_name])
        values = [counts.get(class_id, 0) for class_id in range(len(class_names))]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label={"train": "Treino", "val": "Validação", "test": "Teste"}[split_name],
            color=COLORS[split_name],
        )
        axis.bar_label(bars, padding=2, fontsize=7, rotation=90)
    axis.set_title("Distribuição estratificada das classes do EuroSAT")
    axis.set_xlabel("Classe")
    axis.set_ylabel("Quantidade de imagens")
    axis.set_xticks(x, class_names, rotation=28, ha="right")
    axis.grid(axis="y")
    axis.legend(ncols=3, loc="upper right")
    axis.set_ylim(0, axis.get_ylim()[1] * 1.08)
    figure.tight_layout()
    return save_figure(figure, output_path)


def plot_normalized_confusion(
    metrics: dict,
    classes: list[str],
    output_path: Path,
    title: str,
) -> list[Path]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    labels = translated_classes(classes)
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Proporção por classe real")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Classe predita",
        ylabel="Classe real",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "#222222",
                fontsize=8,
            )
    figure.tight_layout()
    return save_figure(figure, output_path)


def plot_per_class_f1(
    metrics: dict,
    classes: list[str],
    output_path: Path,
    title: str,
) -> list[Path]:
    by_class = {row["class"]: float(row["f1"]) for row in metrics["per_class"]}
    values = [by_class.get(name, 0.0) for name in classes]
    labels = translated_classes(classes)
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(y, values, color=COLORS["bar"])
    axis.set_title(title)
    axis.set_xlabel("F1-score")
    axis.set_ylabel("Classe")
    axis.set_yticks(y, labels)
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="x")
    axis.invert_yaxis()
    for bar, value in zip(bars, values):
        axis.text(
            min(value + 0.015, 0.965),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left" if value < 0.90 else "right",
            color="#222222",
        )
    figure.tight_layout()
    return save_figure(figure, output_path)


def experiment_name(payload: dict) -> str:
    config = payload.get("config", {})
    architecture = str(config.get("architecture", "modelo")).replace("resnet", "ResNet-")
    learning_rate = float(config.get("learning_rate", 0.0))
    learning_rate_label = f"lr={learning_rate:.0e}"
    if config.get("no_pretrained", False):
        suffix = "do zero"
    elif config.get("freeze_backbone", False):
        suffix = "ImageNet, congelada"
    else:
        suffix = "ImageNet, ajuste completo"
    return f"{architecture} ({suffix}; {learning_rate_label})"


def discover_validation_results(outputs_dir: Path, split_path: Path) -> list[dict]:
    expected_split = split_path.resolve()
    results = []
    for metrics_path in outputs_dir.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "validation" not in payload or not payload.get("split_file"):
            continue
        if Path(payload["split_file"]).resolve() != expected_split:
            continue
        results.append(
            {
                "name": experiment_name(payload),
                "seed": int(payload["config"]["seed"]),
                "metrics": payload["validation"],
                "training_seconds": float(payload.get("training_seconds", 0.0)),
                "parameters": int(payload.get("model", {}).get("total_parameters", 0)),
                "path": metrics_path,
                "payload": payload,
            }
        )
    return sorted(results, key=lambda row: (row["name"], row["seed"]))


def plot_baseline_comparison(
    naive_metrics: dict,
    experiments: list[dict],
    output_path: Path,
) -> list[Path]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped["Classe majoritária"].append(naive_metrics)
    for result in experiments:
        grouped[result["name"]].append(result["metrics"])

    names = list(grouped)
    accuracy_means = [np.mean([row["accuracy"] for row in grouped[name]]) for name in names]
    accuracy_stds = [np.std([row["accuracy"] for row in grouped[name]]) for name in names]
    f1_means = [np.mean([row["macro_f1"] for row in grouped[name]]) for name in names]
    f1_stds = [np.std([row["macro_f1"] for row in grouped[name]]) for name in names]

    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(10, len(names) * 2.2), 7))
    accuracy_bars = axis.bar(
        x - width / 2,
        accuracy_means,
        width,
        yerr=accuracy_stds,
        capsize=4,
        label="Acurácia",
        color=COLORS["accuracy"],
    )
    f1_bars = axis.bar(
        x + width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=4,
        label="Macro-F1",
        color=COLORS["macro_f1"],
    )
    axis.set_title("Comparação das baselines e configurações experimentais")
    axis.set_xlabel("Configuração")
    axis.set_ylabel("Desempenho")
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(x, [f"{name}\n(n={len(grouped[name])})" for name in names], rotation=18, ha="right")
    axis.grid(axis="y")
    axis.legend(ncols=2, loc="upper left")
    axis.bar_label(accuracy_bars, labels=[f"{value:.1%}" for value in accuracy_means], padding=3)
    axis.bar_label(f1_bars, labels=[f"{value:.1%}" for value in f1_means], padding=3)
    figure.tight_layout()
    return save_figure(figure, output_path)


def plot_efficiency(experiments: list[dict], output_path: Path) -> list[Path]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in experiments:
        grouped[result["name"]].append(result)
    figure, axis = plt.subplots(figsize=(10, 7))
    for name, rows in grouped.items():
        minutes = np.mean([row["training_seconds"] for row in rows]) / 60
        macro_f1 = np.mean([row["metrics"]["macro_f1"] for row in rows])
        axis.scatter(minutes, macro_f1, s=90, color=COLORS["point"])
        axis.annotate(name, (minutes, macro_f1), xytext=(7, 7), textcoords="offset points")
    axis.set_title("Relação entre desempenho e tempo de treinamento")
    axis.set_xlabel("Tempo médio de treinamento (minutos)")
    axis.set_ylabel("Macro-F1 médio")
    axis.grid()
    axis.margins(x=0.18, y=0.18)
    figure.tight_layout()
    return save_figure(figure, output_path)


def generate_all_plots(
    data_dir: Path,
    outputs_dir: Path,
    plots_dir: Path,
    seed: int,
    split_file: Path | None = None,
) -> tuple[list[Path], list[str]]:
    configure_style()
    dataset = datasets.EuroSAT(root=data_dir, download=False)
    split_indices, split_path = load_or_create_split(dataset, data_dir, seed, split_file)
    generated = []
    skipped = []
    generated.extend(
        plot_class_distribution(
            dataset, split_indices, plots_dir / "class_distribution.png"
        )
    )

    naive_path = outputs_dir / "baseline_naive" / "val_metrics.json"
    if naive_path.exists():
        naive_payload = json.loads(naive_path.read_text(encoding="utf-8"))
        if Path(naive_payload["split_file"]).resolve() == split_path.resolve():
            naive_metrics = naive_payload["metrics"]
            generated.extend(
                plot_normalized_confusion(
                    naive_metrics,
                    list(dataset.classes),
                    outputs_dir / "baseline_naive" / "val_confusion_matrix_normalized.png",
                    "Matriz de confusão normalizada — baseline ingênua",
                )
            )
            generated.extend(
                plot_per_class_f1(
                    naive_metrics,
                    list(dataset.classes),
                    outputs_dir / "baseline_naive" / "val_f1_per_class.png",
                    "F1-score por classe — baseline ingênua",
                )
            )
        else:
            naive_metrics = None
            skipped.append("A baseline ingênua pertence a outro split.")
    else:
        naive_metrics = None
        skipped.append("A baseline ingênua ainda não foi executada.")

    experiments = discover_validation_results(outputs_dir, split_path)
    for result in experiments:
        experiment_dir = result["path"].parent
        generated.extend(
            plot_normalized_confusion(
                result["metrics"],
                list(dataset.classes),
                experiment_dir / "validation_confusion_matrix_normalized.png",
                f"Matriz de confusão normalizada — {result['name']}",
            )
        )
        generated.extend(
            plot_per_class_f1(
                result["metrics"],
                list(dataset.classes),
                experiment_dir / "validation_f1_per_class.png",
                f"F1-score por classe — {result['name']}",
            )
        )

    if naive_metrics is not None and experiments:
        generated.extend(
            plot_baseline_comparison(
                naive_metrics, experiments, plots_dir / "baseline_comparison.png"
            )
        )
    else:
        skipped.append("A comparação aguarda ao menos um experimento válido de ResNet.")
    if len({result["name"] for result in experiments}) >= 2:
        generated.extend(plot_efficiency(experiments, plots_dir / "efficiency.png"))
    else:
        skipped.append("O gráfico de eficiência aguarda ao menos duas configurações.")
    return generated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera gráficos dos experimentos do TCC")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated, skipped = generate_all_plots(
        args.data_dir, args.outputs_dir, args.plots_dir, args.seed, args.split_file
    )
    print(f"Gráficos gerados: {len(generated)}")
    for path in generated:
        print(f"  {path.resolve()}")
    for message in skipped:
        print(f"Aguardando dados: {message}")


if __name__ == "__main__":
    main()
