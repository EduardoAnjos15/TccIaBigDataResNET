"""Consolida desempenho e inferência, gera gráficos e registra a seleção."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ARCHITECTURES = ("resnet18", "resnet34", "resnet50")
EXPECTED_SEEDS = (42, 123, 2026)
MODEL_LABELS = {
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
}
MODEL_COLORS = {
    "resnet18": "#3568A8",
    "resnet34": "#E68632",
    "resnet50": "#7A5195",
}
OFFICIAL_EXPERIMENT_DIRS = {
    "resnet18": {
        42: "baseline_resnet18",
        123: "resnet18_pretrained_seed123",
        2026: "resnet18_pretrained_seed2026",
    },
    "resnet34": {
        42: "resnet34_pretrained",
        123: "resnet34_pretrained_seed123",
        2026: "resnet34_pretrained_seed2026",
    },
    "resnet50": {
        42: "resnet50_pretrained",
        123: "resnet50_pretrained_seed123",
        2026: "resnet50_pretrained_seed2026",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de Pareto e registra a seleção do modelo"
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        default=Path("outputs/inference_benchmark/benchmark_results.json"),
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument(
        "--decision-dir", type=Path, default=Path("outputs/model_selection")
    )
    parser.add_argument(
        "--decision-md", type=Path, default=Path("MODEL_SELECTION.md")
    )
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=0.5,
        help="Perda máxima aceitável de macro-F1, em pontos percentuais.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.65,
        }
    )


def sample_summary(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("São necessárias ao menos duas seeds por arquitetura.")
    return statistics.mean(values), statistics.stdev(values)


def resolve_recorded_path(path_value: str, metrics_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (metrics_path.parent / path).resolve()


def load_validation_summaries(outputs_dir: Path) -> tuple[list[dict], Path]:
    rows = []
    official_split: Path | None = None
    for architecture in ARCHITECTURES:
        accuracy_values = []
        macro_f1_values = []
        source_metrics = []
        checkpoint_seed42: Path | None = None
        for seed in EXPECTED_SEEDS:
            experiment_dir = outputs_dir / OFFICIAL_EXPERIMENT_DIRS[architecture][seed]
            metrics_path = experiment_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Resultado oficial ausente: {metrics_path}")
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = payload["config"]
            if str(config["architecture"]) != architecture:
                raise ValueError(f"Arquitetura incompatível em {metrics_path}")
            if int(config["seed"]) != seed:
                raise ValueError(f"Seed incompatível em {metrics_path}")
            if config.get("no_pretrained", False) or config.get(
                "freeze_backbone", False
            ):
                raise ValueError(f"Configuração não é ajuste completo: {metrics_path}")
            if not math.isclose(
                float(config["learning_rate"]), 1e-4, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Learning rate incompatível em {metrics_path}")
            split_path = resolve_recorded_path(payload["split_file"], metrics_path)
            if official_split is None:
                official_split = split_path
            elif split_path != official_split:
                raise ValueError("Os experimentos oficiais não usam o mesmo split.")
            accuracy_values.append(float(payload["validation"]["accuracy"]))
            macro_f1_values.append(float(payload["validation"]["macro_f1"]))
            source_metrics.append(str(metrics_path.resolve()))
            if seed == 42:
                checkpoint_seed42 = experiment_dir / "best_model.pt"

        if checkpoint_seed42 is None or not checkpoint_seed42.is_file():
            raise FileNotFoundError(f"Checkpoint seed 42 ausente para {architecture}.")
        accuracy_mean, accuracy_std = sample_summary(accuracy_values)
        macro_f1_mean, macro_f1_std = sample_summary(macro_f1_values)
        rows.append(
            {
                "architecture": architecture,
                "label": MODEL_LABELS[architecture],
                "seeds": list(EXPECTED_SEEDS),
                "accuracy_mean": accuracy_mean,
                "accuracy_sample_std": accuracy_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_sample_std": macro_f1_std,
                "source_metrics": source_metrics,
                "checkpoint_seed42": str(checkpoint_seed42.resolve()),
            }
        )
    if official_split is None:
        raise RuntimeError("Nenhum experimento oficial foi carregado.")
    return rows, official_split


def load_benchmark_rows(benchmark_path: Path, official_split: Path) -> dict:
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark oficial ausente: {benchmark_path}")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if payload.get("partition") != "val":
        raise ValueError("O benchmark oficial deve usar somente validação.")
    if int(payload["config"].get("max_batches", -1)) != 0:
        raise ValueError("O benchmark informado não percorreu toda a validação.")
    if Path(payload["split_file"]).resolve() != official_split:
        raise ValueError("O benchmark e os treinamentos usam splits diferentes.")
    by_key = {}
    for result in payload["results"]:
        key = (str(result["architecture"]), int(result["batch_size"]))
        if key in by_key:
            raise ValueError(f"Medição duplicada no benchmark: {key}")
        by_key[key] = result
    for architecture in ARCHITECTURES:
        for batch_size in (1, 64):
            if (architecture, batch_size) not in by_key:
                raise ValueError(
                    f"Medição ausente para {architecture}, batch {batch_size}."
                )
    return by_key


def consolidate_rows(validation_rows: list[dict], benchmark: dict) -> list[dict]:
    consolidated = []
    for validation in validation_rows:
        architecture = validation["architecture"]
        batch1 = benchmark[(architecture, 1)]
        batch64 = benchmark[(architecture, 64)]
        consolidated.append(
            {
                **validation,
                "latency_batch1_ms_mean": batch1["model_only"][
                    "latency_per_image_ms"
                ]["mean"],
                "latency_batch1_ms_std": batch1["model_only"][
                    "latency_per_image_ms"
                ]["sample_std"],
                "throughput_batch64_mean": batch64["model_only"][
                    "throughput_images_per_second"
                ]["mean"],
                "throughput_batch64_std": batch64["model_only"][
                    "throughput_images_per_second"
                ]["sample_std"],
                "pipeline_throughput_batch64_mean": batch64["pipeline"][
                    "throughput_images_per_second"
                ]["mean"],
                "pipeline_throughput_batch64_std": batch64["pipeline"][
                    "throughput_images_per_second"
                ]["sample_std"],
                "memory_batch64_mb": batch64["model_only_peak_memory_mb"],
                "pipeline_memory_batch64_mb": batch64["pipeline_peak_memory_mb"],
                "checkpoint_size_mb": batch1["checkpoint_size_mb"],
                "parameters": batch1["parameters"],
            }
        )
    return consolidated


def select_model(rows: list[dict], tolerance_pp: float) -> dict:
    if tolerance_pp < 0:
        raise ValueError("tolerance-pp não pode ser negativa.")
    best_macro_f1 = max(row["macro_f1_mean"] for row in rows)
    cutoff = best_macro_f1 - tolerance_pp / 100.0
    eligible = [row for row in rows if row["macro_f1_mean"] >= cutoff]
    selected = min(eligible, key=lambda row: row["latency_batch1_ms_mean"])
    best_performance = max(rows, key=lambda row: row["macro_f1_mean"])
    return {
        "tolerance_percentage_points": tolerance_pp,
        "best_macro_f1_mean": best_macro_f1,
        "eligibility_cutoff": cutoff,
        "eligible_architectures": [row["architecture"] for row in eligible],
        "selected_architecture": selected["architecture"],
        "selected_checkpoint": selected["checkpoint_seed42"],
        "selected_training_seed": 42,
        "selected_macro_f1_mean": selected["macro_f1_mean"],
        "selected_latency_batch1_ms": selected["latency_batch1_ms_mean"],
        "best_performance_architecture": best_performance["architecture"],
        "macro_f1_gap_to_best_percentage_points": (
            best_macro_f1 - selected["macro_f1_mean"]
        )
        * 100.0,
        "latency_ratio_best_performance_over_selected": (
            best_performance["latency_batch1_ms_mean"]
            / selected["latency_batch1_ms_mean"]
        ),
        "checkpoint_size_ratio_best_performance_over_selected": (
            best_performance["checkpoint_size_mb"]
            / selected["checkpoint_size_mb"]
        ),
    }


def padded_limits(values: list[float], errors: list[float] | None = None):
    errors = errors or [0.0] * len(values)
    lower = min(value - error for value, error in zip(values, errors))
    upper = max(value + error for value, error in zip(values, errors))
    span = upper - lower
    padding = span * 0.22 if span > 0 else max(abs(upper) * 0.1, 1.0)
    return lower - padding, upper + padding


def draw_tradeoff(
    axis,
    rows: list[dict],
    selected_architecture: str,
    x_key: str,
    x_std_key: str | None,
    title: str,
    xlabel: str,
    ideal: str,
) -> None:
    x_values = [float(row[x_key]) for row in rows]
    x_errors = [float(row[x_std_key]) for row in rows] if x_std_key else None
    y_values = [float(row["macro_f1_mean"]) for row in rows]
    y_errors = [float(row["macro_f1_sample_std"]) for row in rows]

    for row in rows:
        architecture = row["architecture"]
        marker = "*" if architecture == selected_architecture else "o"
        marker_size = 14 if marker == "*" else 8
        axis.errorbar(
            row[x_key],
            row["macro_f1_mean"],
            xerr=row[x_std_key] if x_std_key else None,
            yerr=row["macro_f1_sample_std"],
            fmt=marker,
            markersize=marker_size,
            color=MODEL_COLORS[architecture],
            markeredgecolor="#222222",
            markeredgewidth=0.8,
            elinewidth=1.2,
            capsize=4,
            zorder=3,
        )
        offsets = {
            "resnet18": (7, -18),
            "resnet34": (7, 7),
            "resnet50": (-58, 7),
        }
        axis.annotate(
            row["label"],
            (row[x_key], row["macro_f1_mean"]),
            xytext=offsets[architecture],
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if architecture == selected_architecture else "normal",
        )

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Macro-F1 médio na validação (eixo ampliado)")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axis.set_xlim(*padded_limits(x_values, x_errors))
    axis.set_ylim(*padded_limits(y_values, y_errors))
    axis.grid(True)
    axis.text(
        0.02,
        0.98,
        f"Direção ideal: {ideal}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )


PLOT_SPECS = (
    (
        "latency_batch1_ms_mean",
        "latency_batch1_ms_std",
        "Desempenho × latência — batch 1",
        "Latência do modelo (ms/imagem; menor é melhor)",
        "superior esquerdo",
        "pareto_latency_batch1.png",
    ),
    (
        "throughput_batch64_mean",
        "throughput_batch64_std",
        "Desempenho × throughput — batch 64",
        "Throughput do modelo (imagens/s; maior é melhor)",
        "superior direito",
        "pareto_throughput_batch64.png",
    ),
    (
        "memory_batch64_mb",
        None,
        "Desempenho × memória — batch 64",
        "Pico sincronizado de memória (MB; menor é melhor)",
        "superior esquerdo",
        "pareto_memory_batch64.png",
    ),
    (
        "checkpoint_size_mb",
        None,
        "Desempenho × tamanho do checkpoint",
        "Tamanho do checkpoint (MB; menor é melhor)",
        "superior esquerdo",
        "pareto_checkpoint_size.png",
    ),
)


def save_figure(figure, png_path: Path) -> tuple[Path, Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def generate_plots(
    rows: list[dict], selected_architecture: str, plots_dir: Path
) -> list[Path]:
    generated = []
    for x_key, x_std_key, title, xlabel, ideal, filename in PLOT_SPECS:
        figure, axis = plt.subplots(figsize=(8.5, 6.5))
        draw_tradeoff(
            axis,
            rows,
            selected_architecture,
            x_key,
            x_std_key,
            title,
            xlabel,
            ideal,
        )
        figure.text(
            0.01,
            0.01,
            "Barras: desvio-padrão amostral; estrela: modelo selecionado; eixo Y ampliado.",
            fontsize=8,
            color="#444444",
        )
        figure.tight_layout(rect=(0, 0.035, 1, 1))
        generated.extend(save_figure(figure, plots_dir / filename))

    overview, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, spec in zip(axes.flat, PLOT_SPECS):
        x_key, x_std_key, title, xlabel, ideal, _ = spec
        draw_tradeoff(
            axis,
            rows,
            selected_architecture,
            x_key,
            x_std_key,
            title,
            xlabel,
            ideal,
        )
    overview.suptitle(
        "Custo-benefício das arquiteturas ResNet no EuroSAT",
        fontsize=16,
        fontweight="bold",
    )
    overview.text(
        0.01,
        0.01,
        "Macro-F1: média de três seeds. Barras: desvio-padrão amostral. "
        "Estrela: modelo selecionado. Todos os eixos Y estão ampliados.",
        fontsize=8,
        color="#444444",
    )
    overview.tight_layout(rect=(0, 0.035, 1, 0.96))
    generated.extend(save_figure(overview, plots_dir / "pareto_overview.png"))
    return generated


def write_summary_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "architecture",
        "accuracy_mean",
        "accuracy_sample_std",
        "macro_f1_mean",
        "macro_f1_sample_std",
        "latency_batch1_ms_mean",
        "latency_batch1_ms_std",
        "throughput_batch64_mean",
        "throughput_batch64_std",
        "pipeline_throughput_batch64_mean",
        "pipeline_throughput_batch64_std",
        "memory_batch64_mb",
        "pipeline_memory_batch64_mb",
        "checkpoint_size_mb",
        "parameters",
        "checkpoint_seed42",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_decision_markdown(
    rows: list[dict], decision: dict, path: Path, created_at: str
) -> None:
    selected = next(
        row
        for row in rows
        if row["architecture"] == decision["selected_architecture"]
    )
    best = next(
        row
        for row in rows
        if row["architecture"] == decision["best_performance_architecture"]
    )
    lines = [
        "# Registro prévio da seleção do modelo",
        "",
        f"Data UTC: {created_at}",
        "",
        "Status do conjunto de teste: **não avaliado**.",
        "",
        "## Regra de decisão",
        "",
        f"1. Considerar elegíveis as arquiteturas cuja média de macro-F1 fique até {decision['tolerance_percentage_points']:.2f} ponto percentual abaixo da melhor média de validação.",
        "2. Entre as elegíveis, escolher a arquitetura com menor latência média de inferência do modelo usando batch 1.",
        "3. Usar o checkpoint da seed 42, definido previamente como representante uniforme das arquiteturas no benchmark.",
        "",
        "## Resultado da aplicação da regra",
        "",
        f"Modelo selecionado: **{MODEL_LABELS[selected['architecture']]} pré-treinada com ajuste completo**.",
        "",
        f"Checkpoint selecionado: `{selected['checkpoint_seed42']}`.",
        "",
        f"Macro-F1 médio: {selected['macro_f1_mean']:.4%} ± {selected['macro_f1_sample_std']:.4%}.",
        "",
        f"Latência com batch 1: {selected['latency_batch1_ms_mean']:.3f} ± {selected['latency_batch1_ms_std']:.3f} ms/imagem.",
        "",
        f"A melhor média absoluta pertence à {MODEL_LABELS[best['architecture']]} ({best['macro_f1_mean']:.4%}), com vantagem de {decision['macro_f1_gap_to_best_percentage_points']:.3f} ponto percentual sobre o modelo selecionado.",
        "",
        f"A {MODEL_LABELS[best['architecture']]} apresentou latência {decision['latency_ratio_best_performance_over_selected']:.2f} vezes maior e checkpoint {decision['checkpoint_size_ratio_best_performance_over_selected']:.2f} vezes maior que o modelo selecionado.",
        "",
        "## Justificativa",
        "",
        "A ResNet18 permaneceu dentro da margem prática predefinida em relação ao melhor macro-F1 e apresentou o menor custo de inferência. Portanto, ela foi selecionada para a avaliação final única no teste. Este registro foi produzido sem consultar a partição de teste e não deve ser alterado em função do resultado final.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    decision_json = args.decision_dir / "selection_decision.json"
    if not args.overwrite and (decision_json.exists() or args.decision_md.exists()):
        raise FileExistsError(
            "Já existe um registro de seleção. Use --overwrite somente se a regra "
            "ainda não tiver sido usada para consultar o teste."
        )
    test_results = list(args.outputs_dir.rglob("test_metrics.json"))
    if test_results:
        raise RuntimeError(
            "Foram encontrados resultados de teste. O registro deve ocorrer antes "
            f"da avaliação final: {[str(path) for path in test_results]}"
        )

    configure_style()
    validation_rows, official_split = load_validation_summaries(args.outputs_dir)
    benchmark_rows = load_benchmark_rows(args.benchmark_results, official_split)
    rows = consolidate_rows(validation_rows, benchmark_rows)
    decision = select_model(rows, args.tolerance_pp)
    generated = generate_plots(
        rows, decision["selected_architecture"], args.plots_dir
    )
    summary_csv = args.plots_dir / "pareto_summary.csv"
    write_summary_csv(rows, summary_csv)

    created_at = datetime.now(timezone.utc).isoformat()
    decision_payload = {
        "created_at_utc": created_at,
        "test_evaluated_before_selection": False,
        "split_file": str(official_split),
        "benchmark_results": str(args.benchmark_results.resolve()),
        "rule": {
            "performance_tolerance_percentage_points": args.tolerance_pp,
            "primary_cost_metric": "model latency with batch size 1",
            "checkpoint_rule": "training seed 42",
        },
        "decision": decision,
        "candidates": rows,
    }
    args.decision_dir.mkdir(parents=True, exist_ok=True)
    decision_json.write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_decision_markdown(rows, decision, args.decision_md, created_at)

    print(f"Modelo selecionado: {decision['selected_architecture']}")
    print(f"Checkpoint: {decision['selected_checkpoint']}")
    print(f"Resumo CSV: {summary_csv.resolve()}")
    print(f"Decisão JSON: {decision_json.resolve()}")
    print(f"Decisão Markdown: {args.decision_md.resolve()}")
    print("Gráficos:")
    for path in generated:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
