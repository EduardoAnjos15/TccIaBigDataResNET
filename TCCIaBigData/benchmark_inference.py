"""Benchmark reproduzível de inferência das ResNets na validação do EuroSAT."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from data_utils import TransformSubset, build_eval_transform, load_or_create_split
from train import build_model


DEFAULT_CHECKPOINTS = (
    Path("outputs/baseline_resnet18/best_model.pt"),
    Path("outputs/resnet34_pretrained/best_model.pt"),
    Path("outputs/resnet50_pretrained/best_model.pt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede custo de inferência das ResNets na validação do EuroSAT"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/inference_benchmark")
    )
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument(
        "--model-iterations",
        type=int,
        default=100,
        help="Quantidade de forwards por repetição na medição exclusiva do modelo.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 usa toda a validação; valores positivos servem para teste curto.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoints:
        raise ValueError("Informe ao menos um checkpoint.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("Todos os batch sizes devem ser positivos.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Não repita valores em --batch-sizes.")
    if args.repeats < 2:
        raise ValueError("Use ao menos duas repetições para calcular a variação.")
    if args.model_iterations < 1:
        raise ValueError("model-iterations deve ser positivo.")
    if args.warmup_batches < 0 or args.max_batches < 0:
        raise ValueError("warmup-batches e max-batches não podem ser negativos.")
    if args.num_workers < 0:
        raise ValueError("num-workers não pode ser negativo.")
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoints não encontrados: {missing}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def limited_batches(loader: DataLoader, max_batches: int):
    if max_batches:
        yield from itertools.islice(loader, max_batches)
    else:
        yield from loader


def make_loader(
    dataset,
    validation_indices: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        options["persistent_workers"] = True
    return DataLoader(
        TransformSubset(dataset, validation_indices, build_eval_transform()),
        **options,
    )


@torch.inference_mode()
def warm_up(
    model,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> None:
    if warmup_batches == 0:
        return
    iterator = iter(loader)
    completed = 0
    while completed < warmup_batches:
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        model(images)
        completed += 1
    synchronize(device)


@torch.inference_mode()
def measure_pipeline(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict:
    """Mede leitura, transformação, transferência e execução do modelo."""
    correct = torch.zeros((), dtype=torch.int64, device=device)
    seen = 0
    batches = 0
    synchronize(device)
    started = time.perf_counter()
    for images, labels in limited_batches(loader, max_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum()
        seen += labels.size(0)
        batches += 1
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "samples": seen,
        "batches": batches,
        "accuracy_on_measured_samples": correct.item() / seen,
    }


@torch.inference_mode()
def measure_model_only(
    model,
    images: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> dict:
    """Mede forwards repetidos com um lote já carregado no dispositivo."""
    seen = images.size(0) * iterations
    batches = iterations
    if device.type == "cuda":
        elapsed_ms = 0.0
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model(images)
            end_event.record()
            # Evita acumular forwards assíncronos e inflar o pico de memória.
            end_event.synchronize()
            elapsed_ms += start_event.elapsed_time(end_event)
        seconds = elapsed_ms / 1000.0
    else:
        seconds = 0.0
        for _ in range(iterations):
            started = time.perf_counter()
            model(images)
            seconds += time.perf_counter() - started
    return {"seconds": seconds, "samples": seen, "batches": batches}


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def summarize_runs(runs: list[dict]) -> dict:
    samples = runs[0]["samples"]
    batches = runs[0]["batches"]
    if any(run["samples"] != samples or run["batches"] != batches for run in runs):
        raise RuntimeError("As repetições processaram quantidades diferentes de dados.")
    seconds = [float(run["seconds"]) for run in runs]
    per_image_ms = [value * 1000.0 / samples for value in seconds]
    per_batch_ms = [value * 1000.0 / batches for value in seconds]
    throughput = [samples / value for value in seconds]
    return {
        "samples_per_repeat": samples,
        "batches_per_repeat": batches,
        "seconds": summarize(seconds),
        "latency_per_image_ms": summarize(per_image_ms),
        "latency_per_batch_ms": summarize(per_batch_ms),
        "throughput_images_per_second": summarize(throughput),
    }


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> tuple[dict, int, int, Path]:
    # Mantém o state_dict na CPU para ele não contaminar a memória medida na GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_seed = int(checkpoint["seed"])
    split_seed = int(checkpoint.get("split_seed", training_seed))
    checkpoint_split = checkpoint.get("split_file")
    if checkpoint_split is None:
        raise ValueError(
            f"{checkpoint_path} não registra o split original. "
            "Informe --split-file apenas após confirmar a divisão correta."
        )
    return checkpoint, training_seed, split_seed, Path(checkpoint_split)


def benchmark_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict,
    dataset,
    validation_indices: list[int],
    device: torch.device,
    batch_sizes: list[int],
    repeats: int,
    warmup_batches: int,
    model_iterations: int,
    num_workers: int,
    max_batches: int,
) -> list[dict]:
    architecture = str(checkpoint["architecture"])
    model = build_model(architecture, len(dataset.classes), False, False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    synchronize(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model_memory_mb = (
        torch.cuda.memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    results = []

    for batch_size in batch_sizes:
        loader = make_loader(
            dataset, validation_indices, batch_size, num_workers, device
        )
        warm_up(model, loader, device, warmup_batches)
        model_images, _ = next(iter(loader))
        model_images = model_images.to(device, non_blocking=True)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pipeline_runs = [
            measure_pipeline(model, loader, device, max_batches)
            for _ in range(repeats)
        ]
        pipeline_peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model_only_runs = [
            measure_model_only(model, model_images, device, model_iterations)
            for _ in range(repeats)
        ]
        measured_accuracies = [
            float(run["accuracy_on_measured_samples"]) for run in pipeline_runs
        ]
        if max(measured_accuracies) != min(measured_accuracies):
            raise RuntimeError("A acurácia variou entre repetições determinísticas.")
        model_only_peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        peak_memory_mb = max(
            pipeline_peak_memory_mb, model_only_peak_memory_mb
        )
        results.append(
            {
                "architecture": architecture,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
                "parameters": parameters,
                "batch_size": batch_size,
                "model_memory_mb": model_memory_mb,
                "peak_inference_memory_mb": peak_memory_mb,
                "pipeline_peak_memory_mb": pipeline_peak_memory_mb,
                "model_only_peak_memory_mb": model_only_peak_memory_mb,
                "accuracy_on_measured_samples": measured_accuracies[0],
                "pipeline": summarize_runs(pipeline_runs),
                "model_only": summarize_runs(model_only_runs),
            }
        )
        print(
            f"{architecture} | batch={batch_size} | "
            f"modelo={results[-1]['model_only']['latency_per_image_ms']['mean']:.3f} ms/imagem | "
            f"pipeline={results[-1]['pipeline']['throughput_images_per_second']['mean']:.1f} img/s | "
            f"memória={peak_memory_mb:.1f} MB"
        )
        del loader
        del model_images

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def write_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = [
        "architecture",
        "batch_size",
        "checkpoint",
        "checkpoint_size_mb",
        "parameters",
        "model_memory_mb",
        "peak_inference_memory_mb",
        "pipeline_peak_memory_mb",
        "model_only_peak_memory_mb",
        "accuracy_on_measured_samples",
        "model_latency_per_image_ms_mean",
        "model_latency_per_image_ms_std",
        "model_throughput_images_per_second_mean",
        "model_throughput_images_per_second_std",
        "pipeline_latency_per_image_ms_mean",
        "pipeline_latency_per_image_ms_std",
        "pipeline_throughput_images_per_second_mean",
        "pipeline_throughput_images_per_second_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "architecture": result["architecture"],
                    "batch_size": result["batch_size"],
                    "checkpoint": result["checkpoint"],
                    "checkpoint_size_mb": result["checkpoint_size_mb"],
                    "parameters": result["parameters"],
                    "model_memory_mb": result["model_memory_mb"],
                    "peak_inference_memory_mb": result[
                        "peak_inference_memory_mb"
                    ],
                    "pipeline_peak_memory_mb": result["pipeline_peak_memory_mb"],
                    "model_only_peak_memory_mb": result[
                        "model_only_peak_memory_mb"
                    ],
                    "accuracy_on_measured_samples": result[
                        "accuracy_on_measured_samples"
                    ],
                    "model_latency_per_image_ms_mean": result["model_only"][
                        "latency_per_image_ms"
                    ]["mean"],
                    "model_latency_per_image_ms_std": result["model_only"][
                        "latency_per_image_ms"
                    ]["sample_std"],
                    "model_throughput_images_per_second_mean": result[
                        "model_only"
                    ]["throughput_images_per_second"]["mean"],
                    "model_throughput_images_per_second_std": result[
                        "model_only"
                    ]["throughput_images_per_second"]["sample_std"],
                    "pipeline_latency_per_image_ms_mean": result["pipeline"][
                        "latency_per_image_ms"
                    ]["mean"],
                    "pipeline_latency_per_image_ms_std": result["pipeline"][
                        "latency_per_image_ms"
                    ]["sample_std"],
                    "pipeline_throughput_images_per_second_mean": result[
                        "pipeline"
                    ]["throughput_images_per_second"]["mean"],
                    "pipeline_throughput_images_per_second_std": result[
                        "pipeline"
                    ]["throughput_images_per_second"]["sample_std"],
                }
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    json_path = args.output_dir / "benchmark_results.json"
    csv_path = args.output_dir / "benchmark_summary.csv"
    if not args.overwrite and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            f"Já existem resultados em {args.output_dir}. "
            "Use outro --output-dir ou informe --overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    dataset = datasets.EuroSAT(root=args.data_dir, download=not args.no_download)
    classes = list(dataset.classes)
    print(f"Dispositivo: {device}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Batch sizes: {args.batch_sizes} | repetições: {args.repeats}")

    results = []
    split_path_used: Path | None = None
    validation_size: int | None = None
    checkpoint_metadata = []
    for checkpoint_path in args.checkpoints:
        checkpoint, training_seed, split_seed, checkpoint_split = (
            load_checkpoint_metadata(checkpoint_path)
        )
        if list(checkpoint["classes"]) != classes:
            raise ValueError(
                f"As classes de {checkpoint_path} diferem das classes do EuroSAT."
            )
        requested_split = args.split_file or checkpoint_split
        split_indices, split_path = load_or_create_split(
            dataset, args.data_dir, split_seed, requested_split
        )
        if split_path_used is not None and split_path.resolve() != split_path_used:
            raise ValueError("Os checkpoints não utilizam a mesma divisão dos dados.")
        split_path_used = split_path.resolve()
        validation_size = len(split_indices["val"])
        checkpoint_metadata.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "architecture": checkpoint["architecture"],
                "training_seed": training_seed,
                "split_seed": split_seed,
            }
        )
        results.extend(
            benchmark_checkpoint(
                checkpoint_path,
                checkpoint,
                dataset,
                split_indices["val"],
                device,
                args.batch_sizes,
                args.repeats,
                args.warmup_batches,
                args.model_iterations,
                args.num_workers,
                args.max_batches,
            )
        )
        del checkpoint

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: [str(item) for item in value]
            if key == "checkpoints"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
        "dataset": "EuroSAT",
        "partition": "val",
        "split_file": str(split_path_used),
        "validation_size": validation_size,
        "checkpoint_metadata": checkpoint_metadata,
        "results": results,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(results, csv_path)
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
