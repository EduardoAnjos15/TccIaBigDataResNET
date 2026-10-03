"""Treinamento reproduzível de uma ResNet no conjunto EuroSAT."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from data_utils import (
    TransformSubset,
    build_eval_transform,
    build_train_transform,
    load_or_create_split,
)
from metrics_utils import metrics_from_confusion, save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina ResNet no EuroSAT")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run1"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--architecture", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 no Windows para evitar erros de multiprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Seed da inicialização e do treinamento")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed fixa da divisão dos dados")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Permite substituir resultados existentes")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    base = datasets.EuroSAT(root=args.data_dir, download=not args.no_download)
    split_indices, split_path = load_or_create_split(
        base, args.data_dir, args.split_seed, args.split_file
    )

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            TransformSubset(base, split_indices["train"], train_transform),
            shuffle=True,
            generator=train_generator,
            **common,
        ),
        "val": DataLoader(
            TransformSubset(base, split_indices["val"], eval_transform),
            shuffle=False,
            **common,
        ),
    }
    sizes = {name: len(indices) for name, indices in split_indices.items()}
    return loaders, sizes, base.classes, split_path


def build_model(architecture: str, num_classes: int, pretrained: bool, freeze: bool):
    builders = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    }
    builder, default_weights = builders[architecture]
    model = builder(weights=default_weights if pretrained else None)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    if training and freeze_backbone:
        # Mantém inclusive as estatísticas de BatchNorm congeladas.
        model.eval()
        model.fc.train()
    else:
        model.train(training)
    loss_sum = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size
    return {"loss": loss_sum / seen, "accuracy": correct / seen}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, classes: list[str]):
    model.eval()
    matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    loss_sum = 0.0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        for truth, prediction in zip(labels.cpu(), predictions.cpu()):
            matrix[truth.long(), prediction.long()] += 1
        seen += labels.size(0)

    metrics = metrics_from_confusion(matrix, classes)
    metrics["loss"] = loss_sum / seen
    return metrics


def save_learning_curves(history: list[dict], output_path: Path) -> None:
    """Salva as curvas de aprendizado de treino e validação em PNG."""
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    train_accuracy = [row["train"]["accuracy"] for row in history]
    val_accuracy = [row["val"]["accuracy"] for row in history]
    best_epoch_index = min(range(len(val_loss)), key=val_loss.__getitem__)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Treino")
    axes[0].plot(epochs, val_loss, marker="o", label="Validação")
    axes[0].scatter(
        epochs[best_epoch_index],
        val_loss[best_epoch_index],
        color="red",
        zorder=3,
        label=f"Melhor época ({epochs[best_epoch_index]})",
    )
    axes[0].set_title("Perda durante o treinamento")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="Treino")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Validação")
    axes[1].set_title("Acurácia durante o treinamento")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("Curvas de aprendizado")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs e batch-size devem ser positivos")
    seed_everything(args.seed)
    result_files = (
        args.output_dir / "metrics.json",
        args.output_dir / "best_model.pt",
    )
    if not args.overwrite and any(path.exists() for path in result_files):
        raise FileExistsError(
            f"Já existem resultados em {args.output_dir}. "
            "Escolha outro --output-dir ou use --overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    loaders, sizes, classes, split_path = build_dataloaders(args, device)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Divisão: {sizes}")
    print(f"Arquivo de divisão: {split_path.resolve()}")
    model = build_model(
        args.architecture, len(classes), not args.no_pretrained, args.freeze_backbone
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    # Separa a aleatoriedade dos dados daquela consumida ao construir cada arquitetura.
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            freeze_backbone=args.freeze_backbone,
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        scheduler.step(val_metrics["loss"])
        print(
            f"Época {epoch:02d}/{args.epochs} | "
            f"treino loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": args.architecture,
                    "classes": classes,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "split_file": str(split_path.resolve()),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping acionado.")
                break
    training_seconds = time.time() - training_started

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation_started = time.time()
    validation_metrics = evaluate(model, loaders["val"], criterion, device, classes)
    validation_seconds = time.time() - validation_started
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "split_file": str(split_path.resolve()),
        "dataset_sizes": sizes,
        "model": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "history": history,
        "validation": validation_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    curves_path = args.output_dir / "learning_curves.png"
    save_learning_curves(history, curves_path)
    confusion_path = args.output_dir / "validation_confusion_matrix.png"
    save_confusion_matrix(
        validation_metrics["confusion_matrix"],
        classes,
        confusion_path,
        "Matriz de confusão — validação",
    )
    print(
        f"Validação: acc={validation_metrics['accuracy']:.4f} "
        f"macro-F1={validation_metrics['macro_f1']:.4f}"
    )
    print(f"Curvas de aprendizado salvas em: {curves_path.resolve()}")
    print(f"Resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
