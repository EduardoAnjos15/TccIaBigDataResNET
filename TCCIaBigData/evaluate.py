"""Avaliação final explícita de um checkpoint no conjunto de teste."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from data_utils import TransformSubset, build_eval_transform, load_or_create_split
from metrics_utils import save_confusion_matrix
from train import build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um checkpoint no teste do EuroSAT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size deve ser positivo")
    output_dir = args.output_dir or args.checkpoint.parent
    result_path = output_dir / "test_metrics.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"O resultado {result_path} já existe. Use outro diretório ou --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    training_seed = int(checkpoint["seed"])
    split_seed = int(checkpoint.get("split_seed", training_seed))
    checkpoint_split = checkpoint.get("split_file")
    if checkpoint_split is None and args.split_file is None:
        raise ValueError(
            "Este checkpoint é anterior ao protocolo de divisão persistida. "
            "Treine-o novamente ou informe explicitamente o --split-file original."
        )
    split_file = args.split_file or (Path(checkpoint_split) if checkpoint_split else None)

    dataset = datasets.EuroSAT(root=args.data_dir, download=not args.no_download)
    classes = list(dataset.classes)
    if classes != list(checkpoint["classes"]):
        raise ValueError("As classes do checkpoint diferem das classes do dataset.")
    split_indices, split_path = load_or_create_split(
        dataset, args.data_dir, split_seed, split_file
    )
    loader = DataLoader(
        TransformSubset(dataset, split_indices["test"], build_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(checkpoint["architecture"], len(classes), False, False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()
    started = time.time()
    test_metrics = evaluate(model, loader, criterion, device, classes)
    inference_seconds = time.time() - started
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": checkpoint["architecture"],
        "training_seed": training_seed,
        "split_seed": split_seed,
        "device": str(device),
        "split_file": str(split_path.resolve()),
        "test_size": len(split_indices["test"]),
        "inference_seconds": inference_seconds,
        "test": test_metrics,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    confusion_path = output_dir / "test_confusion_matrix.png"
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        classes,
        confusion_path,
        "Matriz de confusão — teste final",
    )
    print(
        f"Teste final: acc={test_metrics['accuracy']:.4f} "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Resultados salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
