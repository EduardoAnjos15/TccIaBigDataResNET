"""Baseline que sempre prevê a classe majoritária do conjunto de treino."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from torchvision import datasets

from data_utils import load_or_create_split
from metrics_utils import metrics_from_predictions, save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a baseline ingênua no EuroSAT")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_naive"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
        help="Use test somente na avaliação final",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / f"{args.evaluation_split}_metrics.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"O resultado {output_path} já existe. Use outro diretório ou --overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.EuroSAT(root=args.data_dir, download=not args.no_download)
    split_indices, split_path = load_or_create_split(
        dataset, args.data_dir, args.seed, args.split_file
    )
    train_counts = Counter(dataset.targets[index] for index in split_indices["train"])
    majority_class_id, majority_count = max(
        train_counts.items(), key=lambda item: (item[1], -item[0])
    )
    truth = [dataset.targets[index] for index in split_indices[args.evaluation_split]]
    predictions = [majority_class_id] * len(truth)
    metrics = metrics_from_predictions(truth, predictions, list(dataset.classes))

    result = {
        "baseline": "majority_class",
        "seed": args.seed,
        "evaluation_split": args.evaluation_split,
        "split_file": str(split_path.resolve()),
        "dataset_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "majority_class": dataset.classes[majority_class_id],
        "majority_class_id": majority_class_id,
        "majority_count_in_train": majority_count,
        "train_class_counts": {
            dataset.classes[class_id]: train_counts.get(class_id, 0)
            for class_id in range(len(dataset.classes))
        },
        "metrics": metrics,
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    confusion_path = args.output_dir / f"{args.evaluation_split}_confusion_matrix.png"
    save_confusion_matrix(
        metrics["confusion_matrix"],
        list(dataset.classes),
        confusion_path,
        f"Baseline ingênua — {args.evaluation_split}",
    )
    print(f"Classe majoritária: {dataset.classes[majority_class_id]} ({majority_count})")
    print(
        f"{args.evaluation_split}: acc={metrics['accuracy']:.4f} "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )
    print(f"Resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

