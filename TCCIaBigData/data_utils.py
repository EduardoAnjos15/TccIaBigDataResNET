"""Divisão reproduzível e compartilhada do conjunto EuroSAT."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


class TransformSubset(Dataset):
    """Aplica uma transformação a um subconjunto definido por índices."""

    def __init__(self, dataset: Dataset, indices: list[int], transform=None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, label = self.dataset[self.indices[position]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def default_split_file(data_dir: Path, seed: int) -> Path:
    return data_dir / "splits" / f"eurosat_stratified_seed{seed}.json"


def _create_stratified_indices(targets: list[int], seed: int) -> dict[str, list[int]]:
    """Divide cada classe separadamente e depois embaralha cada partição."""
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)

    generator = torch.Generator().manual_seed(seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for class_id in sorted(indices_by_class):
        class_indices = indices_by_class[class_id]
        order = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[position] for position in order]
        train_end = int(len(shuffled) * SPLIT_RATIOS["train"])
        val_end = train_end + int(len(shuffled) * SPLIT_RATIOS["val"])
        splits["train"].extend(shuffled[:train_end])
        splits["val"].extend(shuffled[train_end:val_end])
        splits["test"].extend(shuffled[val_end:])

    for name in splits:
        order = torch.randperm(len(splits[name]), generator=generator).tolist()
        splits[name] = [splits[name][position] for position in order]
    return splits


def _validate_split(
    payload: dict, dataset_size: int, classes: list[str], seed: int
) -> dict[str, list[int]]:
    if payload.get("dataset") != "EuroSAT":
        raise ValueError("O arquivo de divisão não pertence ao EuroSAT.")
    if payload.get("dataset_size") != dataset_size:
        raise ValueError("O tamanho do dataset difere do arquivo de divisão salvo.")
    if payload.get("classes") != classes:
        raise ValueError("As classes diferem do arquivo de divisão salvo.")
    if payload.get("seed") != seed:
        raise ValueError("A seed difere do arquivo de divisão salvo.")

    splits = payload.get("indices", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("O arquivo deve conter train, val e test.")
    all_indices = [index for name in ("train", "val", "test") for index in splits[name]]
    if len(all_indices) != dataset_size or len(set(all_indices)) != dataset_size:
        raise ValueError("A divisão contém índices ausentes ou repetidos.")
    if min(all_indices) < 0 or max(all_indices) >= dataset_size:
        raise ValueError("A divisão contém índice fora do dataset.")
    return {name: [int(index) for index in splits[name]] for name in splits}


def load_or_create_split(
    dataset,
    data_dir: Path,
    seed: int,
    split_file: Path | None = None,
) -> tuple[dict[str, list[int]], Path]:
    """Carrega a divisão congelada ou cria uma nova divisão estratificada."""
    path = split_file or default_split_file(data_dir, seed)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _validate_split(payload, len(dataset), list(dataset.classes), seed), path

    splits = _create_stratified_indices(list(dataset.targets), seed)
    payload = {
        "dataset": "EuroSAT",
        "dataset_size": len(dataset),
        "classes": list(dataset.classes),
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "indices": splits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits, path
