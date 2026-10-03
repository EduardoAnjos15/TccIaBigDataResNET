"""Seleciona e apresenta amostras reproduzíveis do EuroSAT.

Este script usa apenas treino ou validação. A partição de teste é
intencionalmente indisponível para preservar o protocolo experimental.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from data_utils import load_or_create_split


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


def select_indices_by_class(
    dataset,
    partition_indices: list[int],
    samples_per_class: int,
    seed: int,
) -> dict[int, list[int]]:
    """Seleciona a mesma quantidade de exemplos por classe com seed fixa."""
    if samples_per_class < 1:
        raise ValueError("--samples-per-class deve ser maior que zero.")

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for dataset_index in partition_indices:
        class_id = int(dataset.targets[dataset_index])
        indices_by_class[class_id].append(int(dataset_index))

    generator = torch.Generator().manual_seed(seed)
    selected: dict[int, list[int]] = {}
    for class_id in range(len(dataset.classes)):
        candidates = indices_by_class[class_id]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"A classe {dataset.classes[class_id]} possui apenas "
                f"{len(candidates)} imagens na partição selecionada."
            )
        positions = torch.randperm(len(candidates), generator=generator)[
            :samples_per_class
        ].tolist()
        selected[class_id] = [candidates[position] for position in positions]
    return selected


def portable_source_path(source_path: str, data_dir: Path) -> str:
    """Prefere registrar um caminho relativo à pasta de dados."""
    source = Path(source_path).resolve()
    try:
        return source.relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return str(source)


def export_selected_images(
    dataset,
    selected: dict[int, list[int]],
    output_dir: Path,
) -> dict[int, list[Path]]:
    """Exporta cópias PNG sem transformações das imagens selecionadas."""
    exported: dict[int, list[Path]] = {}
    images_dir = output_dir / "selected_images"
    for class_id, dataset_indices in selected.items():
        class_dir = images_dir / dataset.classes[class_id]
        class_dir.mkdir(parents=True, exist_ok=True)
        exported[class_id] = []
        for order, dataset_index in enumerate(dataset_indices, start=1):
            image, _ = dataset[dataset_index]
            destination = class_dir / f"sample_{order:02d}_index_{dataset_index}.png"
            image.save(destination)
            exported[class_id].append(destination)
    return exported


def create_sample_panel(
    dataset,
    selected: dict[int, list[int]],
    partition: str,
    output_path: Path,
) -> tuple[Path, Path]:
    """Cria um painel compacto com duas classes por linha."""
    class_count = len(dataset.classes)
    rows = (class_count + 1) // 2
    samples_per_class = len(next(iter(selected.values())))

    figure = plt.figure(figsize=(14, rows * 2.75 + 0.7), constrained_layout=True)
    figure.patch.set_facecolor("white")
    outer_grid = figure.add_gridspec(rows, 2, wspace=0.08, hspace=0.12)

    for class_id, class_name in enumerate(dataset.classes):
        row, column = divmod(class_id, 2)
        class_grid = outer_grid[row, column].subgridspec(
            2,
            samples_per_class,
            height_ratios=(0.18, 1.0),
            hspace=0.02,
            wspace=0.04,
        )
        label_axis = figure.add_subplot(class_grid[0, :])
        label_axis.axis("off")
        label_pt = CLASS_LABELS_PT.get(class_name, class_name)
        label_axis.text(
            0.5,
            0.45,
            f"{label_pt} ({class_name})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

        for sample_position, dataset_index in enumerate(selected[class_id]):
            image, label = dataset[dataset_index]
            if int(label) != class_id:
                raise RuntimeError("O rótulo da imagem não corresponde à classe esperada.")
            image_axis = figure.add_subplot(class_grid[1, sample_position])
            image_axis.imshow(image, interpolation="nearest")
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            for spine in image_axis.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.6)

    partition_label = {"train": "treino", "val": "validação"}[partition]
    figure.suptitle(
        f"Amostras do EuroSAT — partição de {partition_label}",
        fontsize=15,
        fontweight="bold",
        color="#222222",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path, pdf_path


def create_class_panels(
    dataset,
    selected: dict[int, list[int]],
    partition: str,
    output_dir: Path,
) -> dict[int, tuple[Path, Path]]:
    """Cria uma figura horizontal com a legenda e os exemplos de cada classe."""
    panels_dir = output_dir / "class_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    partition_label = {"train": "treino", "val": "validação"}[partition]
    generated: dict[int, tuple[Path, Path]] = {}

    for class_id, class_name in enumerate(dataset.classes):
        dataset_indices = selected[class_id]
        figure, axes_grid = plt.subplots(
            1,
            len(dataset_indices),
            figsize=(3.1 * len(dataset_indices), 3.5),
            squeeze=False,
            constrained_layout=True,
        )
        figure.patch.set_facecolor("white")
        label_pt = CLASS_LABELS_PT.get(class_name, class_name)
        figure.suptitle(
            f"Classe: {label_pt} ({class_name})",
            fontsize=14,
            fontweight="bold",
            color="#222222",
        )
        for image_axis, dataset_index in zip(axes_grid[0], dataset_indices):
            image, label = dataset[dataset_index]
            if int(label) != class_id:
                raise RuntimeError("O rótulo da imagem não corresponde à classe esperada.")
            image_axis.imshow(image, interpolation="nearest")
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            for spine in image_axis.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.7)

        figure.supxlabel(
            f"Três amostras da partição de {partition_label} do EuroSAT",
            fontsize=9,
            color="#444444",
        )
        png_path = panels_dir / f"{class_name.lower()}_examples.png"
        pdf_path = png_path.with_suffix(".pdf")
        figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        generated[class_id] = (png_path, pdf_path)
    return generated


def save_manifest(
    dataset,
    selected: dict[int, list[int]],
    exported: dict[int, list[Path]],
    class_panels: dict[int, tuple[Path, Path]],
    data_dir: Path,
    output_dir: Path,
    partition: str,
    split_seed: int,
    selection_seed: int,
    split_path: Path,
) -> Path:
    """Registra a seleção para auditoria e reprodução da figura."""
    classes = []
    for class_id, class_name in enumerate(dataset.classes):
        samples = []
        for dataset_index, exported_path in zip(
            selected[class_id], exported[class_id]
        ):
            source_path = dataset.samples[dataset_index][0]
            samples.append(
                {
                    "dataset_index": dataset_index,
                    "source_path_relative_to_data_dir": portable_source_path(
                        source_path, data_dir
                    ),
                    "exported_path_relative_to_output_dir": exported_path.relative_to(
                        output_dir
                    ).as_posix(),
                }
            )
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "class_name_pt": CLASS_LABELS_PT.get(class_name, class_name),
                "panel_png_relative_to_output_dir": class_panels[class_id][
                    0
                ].relative_to(output_dir).as_posix(),
                "panel_pdf_relative_to_output_dir": class_panels[class_id][
                    1
                ].relative_to(output_dir).as_posix(),
                "samples": samples,
            }
        )

    payload = {
        "dataset": "EuroSAT",
        "partition": partition,
        "split_seed": split_seed,
        "selection_seed": selection_seed,
        "split_file": str(split_path.resolve()),
        "samples_per_class": len(next(iter(selected.values()))),
        "total_samples": sum(len(indices) for indices in selected.values()),
        "classes": classes,
    }
    manifest_path = output_dir / f"eurosat_{partition}_samples_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona e apresenta amostras reproduzíveis do EuroSAT"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/dataset_samples")
    )
    parser.add_argument("--partition", choices=("train", "val"), default="train")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Permite baixar o EuroSAT caso ele ainda não esteja disponível.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = datasets.EuroSAT(root=args.data_dir, download=args.download)
    split_indices, split_path = load_or_create_split(
        dataset,
        args.data_dir,
        args.split_seed,
        args.split_file,
    )
    selected = select_indices_by_class(
        dataset,
        split_indices[args.partition],
        args.samples_per_class,
        args.selection_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = export_selected_images(dataset, selected, args.output_dir)
    panel_path, pdf_path = create_sample_panel(
        dataset,
        selected,
        args.partition,
        args.output_dir / f"eurosat_{args.partition}_samples.png",
    )
    class_panels = create_class_panels(
        dataset,
        selected,
        args.partition,
        args.output_dir,
    )
    manifest_path = save_manifest(
        dataset,
        selected,
        exported,
        class_panels,
        args.data_dir,
        args.output_dir,
        args.partition,
        args.split_seed,
        args.selection_seed,
        split_path,
    )

    print(f"Partição: {args.partition}")
    print(f"Amostras: {sum(len(indices) for indices in selected.values())}")
    print(f"Painel PNG: {panel_path.resolve()}")
    print(f"Painel PDF: {pdf_path.resolve()}")
    print(f"Manifesto: {manifest_path.resolve()}")
    print(f"Imagens individuais: {(args.output_dir / 'selected_images').resolve()}")
    print(f"Painéis por classe: {(args.output_dir / 'class_panels').resolve()}")


if __name__ == "__main__":
    main()
