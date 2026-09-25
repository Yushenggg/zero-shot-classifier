"""Export image-classification datasets to CSV for zero-shot-bench.

Uses torchvision, which is already a project dependency: no extra packages and
no manual download. For each dataset it writes everything ``zero-shot-bench``
needs:

    <out>/<name>/data.csv     columns: text,label,image
    <out>/<name>/images/      one image file per example
    <out>/<name>/q.json       choice question whose criteria are the class keys

Datasets differ in their loader signature and split handling; a spec can set
``factory_kwargs`` (e.g. EuroSAT takes no ``train`` argument) and ``per_class``
for a balanced cap when there is no natural test split. ``class_map`` maps a
dataset's own folder class names onto the criteria keys/descriptions, so the
label column and question criteria stay in sync with the loader.

Usage:
    .venv/bin/python benchmark/datasets/prepare_image_classification.py
    .venv/bin/python benchmark/datasets/prepare_image_classification.py -d eurosat --per-class 20
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from torchvision import datasets

_DEFAULT_KWARGS: dict[str, Any] = {"train": False, "download": True}

# Per dataset: the torchvision class, constructor kwargs, the criteria keys (also
# the CSV label column), descriptions, prompt text/instructions, and optional
# per_class cap / class_map (dataset class name -> (key, description)).
SPECS: dict[str, dict[str, Any]] = {
    "mnist": {
        "factory": datasets.MNIST,
        "keys": [str(digit) for digit in range(10)],
        "descriptions": {},
        "text": "Look at the handwritten digit in the image.",
        "instructions": "Which single digit (0-9) is written in this image?",
    },
    "fashion_mnist": {
        "factory": datasets.FashionMNIST,
        "keys": [
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle-boot",
        ],
        "descriptions": {
            "t-shirt": "T-shirt/top",
            "trouser": "Trouser",
            "pullover": "Pullover",
            "dress": "Dress",
            "coat": "Coat",
            "sandal": "Sandal",
            "shirt": "Shirt",
            "sneaker": "Sneaker",
            "bag": "Bag",
            "ankle-boot": "Ankle boot",
        },
        "text": "Look at the image of clothing.",
        "instructions": "Which type of clothing is shown in the image?",
    },
    "kmnist": {
        "factory": datasets.KMNIST,
        "keys": ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
        "descriptions": {
            "o": "Hiragana O",
            "ki": "Hiragana KI",
            "su": "Hiragana SU",
            "tsu": "Hiragana TSU",
            "na": "Hiragana NA",
            "ha": "Hiragana HA",
            "ma": "Hiragana MA",
            "ya": "Hiragana YA",
            "re": "Hiragana RE",
            "wo": "Hiragana WO",
        },
        "text": "Look at the image of a handwritten Japanese character.",
        "instructions": "Which hiragana character is written in the image?",
    },
    "usps": {
        "factory": datasets.USPS,
        "keys": [str(digit) for digit in range(10)],
        "descriptions": {},
        "text": "Look at the image of a handwritten digit.",
        "instructions": "Which single digit (0-9) is written in the image?",
    },
    "eurosat": {
        "factory": datasets.EuroSAT,
        "factory_kwargs": {"download": True},
        "per_class": 200,
        "class_map": {
            "AnnualCrop": (
                "annual crop land",
                "Fields planted with annual crops such as wheat, corn or soy",
            ),
            "Forest": ("forest", "Dense tree cover"),
            "HerbaceousVegetation": (
                "herbaceous vegetation",
                "Grassland, shrubs or low vegetation",
            ),
            "Highway": ("highway or road", "A road, highway or paved transport corridor"),
            "Industrial": (
                "industrial buildings",
                "Factories, warehouses or industrial facilities",
            ),
            "Pasture": ("pasture land", "Grazing land or managed grassland"),
            "PermanentCrop": (
                "permanent crop land",
                "Orchards, vineyards or other permanent crops",
            ),
            "Residential": (
                "residential buildings",
                "Houses or residential neighbourhoods",
            ),
            "River": ("river", "A river or watercourse"),
            "SeaLake": ("sea or lake", "Open water such as a sea, lake or ocean"),
        },
        "text": "Look at the satellite image.",
        "instructions": "What kind of terrain or land use is shown in this satellite image?",
    },
}


def resolve_classes(
    name: str, spec: dict[str, Any], dataset: Any
) -> tuple[list[str], dict[str, str]]:
    """Return ``(keys, descriptions)`` in class-index order for the dataset."""
    class_map = spec.get("class_map")
    if class_map is None:
        return list(spec["keys"]), dict(spec["descriptions"])

    classes = getattr(dataset, "classes", None)
    if not classes:
        raise SystemExit(f"{name}: dataset exposes no .classes, cannot apply class_map")
    missing = [cls for cls in classes if cls not in class_map]
    if missing:
        raise SystemExit(f"{name}: classes missing from class_map: {missing}")
    ordered = [class_map[cls] for cls in classes]
    return [key for key, _ in ordered], {key: desc for key, desc in ordered}


def export(
    name: str,
    spec: dict[str, Any],
    out_dir: Path,
    limit: int | None,
    per_class: int | None,
) -> int:
    kwargs = spec.get("factory_kwargs", _DEFAULT_KWARGS)
    dataset = spec["factory"](root=str(out_dir / "_raw"), **kwargs)
    keys, descriptions = resolve_classes(name, spec, dataset)
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{name}: duplicate criteria keys")

    cap = per_class if per_class is not None else spec.get("per_class")

    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    records: list[dict[str, str]] = []
    for index, (image, target) in enumerate(dataset):
        if limit is not None and len(records) >= limit:
            break
        if cap is not None:
            if all(counts[i] >= cap for i in range(len(keys))):
                break
            if counts[target] >= cap:
                continue
        key = keys[target]
        filename = f"{index:05d}_{key.replace(' ', '-')}.png"
        image.save(images / filename)
        records.append(
            {"text": spec["text"], "label": key, "image": f"images/{filename}"}
        )
        counts[target] += 1

    with (out_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "image"])
        writer.writeheader()
        writer.writerows(records)

    question = {
        "class": {
            "type": "choice",
            "instructions": spec["instructions"],
            "criteria": {key: descriptions.get(key, "") for key in keys},
        }
    }
    (out_dir / "q.json").write_text(json.dumps(question, indent=2))
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare-image-classification",
        description="Export image-classification datasets to CSV.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        choices=sorted(SPECS),
        default=None,
        help="Dataset(s) to export (repeatable). Default: all of " + ", ".join(SPECS),
    )
    parser.add_argument(
        "--out",
        "-o",
        default="benchmark/data/popular",
        help="Output root (default: benchmark/data/popular).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only the first N rows per dataset (default: all).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="Override the per-class cap (balanced sampling where the spec sets one).",
    )
    args = parser.parse_args(argv)

    names = args.dataset or list(SPECS)
    for name in names:
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        count = export(name, SPECS[name], out_dir, args.limit, args.per_class)
        print(f"{name}: {count} rows -> {out_dir / 'data.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
