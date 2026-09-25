"""Export full test splits of popular image-classification datasets to CSV.

Uses torchvision, which is already a project dependency: no extra packages and
no manual download. For each dataset it writes everything ``zero-shot-bench``
needs:

    <out>/<name>/data.csv     columns: text,label,image
    <out>/<name>/images/      one image file per test example
    <out>/<name>/q.json       choice question whose criteria are the class keys

The test split is used. MNIST/Fashion-MNIST/KMNIST have 10,000 test images;
USPS has ~2,000. The ``label`` column and the question criteria keys are
identical, so no ``--label-map`` is required.

Usage:
    .venv/bin/python benchmark/datasets/prepare_image_classification.py
    .venv/bin/python benchmark/datasets/prepare_image_classification.py -d mnist --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from torchvision import datasets

# Per dataset: the torchvision class, the criteria keys in class-index order
# (also the CSV label column), optional per-key descriptions, and the prompt
# text/instructions passed to the classifier.
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
}


def export(name: str, spec: dict[str, Any], out_dir: Path, limit: int | None) -> int:
    keys = spec["keys"]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{name}: duplicate criteria keys")

    dataset = spec["factory"](root=str(out_dir / "_raw"), train=False, download=True)

    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for index, (image, target) in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        key = keys[target]
        filename = f"{index:05d}_{key}.png"
        image.save(images / filename)
        records.append(
            {"text": spec["text"], "label": key, "image": f"images/{filename}"}
        )

    with (out_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "image"])
        writer.writeheader()
        writer.writerows(records)

    descriptions = spec["descriptions"]
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
        description="Export full test splits of popular image datasets to CSV.",
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
        help="Export only the first N rows per dataset (default: full test split).",
    )
    args = parser.parse_args(argv)

    names = args.dataset or list(SPECS)
    for name in names:
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        count = export(name, SPECS[name], out_dir, args.limit)
        print(f"{name}: {count} rows -> {out_dir / 'data.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
