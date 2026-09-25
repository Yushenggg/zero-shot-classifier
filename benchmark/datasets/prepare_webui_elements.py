"""Export a UI-element classification set from the WebUI human element dataset.

`biglab/dreamstruct-human-UI-elements-5k` labels every UI element of a web/mobile
screenshot with an element *type* and a bounding box. This downloads the label
CSV plus only the screenshots it references, crops each box, and writes:

    <out>/data.csv     columns: text,label,image
    <out>/images/      one PNG per cropped element
    <out>/q.json       choice question, criteria = element keys

Only ``huggingface-hub`` (already a dependency) and Pillow are used. The option
keys are multi-token phrases ("Text Button", "Input Field", "Page Indicator"),
so this is a non-single-token probe for calibration behaviour.

Usage:
    .venv/bin/python benchmark/datasets/prepare_webui_elements.py
    .venv/bin/python benchmark/datasets/prepare_webui_elements.py --per-class 20
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "biglab/dreamstruct-human-UI-elements-5k"
DEFAULT_LABELS = "labels/test_ui_data.csv"
TEXT = "Look at the UI element in the image."

# CSV ``Type`` -> (criteria key, human-readable name). Keys are deliberately
# multi-token so they exercise non-single-token option scoring.
CLASSES: dict[str, tuple[str, str]] = {
    "text": ("text", "Text"),
    "icon": ("icon", "Icon"),
    "image": ("image", "Image"),
    "textbutton": ("text button", "Text Button"),
    "uppertaskbar": ("upper task bar", "Upper Task Bar"),
    "inputfield": ("input field", "Input Field"),
    "pageindicator": ("page indicator", "Page Indicator"),
    "checkedview": ("checked view", "Checked View"),
    "backgroundimage": ("background image", "Background Image"),
    "slidingmenu": ("sliding menu", "Sliding Menu"),
    "switch": ("switch", "Switch"),
}


def _to_int(value: str) -> int:
    return int(round(float(value)))


def export(
    out_dir: Path,
    *,
    labels_file: str,
    per_class: int,
    min_size: int,
    keep_min: int,
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    label_path = hf_hub_download(REPO, labels_file, repo_type="dataset")
    with Path(label_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[row["file_name"]].append(row)

    counts: Counter[str] = Counter()
    records: list[dict[str, str]] = []
    index = 0

    for file_name, boxes in by_file.items():
        if all(counts[cls] >= per_class for cls in CLASSES):
            break
        try:
            image_path = hf_hub_download(REPO, f"train/{file_name}", repo_type="dataset")
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - skip missing/unreadable screenshots
            print(f"  skip {file_name}: {exc}")
            continue
        width, height = image.size
        for row in boxes:
            element_type = row["Type"]
            spec = CLASSES.get(element_type)
            if spec is None or counts[element_type] >= per_class:
                continue
            left = _to_int(row["X"])
            top = _to_int(row["Y"])
            right = left + _to_int(row["BB Width"])
            bottom = top + _to_int(row["BB Height"])
            left, right = sorted((max(0, left), min(width, right)))
            top, bottom = sorted((max(0, top), min(height, bottom)))
            if right - left < min_size or bottom - top < min_size:
                continue
            key, _ = spec
            filename = f"{index:06d}_{key.replace(' ', '-')}.png"
            image.crop((left, top, right, bottom)).save(images_dir / filename)
            records.append({"text": TEXT, "label": key, "image": f"images/{filename}"})
            counts[element_type] += 1
            index += 1

    kept = [t for t in CLASSES if counts[t] >= keep_min]
    dropped = [t for t in CLASSES if counts[t] < keep_min]
    if len(kept) < 2:
        raise SystemExit(
            f"only {len(kept)} class(es) reached {keep_min} crops; lower --keep-min"
        )
    kept_keys = {CLASSES[t][0] for t in kept}
    records = [record for record in records if record["label"] in kept_keys]
    for element_type in dropped:
        key = CLASSES[element_type][0].replace(" ", "-")
        for orphan in images_dir.glob(f"*_{key}.png"):
            orphan.unlink()

    with (out_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "image"])
        writer.writeheader()
        writer.writerows(records)

    (out_dir / "q.json").write_text(
        json.dumps(
            {
                "class": {
                    "type": "choice",
                    "instructions": "What kind of UI element is shown in this image?",
                    "criteria": {CLASSES[t][0]: CLASSES[t][1] for t in kept},
                }
            },
            indent=2,
        )
    )

    print(f"\n{len(records)} crops across {len(kept)} classes -> {out_dir / 'data.csv'}")
    for element_type in kept:
        print(f"  {CLASSES[element_type][0]:18} {counts[element_type]:4}")
    if dropped:
        print("dropped (below --keep-min):")
        for element_type in dropped:
            print(f"  {CLASSES[element_type][0]:18} {counts[element_type]:4}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare-webui-elements",
        description="Export a UI-element classification set from the WebUI human element data.",
    )
    parser.add_argument("--out", "-o", default="benchmark/data/webui-elements")
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS,
        help="Label CSV within the dataset (default: the test split).",
    )
    parser.add_argument("--per-class", type=int, default=100, help="Crops per class target.")
    parser.add_argument("--min-size", type=int, default=8, help="Skip crops smaller than N px.")
    parser.add_argument(
        "--keep-min",
        type=int,
        default=10,
        help="Drop classes with fewer than N crops (default: 10).",
    )
    args = parser.parse_args(argv)

    if args.per_class <= 0:
        raise SystemExit("--per-class must be > 0")
    export(
        Path(args.out),
        labels_file=args.labels_file,
        per_class=args.per_class,
        min_size=args.min_size,
        keep_min=args.keep_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
