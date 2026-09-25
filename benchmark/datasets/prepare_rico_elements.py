"""Export a UI-element classification dataset from RICO's semantic annotations.

RICO annotates every element of a mobile screenshot with one of 25 design-semantic
component classes. This streams the ``ui-screenshots-and-hierarchies-with-
semantic-annotations`` config from the Hugging Face datasets-server, crops each
labelled element from its screenshot, and writes a balanced classification set:

    <out>/data.csv     columns: text,label,image
    <out>/images/      one PNG per cropped element
    <out>/q.json       choice question, criteria = the 25 component keys

No extra dependencies: stdlib + Pillow. The labels are multi-token (e.g.
``text-button``, ``bottom-navigation``), so this is a useful non-single-token
probe for calibration behaviour.

Usage:
    .venv/bin/python benchmark/datasets/prepare_rico_elements.py
    .venv/bin/python benchmark/datasets/prepare_rico_elements.py --per-class 20 --pages 10
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

DATASET = "creative-graphic-design/Rico"
CONFIG = "ui-screenshots-and-hierarchies-with-semantic-annotations"
ROWS_API = "https://datasets-server.huggingface.co/rows"

TEXT = "Look at the UI element in the image."

# Index -> (criteria key, human-readable name). Order matches the dataset's
# component_label ClassLabel.
CLASSES: list[tuple[str, str]] = [
    ("text", "Text"),
    ("image", "Image"),
    ("icon", "Icon"),
    ("text-button", "Text Button"),
    ("list-item", "List Item"),
    ("input", "Input"),
    ("background-image", "Background Image"),
    ("card", "Card"),
    ("web-view", "Web View"),
    ("radio-button", "Radio Button"),
    ("drawer", "Drawer"),
    ("checkbox", "Checkbox"),
    ("advertisement", "Advertisement"),
    ("modal", "Modal"),
    ("pager-indicator", "Pager Indicator"),
    ("slider", "Slider"),
    ("on-off-switch", "On/Off Switch"),
    ("button-bar", "Button Bar"),
    ("toolbar", "Toolbar"),
    ("number-stepper", "Number Stepper"),
    ("multi-tab", "Multi-Tab"),
    ("date-picker", "Date Picker"),
    ("map-view", "Map View"),
    ("video", "Video"),
    ("bottom-navigation", "Bottom Navigation"),
]


def fetch_rows(split: str, offset: int, length: int, *, retries: int = 5) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": split, "offset": offset, "length": length}
    )
    url = f"{ROWS_API}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)["rows"]
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                wait = int(exc.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
                print(f"  rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception:  # noqa: BLE001 - transient network errors; retry
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return []


def iter_elements(row: dict[str, Any]) -> Iterator[tuple[list[int], int]]:
    """Yield (bbox, class_index) for every labelled element in a screenshot row."""
    for child in row.get("children") or []:
        bounds = child.get("bounds") or []
        labels = child.get("component_label") or []
        for box, label in zip(bounds, labels, strict=False):
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box):
                yield box, int(label)


def clamp_box(box: list[int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def download_image(src: str) -> Image.Image:
    with urllib.request.urlopen(src, timeout=90) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def export(
    out_dir: Path,
    *,
    split: str,
    per_class: int,
    pages: int,
    length: int,
    min_size: int,
    keep_min: int,
    sleep: float,
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    records: list[dict[str, str]] = []
    seen = 0
    index = 0
    needed = set(range(len(CLASSES)))

    for page in range(pages):
        rows = fetch_rows(split, page * length, length)
        if not rows:
            break
        for item in rows:
            seen += 1
            row = item["row"]
            elements = list(iter_elements(row))
            wanted = [e for e in elements if counts[e[1]] < per_class]
            if not wanted:
                continue
            try:
                image = download_image(row["screenshot"]["src"])
            except Exception as exc:  # noqa: BLE001 - skip unreadable screenshots
                print(f"  skip row {item['row_idx']}: {exc}")
                continue
            width, height = image.size
            for box, label in wanted:
                if counts[label] >= per_class:
                    continue
                clamped = clamp_box(box, width, height)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue
                key, _ = CLASSES[label]
                filename = f"{index:06d}_{key}.png"
                image.crop(clamped).save(images_dir / filename)
                records.append({"text": TEXT, "label": key, "image": f"images/{filename}"})
                counts[label] += 1
                index += 1
        needed = {i for i in range(len(CLASSES)) if counts[i] < per_class}
        print(
            f"page {page + 1}/{pages}: rows={seen} crops={len(records)} "
            f"classes_done={len(CLASSES) - len(needed)}/{len(CLASSES)}"
        )
        if not needed:
            break
        if sleep:
            time.sleep(sleep)

    kept = [i for i in range(len(CLASSES)) if counts[i] >= keep_min]
    dropped = [i for i in range(len(CLASSES)) if counts[i] < keep_min]
    if len(kept) < 2:
        raise SystemExit(
            f"only {len(kept)} class(es) reached {keep_min} crops; "
            "lower --keep-min or --per-class, or scan more --pages"
        )

    kept_keys = {CLASSES[i][0] for i in kept}
    records = [record for record in records if record["label"] in kept_keys]
    for i in dropped:
        key = CLASSES[i][0]
        for orphan in images_dir.glob(f"*_{key}.png"):
            orphan.unlink()

    with (out_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "image"])
        writer.writeheader()
        writer.writerows(records)

    question = {
        "class": {
            "type": "choice",
            "instructions": "What kind of UI element is shown in this image?",
            "criteria": {CLASSES[i][0]: CLASSES[i][1] for i in kept},
        }
    }
    (out_dir / "q.json").write_text(json.dumps(question, indent=2))

    print(f"\n{len(records)} crops across {len(kept)} classes -> {out_dir / 'data.csv'}")
    for i in kept:
        print(f"  {CLASSES[i][0]:18} {counts[i]:4}")
    if dropped:
        print("dropped (below --keep-min):")
        for i in dropped:
            print(f"  {CLASSES[i][0]:18} {counts[i]:4}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare-rico-elements",
        description="Export a balanced RICO UI-element classification set.",
    )
    parser.add_argument("--out", "-o", default="benchmark/data/rico-elements")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--per-class", type=int, default=100, help="Crops per class target.")
    parser.add_argument("--pages", type=int, default=200, help="Max 100-row API pages to scan.")
    parser.add_argument("--length", type=int, default=100, help="Rows per API request.")
    parser.add_argument("--min-size", type=int, default=8, help="Skip crops smaller than N px.")
    parser.add_argument(
        "--keep-min",
        type=int,
        default=10,
        help="Drop classes with fewer than N crops from the final set (default: 10).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to pause between API pages (rate-limit courtesy; default: 1.0).",
    )
    args = parser.parse_args(argv)

    if args.per_class <= 0:
        raise SystemExit("--per-class must be > 0")
    export(
        Path(args.out),
        split=args.split,
        per_class=args.per_class,
        pages=args.pages,
        length=args.length,
        min_size=args.min_size,
        keep_min=args.keep_min,
        sleep=args.sleep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
