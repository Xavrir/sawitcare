from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.image_utils import ensure_dir, find_images

LABEL_MAP = {"PalmSan": "healthy", "PalmAnom": "suspicious"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split classification images into healthy/suspicious train/val/test folders.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/classification"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/classification"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-padding", type=float, default=0.05, help="Padding ratio for annotation crops.")
    return parser.parse_args()


def infer_label(path: Path) -> str | None:
    parts = set(path.parts)
    for raw_label, mapped in LABEL_MAP.items():
        if raw_label in parts or raw_label.lower() in path.name.lower():
            return mapped
    return None


def stratified_three_way_split(items: list, labels: list[str], seed: int):
    grouped: dict[str, list] = defaultdict(list)
    for item, label in zip(items, labels):
        grouped[label].append(item)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    for label, group in grouped.items():
        rng.shuffle(group)
        n = len(group)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)
        for split, split_items in {"train": group[:train_end], "val": group[train_end:val_end], "test": group[val_end:]}.items():
            splits[split].extend(split_items)
            split_labels[split].extend([label] * len(split_items))
    for split in splits:
        combined = list(zip(splits[split], split_labels[split]))
        rng.shuffle(combined)
        if combined:
            splits[split], split_labels[split] = map(list, zip(*combined))
    return splits, split_labels


def find_annotation_csvs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("_annotations.csv") if p.is_file())


def collect_annotated_rows(raw_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    labels: list[str] = []
    for csv_path in find_annotation_csvs(raw_dir):
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                mapped = LABEL_MAP.get(row.get("class", ""))
                if mapped is None:
                    continue
                image_path = csv_path.parent / row["filename"]
                if not image_path.exists():
                    print(f"Skipping missing annotated image: {image_path}")
                    continue
                row = {**row, "image_path": str(image_path), "mapped_label": mapped}
                rows.append(row)
                labels.append(mapped)
    return rows, labels


def crop_annotation(row: dict[str, str], output_path: Path, padding: float) -> bool:
    image = cv2.imread(row["image_path"])
    if image is None:
        print(f"Skipping unreadable image: {row['image_path']}")
        return False
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(float(row[k])) for k in ["xmin", "ymin", "xmax", "ymax"])
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        print(f"Skipping invalid annotation box: {row}")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image[y1:y2, x1:x2]))


def write_annotated_split(rows: list[dict[str, str]], labels: list[str], output_dir: Path, seed: int, padding: float) -> None:
    splits, split_labels_by_name = stratified_three_way_split(rows, labels, seed)
    for split, split_rows in splits.items():
        written = 0
        for idx, row in enumerate(split_rows, start=1):
            label = row["mapped_label"]
            source = Path(row["image_path"])
            out = output_dir / split / label / f"{source.stem}_{idx:05d}.jpg"
            if crop_annotation(row, out, padding):
                written += 1
        print(f"{split}: {written} annotated crops")


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw classification directory not found: {args.raw_dir}")

    annotated_rows, annotated_labels = collect_annotated_rows(args.raw_dir)
    if annotated_rows:
        write_annotated_split(annotated_rows, annotated_labels, args.output_dir, args.seed, args.crop_padding)
        return

    images, labels = [], []
    for image in find_images(args.raw_dir):
        label = infer_label(image)
        if label is None:
            print(f"Skipping image with unknown label: {image}")
            continue
        images.append(image)
        labels.append(label)
    if not images:
        raise ValueError(f"No classification images found in {args.raw_dir}")
    splits, split_labels_by_name = stratified_three_way_split(images, labels, args.seed)
    for split, split_images in splits.items():
        split_labels = split_labels_by_name[split]
        for image, label in zip(split_images, split_labels):
            out_dir = ensure_dir(args.output_dir / split / label)
            shutil.copy2(image, out_dir / image.name)
        print(f"{split}: {len(split_images)} images")


if __name__ == "__main__":
    main()
