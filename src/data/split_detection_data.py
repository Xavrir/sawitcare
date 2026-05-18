from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean_detection_labels import validate_yolo_parts
from src.utils.image_utils import ensure_dir, find_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split detection images and YOLO labels into train/val/test.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/detection"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/detection"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-to-single-class", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def label_for(image: Path, raw_dir: Path) -> Path | None:
    candidates = [raw_dir / "labels" / f"{image.stem}.txt", image.with_suffix(".txt")]
    if image.parent.name == "images":
        candidates.append(image.parent.parent / "labels" / f"{image.stem}.txt")
    if "images" in image.parts:
        idx = len(image.parts) - 1 - list(reversed(image.parts)).index("images")
        rel_after_images = Path(*image.parts[idx + 1 :]).with_suffix(".txt")
        candidates.append(raw_dir / "labels" / rel_after_images)
    return next((p for p in candidates if p.exists()), None)


def collect_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw detection directory not found: {raw_dir}")
    roots = [raw_dir / "images"] if (raw_dir / "images").exists() else [raw_dir]
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        for image in find_images(root):
            label = label_for(image, raw_dir)
            if label is None:
                print(f"Skipping image without label: {image}")
                continue
            if "labels" in image.parts:
                continue
            pairs.append((image, label))
    if not pairs:
        raise ValueError(f"No image/label pairs found in {raw_dir}")
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], seed: int) -> dict[str, list[tuple[Path, Path]]]:
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    return {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}


def clean_label_text(label: Path, merge_to_single_class: bool) -> str:
    rows: list[str] = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        ok, reason = validate_yolo_parts(parts)
        if not ok:
            print(f"Skipping broken label row {label}:{line_no} ({reason})")
            continue
        class_id = "0" if merge_to_single_class else parts[0]
        rows.append(" ".join([class_id, *parts[1:]]))
    return "\n".join(rows) + ("\n" if rows else "")


def copy_split(splits: dict[str, list[tuple[Path, Path]]], output_dir: Path, merge_to_single_class: bool) -> None:
    for split, pairs in splits.items():
        image_dir = ensure_dir(output_dir / "images" / split)
        label_dir = ensure_dir(output_dir / "labels" / split)
        for image, label in pairs:
            shutil.copy2(image, image_dir / image.name)
            (label_dir / f"{image.stem}.txt").write_text(clean_label_text(label, merge_to_single_class), encoding="utf-8")
        print(f"{split}: {len(pairs)} pairs")


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.raw_dir)
    copy_split(split_pairs(pairs, args.seed), args.output_dir, args.merge_to_single_class)


if __name__ == "__main__":
    main()
