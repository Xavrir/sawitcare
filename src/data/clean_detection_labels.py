from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge YOLO detection labels into one oil_palm_tree class.")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/detection/labels"))
    parser.add_argument("--output-labels-dir", type=Path, default=Path("data/processed/detection/clean_labels"))
    parser.add_argument("--data-yaml-output", type=Path, default=Path("configs/detection_data.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/detection"))
    return parser.parse_args()


def validate_yolo_parts(parts: list[str]) -> tuple[bool, str]:
    if len(parts) != 5:
        return False, "expected 5 YOLO columns"
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return False, "non-numeric bbox value"
    x, y, w, h = values
    if any(v < 0 or v > 1 for v in values):
        return False, "bbox values must be in 0..1"
    if w <= 0 or h <= 0:
        return False, "bbox width/height must be positive"
    if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
        return False, "bbox extends outside image"
    return True, ""


def clean_labels(labels_dir: Path, output_dir: Path) -> dict[str, int]:
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "written": 0, "broken_files": 0, "broken_rows": 0}
    for label_path in tqdm(sorted(labels_dir.rglob("*.txt")), desc="Cleaning labels"):
        stats["files"] += 1
        rows: list[str] = []
        broken = False
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            ok, reason = validate_yolo_parts(parts)
            if not ok:
                print(f"Broken label skipped: {label_path}:{line_no} ({reason})")
                stats["broken_rows"] += 1
                broken = True
                continue
            rows.append("0 " + " ".join(parts[1:]))
        if broken:
            stats["broken_files"] += 1
        rel = label_path.relative_to(labels_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        stats["written"] += 1
    return stats


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"path": str(dataset_root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "oil_palm_tree"}}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stats = clean_labels(args.labels_dir, args.output_labels_dir)
    write_data_yaml(args.data_yaml_output, args.dataset_root)
    print(stats)
    print(f"Wrote data yaml: {args.data_yaml_output}")


if __name__ == "__main__":
    main()
