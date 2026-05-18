from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.inference.pipeline import load_classifier, load_detector, resolve_device, run_image_pipeline
from src.utils.image_utils import find_images, read_image, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SawitCare pipeline on sample images.")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--detector", type=Path, default=Path("models/detector/yolo11n_best.pt"))
    parser.add_argument("--classifier", type=Path, default=Path("models/classifier/efficientnet_b0_best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.samples.exists():
        raise FileNotFoundError(f"Samples directory not found: {args.samples}")
    device = resolve_device(args.device)
    detector = load_detector(args.detector)
    classifier, classes = load_classifier(args.classifier, device)
    rows = []
    crop_root = Path("outputs/crops")
    for image_path in find_images(args.samples):
        image = read_image(image_path)
        annotated, predictions = run_image_pipeline(image, detector, classifier, classes, device, args.conf, args.padding, keep_crops=True)
        write_image(Path("outputs/images") / f"{image_path.stem}_annotated{image_path.suffix}", annotated)
        for pred in predictions:
            if pred.crop is not None:
                write_image(crop_root / image_path.stem / f"tree_{pred.tree_id:04d}_{pred.health_label}.jpg", pred.crop)
            rows.append({"image_name": image_path.name, "tree_id": pred.tree_id, "x1": pred.box[0], "y1": pred.box[1], "x2": pred.box[2], "y2": pred.box[3], "detector_confidence": pred.detector_confidence, "health_label": pred.health_label, "classifier_confidence": pred.classifier_confidence})
    out_csv = Path("outputs/predictions/pipeline_sample_predictions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved sample pipeline CSV: {out_csv}")


if __name__ == "__main__":
    main()
