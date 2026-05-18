from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.inference.pipeline import load_classifier, load_detector, resolve_device, run_image_pipeline
from src.utils.image_utils import read_image, write_image


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def overlap_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 0 and < 1")
    return parsed


def iou_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def confidence_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SawitCare image inference.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--tile-size", type=nonnegative_int, default=0, help="Enable tiled detection with this tile size, e.g. 640. Use 0 to disable.")
    parser.add_argument("--tile-overlap", type=overlap_float, default=0.2, help="Fractional overlap between tiles when tiled detection is enabled.")
    parser.add_argument("--nms-iou", type=iou_float, default=0.5, help="IoU threshold for merging duplicate tiled detections.")
    parser.add_argument("--classifier-conf", type=confidence_float, default=0.0, help="Show uncertain when classifier confidence is below this threshold.")
    parser.add_argument("--short-labels", action="store_true", help="Draw compact labels like H, S, and U.")
    parser.add_argument("--hide-detector-conf", action="store_true", help="Hide detector confidence in drawn labels.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    device = resolve_device(args.device)
    detector = load_detector(args.detector)
    classifier, classes = load_classifier(args.classifier, device)
    annotated, predictions = run_image_pipeline(
        image,
        detector,
        classifier,
        classes,
        device,
        args.conf,
        args.padding,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        nms_iou=args.nms_iou,
        classifier_conf=args.classifier_conf,
        short_labels=args.short_labels,
        show_detector_conf=not args.hide_detector_conf,
    )
    out_image = Path("outputs/images") / f"{args.image.stem}_annotated{args.image.suffix}"
    write_image(out_image, annotated)
    rows = [{"image_name": args.image.name, "tree_id": p.tree_id, "x1": p.box[0], "y1": p.box[1], "x2": p.box[2], "y2": p.box[3], "detector_confidence": p.detector_confidence, "health_label": p.health_label, "classifier_confidence": p.classifier_confidence} for p in predictions]
    out_csv = Path("outputs/predictions") / f"{args.image.stem}_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved annotated image: {out_image}")
    print(f"Saved predictions CSV: {out_csv}")


if __name__ == "__main__":
    main()
