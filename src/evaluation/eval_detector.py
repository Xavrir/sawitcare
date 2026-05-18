from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.utils.metrics_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SawitCare YOLO detector on test split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/detection_data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/detector_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Detector model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data yaml not found: {args.data}")
    model = YOLO(str(args.model))
    results = model.val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device)
    box = results.box
    metrics = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "speed": getattr(results, "speed", None),
    }
    print(metrics)
    save_json(args.output, metrics)


if __name__ == "__main__":
    main()
