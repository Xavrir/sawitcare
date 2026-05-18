from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd

from src.inference.pipeline import load_classifier, load_detector, resolve_device, run_image_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SawitCare video inference.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--frame_step", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    device = resolve_device(args.device)
    detector = load_detector(args.detector)
    classifier, classes = load_classifier(args.classifier, device)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path("outputs/videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{args.video.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rows = []
    frame_id = 0
    last_annotated = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % args.frame_step == 0:
            annotated, preds = run_image_pipeline(frame, detector, classifier, classes, device, args.conf, args.padding)
            last_annotated = annotated
            healthy = sum(p.health_label == "healthy" for p in preds)
            suspicious = sum(p.health_label == "suspicious" for p in preds)
            total = len(preds)
            rows.append({"frame_id": frame_id, "total_trees": total, "healthy": healthy, "suspicious": suspicious, "suspicious_ratio": suspicious / total if total else 0.0})
        writer.write(last_annotated if last_annotated is not None else frame)
        frame_id += 1
    cap.release()
    writer.release()
    out_csv = Path("outputs/predictions") / f"{args.video.stem}_frame_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved annotated video: {out_video}")
    print(f"Saved predictions CSV: {out_csv}")


if __name__ == "__main__":
    main()
