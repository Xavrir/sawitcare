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
from src.utils.draw_utils import draw_prediction, draw_summary_box


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def optional_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
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
    parser = argparse.ArgumentParser(description="Run SawitCare video inference.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--frame_step", type=positive_int, default=15)
    parser.add_argument("--start-frame", type=nonnegative_int, default=0, help="First frame to process/write.")
    parser.add_argument("--max-frames", type=optional_positive_int, default=None, help="Maximum number of frames to write.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--tile-size", type=nonnegative_int, default=0, help="Enable tiled detection with this tile size, e.g. 640. Use 0 to disable.")
    parser.add_argument("--tile-overlap", type=overlap_float, default=0.2, help="Fractional overlap between tiles when tiled detection is enabled.")
    parser.add_argument("--nms-iou", type=iou_float, default=0.5, help="IoU threshold for merging duplicate tiled detections.")
    parser.add_argument("--classifier-conf", type=confidence_float, default=0.0, help="Show uncertain when classifier confidence is below this threshold.")
    parser.add_argument("--short-labels", action="store_true", help="Draw compact labels like H, S, and U.")
    parser.add_argument("--hide-detector-conf", action="store_true", help="Hide detector confidence in drawn labels.")
    parser.add_argument("--persist-annotations", action="store_true", help="Keep the last annotated frame visible between processed frames.")
    parser.add_argument("--summary-box", action="store_true", help="Draw a per-frame count summary in the top-left corner.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def count_predictions(predictions) -> tuple[int, int, int, int]:
    healthy = sum(p.health_label == "healthy" for p in predictions)
    suspicious = sum(p.health_label == "suspicious" for p in predictions)
    uncertain = sum(p.health_label == "uncertain" for p in predictions)
    return len(predictions), healthy, suspicious, uncertain


def draw_cached_predictions(frame, predictions, short_labels: bool, show_detector_conf: bool, summary_box: bool):
    annotated = frame.copy()
    for prediction in predictions:
        draw_prediction(
            annotated,
            prediction.box,
            prediction.health_label,
            prediction.classifier_confidence,
            prediction.detector_confidence,
            short_labels,
            show_detector_conf,
        )
    if summary_box:
        total, healthy, suspicious, uncertain = count_predictions(predictions)
        draw_summary_box(annotated, total, healthy, suspicious, uncertain)
    return annotated


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
    detail_rows = []
    frame_id = 0
    last_annotated = None
    last_predictions = []
    written_frames = 0
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        frame_id = args.start_frame
    while True:
        if args.max_frames is not None and written_frames >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % args.frame_step == 0:
            annotated, preds = run_image_pipeline(
                frame,
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
            if args.summary_box:
                total, healthy, suspicious, uncertain = count_predictions(preds)
                draw_summary_box(annotated, total, healthy, suspicious, uncertain)
            last_annotated = annotated
            last_predictions = preds
            total, healthy, suspicious, uncertain = count_predictions(preds)
            rows.append({"frame_id": frame_id, "total_trees": total, "healthy": healthy, "suspicious": suspicious, "uncertain": uncertain, "suspicious_ratio": suspicious / total if total else 0.0})
            detail_rows.extend(
                {
                    "frame_id": frame_id,
                    "tree_id": p.tree_id,
                    "x1": p.box[0],
                    "y1": p.box[1],
                    "x2": p.box[2],
                    "y2": p.box[3],
                    "detector_confidence": p.detector_confidence,
                    "health_label": p.health_label,
                    "classifier_confidence": p.classifier_confidence,
                }
                for p in preds
            )
        if frame_id % args.frame_step == 0 and last_annotated is not None:
            output_frame = last_annotated
        elif args.persist_annotations and last_predictions:
            output_frame = draw_cached_predictions(frame, last_predictions, args.short_labels, not args.hide_detector_conf, args.summary_box)
        else:
            output_frame = frame
        writer.write(output_frame)
        frame_id += 1
        written_frames += 1
    cap.release()
    writer.release()
    out_csv = Path("outputs/predictions") / f"{args.video.stem}_frame_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_detail_csv = Path("outputs/predictions") / f"{args.video.stem}_tree_predictions.csv"
    pd.DataFrame(detail_rows).to_csv(out_detail_csv, index=False)
    print(f"Saved annotated video: {out_video}")
    print(f"Saved predictions CSV: {out_csv}")
    print(f"Saved tree predictions CSV: {out_detail_csv}")


if __name__ == "__main__":
    main()
