from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics_utils import save_json

DEFAULT_CANDIDATES = ("yolov8n=yolov8n.pt", "yolo11n=yolo11n.pt", "yolo26=yolo26n.pt")


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        source = value
        return Path(source).stem.replace("_best", ""), source
    name, source = value.split("=", 1)
    name = name.strip()
    source = source.strip()
    if not name or not source:
        raise argparse.ArgumentTypeError("candidate must be NAME=MODEL_OR_WEIGHT")
    return name, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate SawitCare detector candidates on the same test split.")
    parser.add_argument("--candidate", action="append", default=None, help="Detector candidate as NAME=MODEL_OR_WEIGHT. May be repeated.")
    parser.add_argument("--data", type=Path, default=Path("configs/detection_data.yaml"))
    parser.add_argument("--weights-dir", type=Path, default=Path("models/detector"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-missing", action="store_true", help="Fine-tune candidates whose best checkpoint is missing.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/metrics/detector_comparison.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/detector_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/metrics/detector_comparison.md"))
    return parser.parse_args()


def resolve_device(device: str | None) -> str | None:
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"
    if device.lower() in {"cpu", "none"}:
        return "cpu"
    if torch.cuda.is_available():
        return device
    print("CUDA is not available. Falling back to CPU.")
    return "cpu"


def expected_checkpoint(source: str, weights_dir: Path) -> Path:
    source_path = Path(source)
    if source_path.exists() and source_path.stem.endswith("_best"):
        return source_path
    return weights_dir / f"{source_path.stem}_best.pt"


def train_candidate(name: str, source: str, data: Path, args: argparse.Namespace, device: str | None) -> Path:
    model = YOLO(source)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project="outputs/metrics/detector_runs",
        name=name,
        exist_ok=True,
    )
    best_path = Path(getattr(model.trainer, "best", "")) if getattr(model, "trainer", None) else None
    if not best_path or not best_path.exists():
        raise RuntimeError(f"Training finished for {name}, but best checkpoint was not found.")
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.weights_dir / f"{Path(source).stem}_best.pt"
    shutil.copy2(best_path, final_path)
    return final_path


def model_stats(model: YOLO, checkpoint: Path) -> dict[str, float]:
    params_m = None
    if getattr(model, "model", None) is not None:
        params_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1_000_000
    return {
        "params_m": round(float(params_m), 3) if params_m is not None else None,
        "weight_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
    }


def evaluate_candidate(name: str, checkpoint: Path, data: Path, args: argparse.Namespace, device: str | None) -> dict[str, Any]:
    model = YOLO(str(checkpoint))
    results = model.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        device=device,
        project="outputs/metrics/detector_comparison_runs",
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    box = results.box
    speed = getattr(results, "speed", {}) or {}
    return {
        "name": name,
        "status": "evaluated",
        "checkpoint": str(checkpoint),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "speed_preprocess_ms": speed.get("preprocess"),
        "speed_inference_ms": speed.get("inference"),
        "speed_postprocess_ms": speed.get("postprocess"),
        **model_stats(model, checkpoint),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "status",
        "checkpoint",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "speed_inference_ms",
        "speed_preprocess_ms",
        "speed_postprocess_ms",
        "params_m",
        "weight_mb",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("name", "status", "precision", "recall", "mAP50", "mAP50_95", "speed_inference_ms", "params_m", "weight_mb")
    lines = [
        "# SawitCare Detector Comparison",
        "",
        "| Model | Status | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Params M | Weight MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_metric(row.get(column)) for column in columns) + " |")
    lines.append("")
    lines.append("All evaluated rows use the same SawitCare detection test split from `configs/detection_data.yaml`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data yaml not found: {args.data}")
    device = resolve_device(args.device)
    candidates = [parse_candidate(value) for value in (args.candidate or DEFAULT_CANDIDATES)]
    rows: list[dict[str, Any]] = []
    for name, source in candidates:
        checkpoint = expected_checkpoint(source, args.weights_dir)
        if not checkpoint.exists():
            if not args.train_missing:
                rows.append(
                    {
                        "name": name,
                        "status": "missing_checkpoint",
                        "checkpoint": str(checkpoint),
                        "note": "Run with --train-missing or train this model before comparison.",
                    }
                )
                continue
            checkpoint = train_candidate(name, source, args.data, args, device)
        rows.append(evaluate_candidate(name, checkpoint, args.data, args, device))
    save_json(args.output_json, {"data": str(args.data), "imgsz": args.imgsz, "device": device, "models": rows})
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote comparison JSON: {args.output_json}")
    print(f"Wrote comparison CSV: {args.output_csv}")
    print(f"Wrote comparison table: {args.output_md}")


if __name__ == "__main__":
    main()
