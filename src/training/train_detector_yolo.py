from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SawitCare YOLO detector.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model name or path, e.g. yolo11n.pt or yolov8n.pt")
    parser.add_argument("--data", type=Path, default=Path("configs/detection_data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("models/detector"))
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if str(device).lower() in {"cpu", "none"}:
        return "cpu"
    if torch.cuda.is_available():
        return str(device)
    print("CUDA is not available. Falling back to CPU.")
    return "cpu"


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Detection data yaml not found: {args.data}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.train(data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=resolve_device(args.device), project="outputs/metrics/detector_runs", name=Path(args.model).stem, exist_ok=True)
    best_path = Path(getattr(model.trainer, "best", "")) if getattr(model, "trainer", None) else None
    if best_path and best_path.exists():
        final_path = args.output_dir / f"{Path(args.model).stem}_best.pt"
        shutil.copy2(best_path, final_path)
        print(f"Saved best detector: {final_path}")
    print("Final training metrics:")
    print(results)


if __name__ == "__main__":
    main()
