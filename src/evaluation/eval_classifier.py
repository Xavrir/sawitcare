from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.inference.pipeline import build_classifier, classifier_transform, resolve_device
from src.utils.metrics_utils import classification_metrics, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SawitCare EfficientNet B0 classifier.")
    parser.add_argument("--model", type=Path, default=Path("models/classifier/efficientnet_b0_best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/classification"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/classifier_eval_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Classifier model not found: {args.model}")
    test_dir = args.data / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test split not found: {test_dir}")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    classes = checkpoint.get("classes", ["healthy", "suspicious"])
    model = build_classifier(len(classes)).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    dataset = datasets.ImageFolder(test_dir, transform=classifier_transform())
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=2)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating classifier"):
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, dataset.classes)
    if "suspicious" in dataset.classes:
        metrics["suspicious_recall"] = metrics["per_class"]["suspicious"]["recall"]
    print(metrics)
    save_json(args.output, metrics)


if __name__ == "__main__":
    main()
