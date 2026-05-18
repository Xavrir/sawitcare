from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from src.utils.metrics_utils import classification_metrics, save_json

CLASS_NAMES = ["healthy", "suspicious"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet B0 palm health classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/classification"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("models/classifier/efficientnet_b0_best.pt"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_model() -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def transforms_for(split: str):
    if split == "train":
        return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_datasets(data_dir: Path):
    datasets_by_split = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing classification split: {split_dir}")
        ds = datasets.ImageFolder(split_dir, transform=transforms_for(split))
        if ds.classes != CLASS_NAMES:
            print(f"Warning: expected classes {CLASS_NAMES}, found {ds.classes}. Class order will follow dataset folders.")
        datasets_by_split[split] = ds
    return datasets_by_split


def class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor | None:
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes)).float()
    if len(set(counts.tolist())) <= 1:
        return None
    weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    return weights.to(device)


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, list[int], list[int]]:
    model.train(train)
    total_loss, y_true, y_pred = 0.0, [], []
    for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(1, len(loader.dataset)), y_true, y_pred


def set_trainable(model: nn.Module, fine_tune: bool) -> None:
    for p in model.parameters():
        p.requires_grad = fine_tune
    for p in model.classifier.parameters():
        p.requires_grad = True
    if fine_tune:
        for p in model.features[-2:].parameters():
            p.requires_grad = True


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ds = load_datasets(args.data)
    loaders = {split: DataLoader(dataset, batch_size=args.batch, shuffle=(split == "train"), num_workers=2) for split, dataset in ds.items()}
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(ds["train"], device))
    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        fine_tune = epoch > args.head_epochs
        set_trainable(model, fine_tune=fine_tune)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr if fine_tune else args.lr * 3)
        train_loss, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, y_true, y_pred = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        metrics = classification_metrics(y_true, y_pred, ds["val"].classes)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model_state": model.state_dict(), "classes": ds["train"].classes, "image_size": 224}, args.output)
            print(f"Saved best classifier: {args.output}")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    _, y_true, y_pred = run_epoch(model, loaders["test"], criterion, torch.optim.AdamW(model.parameters()), device, train=False)
    metrics = classification_metrics(y_true, y_pred, ds["test"].classes)
    print(metrics)
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred).tolist())
    save_json(Path("outputs/metrics/classifier_metrics.json"), metrics)


if __name__ == "__main__":
    main()
