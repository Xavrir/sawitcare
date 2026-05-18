from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

from src.utils.draw_utils import draw_prediction
from src.utils.image_utils import crop_with_padding

CLASS_NAMES = ["healthy", "suspicious"]


@dataclass
class TreePrediction:
    tree_id: int
    box: tuple[int, int, int, int]
    detector_confidence: float
    health_label: str
    classifier_confidence: float
    crop: np.ndarray | None = None


def resolve_device(device: str | None = None) -> torch.device:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if device and device not in {"cuda", "0"}:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detector(path: Path) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(f"Detector model not found: {path}")
    return YOLO(str(path))


def build_classifier(num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_classifier(path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Classifier model not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint.get("classes", CLASS_NAMES) if isinstance(checkpoint, dict) else CLASS_NAMES
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_classifier(len(classes)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def classifier_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def detect_boxes(detector: YOLO, image: np.ndarray, conf: float = 0.25) -> list[tuple[tuple[int, int, int, int], float]]:
    result = detector.predict(source=image, conf=conf, verbose=False)[0]
    boxes = []
    if result.boxes is None:
        return boxes
    for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        boxes.append(((x1, y1, x2, y2), float(score)))
    return boxes


def classify_crop(model: nn.Module, classes: list[str], crop_bgr: np.ndarray, device: torch.device) -> tuple[str, float]:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = classifier_transform()(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return classes[idx], float(probs[idx].item())


def run_image_pipeline(image: np.ndarray, detector: YOLO, classifier: nn.Module, classes: list[str], device: torch.device, conf: float = 0.25, padding: float = 0.2, keep_crops: bool = False) -> tuple[np.ndarray, list[TreePrediction]]:
    annotated = image.copy()
    predictions: list[TreePrediction] = []
    for tree_id, (box, det_conf) in enumerate(detect_boxes(detector, image, conf), start=1):
        crop, padded_box = crop_with_padding(image, box, padding)
        label, cls_conf = classify_crop(classifier, classes, crop, device)
        draw_prediction(annotated, padded_box, label, cls_conf, det_conf)
        predictions.append(TreePrediction(tree_id, padded_box, det_conf, label, cls_conf, crop if keep_crops else None))
    return annotated, predictions
