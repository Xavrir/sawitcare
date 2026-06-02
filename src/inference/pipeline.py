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


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _valid_box(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def _box_iou(box: tuple[int, int, int, int], boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-6)


def _nms_boxes(
    boxes: list[tuple[tuple[int, int, int, int], float]],
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if not boxes:
        return []
    coords = np.array([box for box, _ in boxes], dtype=np.float32)
    scores = np.array([score for _, score in boxes], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = _box_iou(tuple(coords[current].astype(int)), coords[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return [boxes[index] for index in keep]


def detect_boxes_tiled(
    detector: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    tile_size: int = 640,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if tile_size <= 0:
        return detect_boxes(detector, image, conf)
    height, width = image.shape[:2]
    tile_size = max(1, tile_size)
    overlap = min(max(overlap, 0.0), 0.9)
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            for (x1, y1, x2, y2), score in detect_boxes(detector, tile, conf):
                full_box = _clip_box((x1 + x, y1 + y, x2 + x, y2 + y), width, height)
                if _valid_box(full_box):
                    boxes.append((full_box, score))
    return _nms_boxes(boxes, nms_iou)


def classify_crop(model: nn.Module, classes: list[str], crop_bgr: np.ndarray, device: torch.device) -> tuple[str, float]:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = classifier_transform()(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return classes[idx], float(probs[idx].item())


def run_image_pipeline(
    image: np.ndarray,
    detector: YOLO,
    classifier: nn.Module,
    classes: list[str],
    device: torch.device,
    conf: float = 0.25,
    padding: float = 0.2,
    keep_crops: bool = False,
    tile_size: int = 0,
    tile_overlap: float = 0.2,
    nms_iou: float = 0.5,
    classifier_conf: float = 0.0,
    short_labels: bool = False,
    show_classifier_conf: bool = True,
    show_detector_conf: bool = True,
    display_mode: str = "all",
) -> tuple[np.ndarray, list[TreePrediction]]:
    annotated = image.copy()
    predictions: list[TreePrediction] = []
    if tile_size > 0:
        boxes = detect_boxes_tiled(detector, image, conf, tile_size, tile_overlap, nms_iou)
    else:
        boxes = detect_boxes(detector, image, conf)
    for tree_id, (box, det_conf) in enumerate(boxes, start=1):
        crop, padded_box = crop_with_padding(image, box, padding)
        label, cls_conf = classify_crop(classifier, classes, crop, device)
        display_label = label if cls_conf >= classifier_conf else "uncertain"
        if display_mode == "all" or display_label == display_mode:
            draw_prediction(annotated, padded_box, display_label, cls_conf, det_conf, short_labels, show_classifier_conf, show_detector_conf)
        predictions.append(TreePrediction(tree_id, padded_box, det_conf, display_label, cls_conf, crop if keep_crops else None))
    return annotated, predictions
