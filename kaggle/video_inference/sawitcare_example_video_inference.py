from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("KAGGLE_WORK_ROOT", "/kaggle/working/sawitcare"))
VIDEO_NAME = os.environ.get("VIDEO_NAME", "road_rainforest_oil_palm_indonesia.mp4")
DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "yolo11n_best.pt")
CLASSIFIER_NAME = os.environ.get("CLASSIFIER_NAME", "efficientnet_b0_best.pt")
FRAME_STEP = int(os.environ.get("FRAME_STEP", "15"))
START_FRAME = int(os.environ.get("START_FRAME", "0"))
MAX_FRAMES = os.environ.get("MAX_FRAMES", "300")
MAX_FRAMES_INT = int(MAX_FRAMES) if MAX_FRAMES else None
CONF = float(os.environ.get("CONF", "0.45"))
CLASSIFIER_CONF = float(os.environ.get("CLASSIFIER_CONF", "0.70"))
PADDING = float(os.environ.get("PADDING", "0.2"))
TILE_SIZE = int(os.environ.get("TILE_SIZE", "640"))
TILE_OVERLAP = float(os.environ.get("TILE_OVERLAP", "0.2"))
NMS_IOU = float(os.environ.get("NMS_IOU", "0.35"))
MIN_BOX_WIDTH = int(os.environ.get("MIN_BOX_WIDTH", "28"))
MIN_BOX_HEIGHT = int(os.environ.get("MIN_BOX_HEIGHT", "28"))
MIN_BOX_AREA = int(os.environ.get("MIN_BOX_AREA", "1200"))
SHORT_LABELS = os.environ.get("SHORT_LABELS", "1") != "0"
SUMMARY_BOX = os.environ.get("SUMMARY_BOX", "1") != "0"
PERSIST_ANNOTATIONS = os.environ.get("PERSIST_ANNOTATIONS", "1") != "0"
SHOW_CLASSIFIER_CONF = os.environ.get("SHOW_CLASSIFIER_CONF", "0") != "0"
SHOW_DETECTOR_CONF = os.environ.get("SHOW_DETECTOR_CONF", "0") != "0"


def find_file(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {INPUT_ROOT}")
    return matches[0]


def install_offline_wheels() -> None:
    try:
        import ultralytics  # noqa: F401

        return
    except Exception:
        pass

    wheels = [
        *sorted(INPUT_ROOT.rglob("ultralytics_thop-*.whl")),
        *sorted(INPUT_ROOT.rglob("ultralytics-*.whl")),
    ]
    if not wheels:
        raise RuntimeError("Ultralytics is not installed and no offline wheels were found.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *map(str, wheels)], check=True)


install_offline_wheels()

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


CLASS_NAMES = ["healthy", "suspicious"]
COLORS = {
    "healthy": (48, 180, 90),
    "suspicious": (40, 120, 255),
    "uncertain": (160, 160, 160),
}
SHORT_LABEL_MAP = {
    "healthy": "H",
    "suspicious": "S",
    "uncertain": "U",
}


@dataclass
class TreePrediction:
    tree_id: int
    box: tuple[int, int, int, int]
    detector_confidence: float
    health_label: str
    classifier_confidence: float


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yolo_device(device: torch.device) -> str | int:
    return 0 if device.type == "cuda" else "cpu"


def build_classifier(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_classifier(path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", CLASS_NAMES) if isinstance(checkpoint, dict) else CLASS_NAMES
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_classifier(len(classes)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def classifier_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def crop_with_padding(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    padding: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(width, x2 + pad_x)
    py2 = min(height, y2 + pad_y)
    return image[py1:py2, px1:px2], (px1, py1, px2, py2)


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
    width = x2 - x1
    height = y2 - y1
    return width >= MIN_BOX_WIDTH and height >= MIN_BOX_HEIGHT and width * height >= MIN_BOX_AREA


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


def detect_boxes(
    detector: YOLO,
    image: np.ndarray,
    device: torch.device,
    conf: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    result = detector.predict(source=image, conf=conf, verbose=False, device=yolo_device(device))[0]
    boxes = []
    if result.boxes is None:
        return boxes
    for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        boxes.append(((x1, y1, x2, y2), float(score)))
    return boxes


def detect_boxes_tiled(
    detector: YOLO,
    image: np.ndarray,
    device: torch.device,
    conf: float,
    tile_size: int,
    overlap: float,
    nms_iou: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if tile_size <= 0:
        return detect_boxes(detector, image, device, conf)
    height, width = image.shape[:2]
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            for (x1, y1, x2, y2), score in detect_boxes(detector, tile, device, conf):
                full_box = _clip_box((x1 + x, y1 + y, x2 + x, y2 + y), width, height)
                if _valid_box(full_box):
                    boxes.append((full_box, score))
    return _nms_boxes(boxes, nms_iou)


def classify_crop(model: nn.Module, classes: list[str], crop_bgr: np.ndarray, device: torch.device) -> tuple[str, float]:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = classifier_transform()(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return classes[idx], float(probs[idx].item())


def draw_prediction(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    cls_conf: float,
    det_conf: float,
) -> None:
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (255, 255, 255))
    display_label = SHORT_LABEL_MAP.get(label, label) if SHORT_LABELS else label
    text = f"{display_label} {cls_conf:.2f}" if SHOW_CLASSIFIER_CONF else display_label
    if SHOW_DETECTOR_CONF:
        text += f" | tree {det_conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(image, text, (x1 + 4, y_text + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_summary_box(image: np.ndarray, total: int, healthy: int, suspicious: int, uncertain: int) -> None:
    text = f"Total: {total} | H: {healthy} | S: {suspicious} | U: {uncertain}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 14
    padding_x = 12
    padding_y = 10
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = margin, margin
    x2, y2 = x1 + tw + padding_x * 2, y1 + th + padding_y * 2
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cv2.putText(image, text, (x1 + padding_x, y1 + padding_y + th), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def count_predictions(predictions: list[TreePrediction]) -> tuple[int, int, int, int]:
    healthy = sum(p.health_label == "healthy" for p in predictions)
    suspicious = sum(p.health_label == "suspicious" for p in predictions)
    uncertain = sum(p.health_label == "uncertain" for p in predictions)
    return len(predictions), healthy, suspicious, uncertain


def run_frame(
    frame: np.ndarray,
    detector: YOLO,
    classifier: nn.Module,
    classes: list[str],
    device: torch.device,
) -> tuple[np.ndarray, list[TreePrediction]]:
    annotated = frame.copy()
    predictions: list[TreePrediction] = []
    boxes = detect_boxes_tiled(detector, frame, device, CONF, TILE_SIZE, TILE_OVERLAP, NMS_IOU)
    for tree_id, (box, det_conf) in enumerate(boxes, start=1):
        crop, padded_box = crop_with_padding(frame, box, PADDING)
        label, cls_conf = classify_crop(classifier, classes, crop, device)
        display_label = label if cls_conf >= CLASSIFIER_CONF else "uncertain"
        draw_prediction(annotated, padded_box, display_label, cls_conf, det_conf)
        predictions.append(TreePrediction(tree_id, padded_box, det_conf, display_label, cls_conf))
    if SUMMARY_BOX:
        draw_summary_box(annotated, *count_predictions(predictions))
    return annotated, predictions


def draw_cached_frame(frame: np.ndarray, predictions: list[TreePrediction]) -> np.ndarray:
    annotated = frame.copy()
    for prediction in predictions:
        draw_prediction(
            annotated,
            prediction.box,
            prediction.health_label,
            prediction.classifier_confidence,
            prediction.detector_confidence,
        )
    if SUMMARY_BOX:
        draw_summary_box(annotated, *count_predictions(predictions))
    return annotated


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    video_path = find_file(VIDEO_NAME)
    detector_path = find_file(DETECTOR_NAME)
    classifier_path = find_file(CLASSIFIER_NAME)

    print(f"Input video: {video_path}", flush=True)
    print(f"Detector: {detector_path}", flush=True)
    print(f"Classifier: {classifier_path}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Max frames: {MAX_FRAMES_INT if MAX_FRAMES_INT is not None else 'full video'}", flush=True)
    print(
        "Noise controls: "
        f"conf={CONF}, nms_iou={NMS_IOU}, classifier_conf={CLASSIFIER_CONF}, "
        f"min_box={MIN_BOX_WIDTH}x{MIN_BOX_HEIGHT}, min_area={MIN_BOX_AREA}",
        flush=True,
    )

    device = resolve_device()
    detector = YOLO(str(detector_path))
    classifier, classes = load_classifier(classifier_path, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_video = WORK_ROOT / "outputs/videos" / f"{video_path.stem}_annotated.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    frame_id = START_FRAME
    written_frames = 0
    last_annotated = None
    last_predictions: list[TreePrediction] = []

    if START_FRAME:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    while True:
        if MAX_FRAMES_INT is not None and written_frames >= MAX_FRAMES_INT:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % FRAME_STEP == 0:
            annotated, predictions = run_frame(frame, detector, classifier, classes, device)
            last_annotated = annotated
            last_predictions = predictions
            total, healthy, suspicious, uncertain = count_predictions(predictions)
            rows.append(
                {
                    "frame_id": frame_id,
                    "total_trees": total,
                    "healthy": healthy,
                    "suspicious": suspicious,
                    "uncertain": uncertain,
                    "suspicious_ratio": suspicious / total if total else 0.0,
                }
            )
            for prediction in predictions:
                detail_rows.append(
                    {
                        "frame_id": frame_id,
                        "tree_id": prediction.tree_id,
                        "x1": prediction.box[0],
                        "y1": prediction.box[1],
                        "x2": prediction.box[2],
                        "y2": prediction.box[3],
                        "detector_confidence": prediction.detector_confidence,
                        "health_label": prediction.health_label,
                        "classifier_confidence": prediction.classifier_confidence,
                    }
                )
            if len(rows) % 10 == 0:
                print(f"Processed {len(rows)} inference frames / source frame {frame_id}", flush=True)

        if frame_id % FRAME_STEP == 0 and last_annotated is not None:
            output_frame = last_annotated
        elif PERSIST_ANNOTATIONS and last_predictions:
            output_frame = draw_cached_frame(frame, last_predictions)
        else:
            output_frame = frame

        writer.write(output_frame)
        frame_id += 1
        written_frames += 1

    cap.release()
    writer.release()

    frame_summary = WORK_ROOT / "outputs/predictions" / f"{video_path.stem}_frame_summary.csv"
    tree_predictions = WORK_ROOT / "outputs/predictions" / f"{video_path.stem}_tree_predictions.csv"
    write_csv(frame_summary, ["frame_id", "total_trees", "healthy", "suspicious", "uncertain", "suspicious_ratio"], rows)
    write_csv(
        tree_predictions,
        ["frame_id", "tree_id", "x1", "y1", "x2", "y2", "detector_confidence", "health_label", "classifier_confidence"],
        detail_rows,
    )

    latest_video = WORK_ROOT / "example_video_annotated.mp4"
    shutil.copy2(out_video, latest_video)

    print("Done.", flush=True)
    print(f"Source frames reported: {source_frames}", flush=True)
    print(f"Written frames: {written_frames}", flush=True)
    print(f"Inference frames: {len(rows)}", flush=True)
    print(f"Saved annotated video: {out_video}", flush=True)
    print(f"Saved presentation copy: {latest_video}", flush=True)
    print(f"Saved frame summary CSV: {frame_summary}", flush=True)
    print(f"Saved tree predictions CSV: {tree_predictions}", flush=True)


if __name__ == "__main__":
    main()
