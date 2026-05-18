from __future__ import annotations

import cv2
import numpy as np

COLORS = {
    "healthy": (48, 180, 90),
    "suspicious": (40, 120, 255),
    "uncertain": (160, 160, 160),
}

SHORT_LABELS = {
    "healthy": "H",
    "suspicious": "S",
    "uncertain": "U",
}


def draw_prediction(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    cls_conf: float,
    det_conf: float | None = None,
    short_label: bool = False,
    show_detector_conf: bool = True,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (255, 255, 255))
    display_label = SHORT_LABELS.get(label, label) if short_label else label
    text = f"{display_label} {cls_conf:.2f}"
    if det_conf is not None and show_detector_conf:
        text += f" | tree {det_conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(image, text, (x1 + 4, y_text + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def draw_summary_box(image: np.ndarray, total: int, healthy: int, suspicious: int, uncertain: int) -> np.ndarray:
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
    return image
