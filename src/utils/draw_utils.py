from __future__ import annotations

import cv2
import numpy as np

COLORS = {
    "healthy": (48, 180, 90),
    "suspicious": (40, 120, 255),
}


def draw_prediction(image: np.ndarray, box: tuple[int, int, int, int], label: str, cls_conf: float, det_conf: float | None = None) -> np.ndarray:
    x1, y1, x2, y2 = box
    color = COLORS.get(label, (255, 255, 255))
    text = f"{label} {cls_conf:.2f}"
    if det_conf is not None:
        text += f" | tree {det_conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(image, text, (x1 + 4, y_text + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return image
