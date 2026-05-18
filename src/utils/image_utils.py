from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not write image: {path}")


def crop_with_padding(image: np.ndarray, box: Iterable[float], padding: float = 0.2) -> Tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box after padding: {(x1, y1, x2, y2)}")
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
