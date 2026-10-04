"""Stage 1 — capture: load finger photos and scanner prints from disk.

Later this stage will receive frames from the guided-capture app; for the
feasibility experiment it just reads files from data/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load a colour image as BGR uint8, applying EXIF orientation (phone photos).

    Uses imdecode on raw bytes so non-ASCII Windows paths work.
    """
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"empty or missing image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def load_scan(path: str | Path) -> np.ndarray:
    """Load a contact-scanner print as grayscale uint8."""
    return to_grayscale(load_image(path))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR/BGRA/gray uint8 -> gray uint8."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
