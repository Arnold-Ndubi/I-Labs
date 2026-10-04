"""Data types passed between pipeline stages.

Coordinate conventions (used everywhere in alama):
- Pixel coordinates: origin top-left, x to the right (column), y down (row).
- Minutia angles: radians in [0, 2*pi), measured COUNTER-CLOCKWISE from the +x axis
  as seen on screen (ISO/IEC 19794-2 convention). Because y points down, a vector
  (dx, dy) in pixel coordinates has angle atan2(-dy, dx).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MinutiaType(str, Enum):
    ENDING = "ending"
    BIFURCATION = "bifurcation"
    OTHER = "other"


@dataclass(frozen=True)
class Minutia:
    x: float
    y: float
    angle: float  # radians, CCW from +x, see module docstring
    kind: MinutiaType
    quality: int = 0  # 0 = not computed; otherwise 1..100


@dataclass
class Template:
    """Minutiae of one finger impression plus the image geometry they refer to."""

    minutiae: list[Minutia]
    width: int
    height: int
    dpi: int = 500

    def __len__(self) -> int:
        return len(self.minutiae)

    def as_array(self) -> np.ndarray:
        """(N, 4) float array: x, y, angle, kind (0 ending, 1 bifurcation, 2 other)."""
        codes = {MinutiaType.ENDING: 0, MinutiaType.BIFURCATION: 1, MinutiaType.OTHER: 2}
        if not self.minutiae:
            return np.zeros((0, 4), dtype=float)
        return np.array(
            [[m.x, m.y, m.angle, codes[m.kind]] for m in self.minutiae], dtype=float
        )


@dataclass
class FingerCrop:
    """One fingertip cut out of a photo, rotated so the finger points up."""

    image: np.ndarray  # BGR or grayscale uint8
    mask: np.ndarray  # bool, True on the finger
    index: int  # 0-based, left to right in the source photo
    source_bbox: tuple[int, int, int, int]  # x, y, w, h of the whole finger in the source
    rotation_deg: float  # rotation applied (OpenCV convention, CCW positive)


@dataclass
class EnhancedPrint:
    """Output of the enhancement stage: a ridge image comparable to a scanned print."""

    normalised: np.ndarray  # float, zero mean / unit variance inside mask
    orientation: np.ndarray  # float radians in [0, pi), ridge direction per pixel
    wavelength: float  # median ridge period in pixels
    filtered: np.ndarray  # float Gabor response
    ridges: np.ndarray  # bool, True on ridges
    skeleton: np.ndarray  # bool, one-pixel-wide ridges
    mask: np.ndarray  # bool, region of interest (usable print area)
    scale: float = 1.0  # factor applied to the input before enhancement
    extras: dict = field(default_factory=dict)
