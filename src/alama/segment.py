"""Stage 2 — segment: find each finger in a photo, straighten it, crop the fingertip.

Current assumptions (fine for guided capture, revisit with real photos):
- Fingers point roughly upward in the frame and are spread apart.
- Skin is separable from the background by colour (YCrCb thresholds). The palm
  should be mostly out of frame, otherwise it merges with the fingers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from alama.models import FingerCrop


@dataclass(frozen=True)
class SegmentConfig:
    cr_range: tuple[int, int] = (133, 173)
    cb_range: tuple[int, int] = (77, 127)
    min_area_frac: float = 0.005  # ignore blobs smaller than this fraction of the image
    max_fingers: int = 4
    tip_fraction: float = 0.35  # top share of the finger length kept as the fingertip


def skin_mask(image_bgr: np.ndarray, cfg: SegmentConfig = SegmentConfig()) -> np.ndarray:
    """Boolean mask of skin-coloured pixels, cleaned with morphology."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, cfg.cr_range[0], cfg.cb_range[0]], dtype=np.uint8)
    upper = np.array([255, cfg.cr_range[1], cfg.cb_range[1]], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    k = max(3, (min(mask.shape[:2]) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _upright_angle(component: np.ndarray) -> float:
    """OpenCV rotation (degrees, CCW positive) that makes the blob's long axis vertical."""
    ys, xs = np.nonzero(component)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = eigvecs[:, -1]  # principal axis
    angle = math.degrees(math.atan2(-vx, vy))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


def _crop_finger(
    image: np.ndarray, component: np.ndarray, index: int, cfg: SegmentConfig
) -> FingerCrop:
    x, y, w, h = cv2.boundingRect(component.astype(np.uint8))
    patch = image[y : y + h, x : x + w]
    patch_mask = component[y : y + h, x : x + w].astype(np.uint8) * 255

    side = int(math.ceil(math.hypot(w, h)))
    top, left = (side - h) // 2, (side - w) // 2
    pads = (top, side - h - top, left, side - w - left)
    patch = cv2.copyMakeBorder(patch, *pads, cv2.BORDER_CONSTANT, value=0)
    patch_mask = cv2.copyMakeBorder(patch_mask, *pads, cv2.BORDER_CONSTANT, value=0)

    angle = _upright_angle(component[y : y + h, x : x + w])
    rot = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    patch = cv2.warpAffine(patch, rot, (side, side), flags=cv2.INTER_LINEAR)
    patch_mask = cv2.warpAffine(patch_mask, rot, (side, side), flags=cv2.INTER_NEAREST) > 0

    rows = np.flatnonzero(patch_mask.any(axis=1))
    r0, r1 = rows[0], rows[-1] + 1
    r1 = r0 + max(1, int(round((r1 - r0) * cfg.tip_fraction)))
    cols = np.flatnonzero(patch_mask[r0:r1].any(axis=0))
    c0, c1 = cols[0], cols[-1] + 1

    return FingerCrop(
        image=patch[r0:r1, c0:c1].copy(),
        mask=patch_mask[r0:r1, c0:c1].copy(),
        index=index,
        source_bbox=(x, y, w, h),
        rotation_deg=angle,
    )


def segment_fingertips(
    image_bgr: np.ndarray, cfg: SegmentConfig = SegmentConfig()
) -> list[FingerCrop]:
    """Find up to cfg.max_fingers fingers and return upright fingertip crops, left to right."""
    mask = skin_mask(image_bgr, cfg)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    min_area = cfg.min_area_frac * mask.size
    blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    blobs = sorted(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    blobs = sorted(blobs[: cfg.max_fingers], key=lambda i: centroids[i][0])
    return [_crop_finger(image_bgr, labels == i, k, cfg) for k, i in enumerate(blobs)]
