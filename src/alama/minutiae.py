"""Stage 4a — minutiae: find ridge endings and bifurcations on a thinned ridge image.

Uses the crossing number on the skeleton, then removes minutiae near the edge of
the print and pairs that are too close together (usually broken ridges or spurs).

Angle convention (see alama.models): CCW from +x, y down. For an ending the angle
points from the ridge body towards the ending; for a bifurcation it points from
the stem towards the fork. Compare prints only with templates made by the SAME
extractor: conventions differ between extractors (e.g. MINDTCT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from alama.models import EnhancedPrint, Minutia, MinutiaType, Template

# 8-neighbourhood in circular order: N, NE, E, SE, S, SW, W, NW as (dy, dx)
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@dataclass(frozen=True)
class MinutiaeConfig:
    border_margin: float = 10.0  # px from the edge of the mask
    min_distance: float = 8.0  # drop both minutiae of any closer pair
    trace_length: int = 10  # px followed along the ridge to estimate direction
    dpi: int = 500


def crossing_numbers(skeleton: np.ndarray) -> np.ndarray:
    """Crossing number per pixel (0 off the skeleton). 1 = ending, 3 = bifurcation."""
    sk = skeleton.astype(np.int16)
    h, w = sk.shape
    p = np.pad(sk, 1)
    ring = [p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in _RING]
    cn = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2
    return cn * sk


def _neighbours(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    out = []
    for dy, dx in _RING:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx]:
            out.append((yy, xx))
    return out


def _trace(
    skeleton: np.ndarray, start: tuple[int, int], visited: set, length: int
) -> tuple[int, int]:
    """Follow the ridge from `start` for up to `length` px; stop at junctions."""
    current = start
    visited = set(visited) | {start}
    for _ in range(length - 1):
        nxt = [n for n in _neighbours(skeleton, *current) if n not in visited]
        if len(nxt) != 1:
            break
        current = nxt[0]
        visited.add(current)
    return current


def _angle(dx: float, dy: float) -> float:
    """Pixel-space vector -> CCW angle in [0, 2pi)."""
    return math.atan2(-dy, dx) % (2 * math.pi)


def _ending_angle(skeleton, y, x, length) -> float:
    nbrs = _neighbours(skeleton, y, x)
    if not nbrs:
        return 0.0
    ey, ex = _trace(skeleton, nbrs[0], {(y, x)}, length)
    return _angle(x - ex, y - ey)


def _bifurcation_angle(skeleton, y, x, length) -> float:
    nbrs = _neighbours(skeleton, y, x)
    blocked = {(y, x), *nbrs}
    dirs = []
    for n in nbrs:
        ey, ex = _trace(skeleton, n, blocked - {n}, length)
        v = np.array([ex - x, ey - y], dtype=float)
        norm = np.linalg.norm(v)
        if norm > 0:
            dirs.append(v / norm)
    if not dirs:
        return 0.0
    if len(dirs) < 3:
        s = -np.sum(dirs, axis=0)
        return _angle(s[0], s[1])
    # The stem is the branch pointing most away from the others.
    sims = [sum(float(dirs[i] @ dirs[j]) for j in range(len(dirs)) if j != i)
            for i in range(len(dirs))]
    stem = dirs[int(np.argmin(sims))]
    return _angle(-stem[0], -stem[1])


def detect_minutiae(
    skeleton: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: MinutiaeConfig = MinutiaeConfig(),
    quality_map: np.ndarray | None = None,
) -> list[Minutia]:
    skeleton = skeleton.astype(bool)
    cn = crossing_numbers(skeleton)

    if mask is None:
        mask = np.ones_like(skeleton)
    dist = cv2.distanceTransform(
        np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5
    )[1:-1, 1:-1]

    found: list[Minutia] = []
    for kind, value in ((MinutiaType.ENDING, 1), (MinutiaType.BIFURCATION, 3)):
        for y, x in zip(*np.nonzero(cn == value), strict=True):
            if dist[y, x] < cfg.border_margin:
                continue
            if kind is MinutiaType.ENDING:
                angle = _ending_angle(skeleton, y, x, cfg.trace_length)
            else:
                angle = _bifurcation_angle(skeleton, y, x, cfg.trace_length)
            quality = 0
            if quality_map is not None:
                quality = int(np.clip(round(1 + 99 * float(quality_map[y, x])), 1, 100))
            found.append(Minutia(float(x), float(y), angle, kind, quality))

    return _drop_close_pairs(found, cfg.min_distance)


def _drop_close_pairs(minutiae: list[Minutia], min_distance: float) -> list[Minutia]:
    if min_distance <= 0 or len(minutiae) < 2:
        return minutiae
    xy = np.array([[m.x, m.y] for m in minutiae])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    keep = d.min(axis=1) >= min_distance
    return [m for m, k in zip(minutiae, keep, strict=True) if k]


def extract_template(
    enhanced: EnhancedPrint, cfg: MinutiaeConfig = MinutiaeConfig()
) -> Template:
    """Minutiae template from the enhancement stage output."""
    minutiae = detect_minutiae(
        enhanced.skeleton, enhanced.mask, cfg, quality_map=enhanced.extras.get("coherence")
    )
    h, w = enhanced.skeleton.shape
    return Template(minutiae=minutiae, width=w, height=h, dpi=cfg.dpi)
