"""Synthetic test images. Tests use SYNTHETIC data only — never real fingerprints."""

from __future__ import annotations

import numpy as np


def stripes(shape=(256, 256), wavelength=9.0, theta=0.0, noise=0.0, seed=0) -> np.ndarray:
    """uint8 image of straight dark ridges running in direction theta (pixel coords, y down)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = -np.sin(theta), np.cos(theta)  # normal to the ridges
    img = np.cos(2 * np.pi * (xx * nx + yy * ny) / wavelength)
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(127.5 + 100 * img, 0, 255).astype(np.uint8)


def angle_diff_mod_pi(a: float, b: float) -> float:
    d = (a - b) % np.pi
    return min(d, np.pi - d)
