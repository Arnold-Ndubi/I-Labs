"""Stage 3 — enhance: turn a fingertip photo or a scan into a clean ridge image.

Steps (each a separate function so they can be swapped or tuned):
    contrast (CLAHE) -> scale to a target ridge period -> normalise -> region of interest
    -> orientation field -> ridge wavelength -> oriented Gabor filter -> binarise -> thin

Based on Hong, Wan & Jain (1998), "Fingerprint image enhancement: algorithm and
performance evaluation". Orientation angles here are RIDGE directions in pixel
coordinates (x right, y down), in [0, pi).

Ridge polarity: on contact scans ridges are dark. On camera photos they can be
dark OR bright depending on lighting and flash; set `ridge_polarity` per capture
setup and check it on real photos early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.morphology import skeletonize

from alama.models import EnhancedPrint


@dataclass(frozen=True)
class EnhanceConfig:
    ridge_polarity: Literal["dark", "bright"] = "dark"
    clahe: bool = False
    # Rescale so the median ridge period becomes this many pixels (~9 px is 500 ppi).
    # None = keep the input scale (contact scans already at 500 ppi).
    target_wavelength: float | None = None
    roi_block: int = 16
    roi_std_thresh: float = 0.1
    grad_sigma: float = 1.0
    block_sigma: float = 7.0
    orient_smooth_sigma: float = 7.0
    freq_block: int = 32
    min_wavelength: float = 4.0
    max_wavelength: float = 16.0
    gabor_kx: float = 0.65
    gabor_ky: float = 0.65
    gabor_angles: int = 36
    min_ridge_area: int = 20
    max_hole_area: int = 20


# --------------------------------------------------------------------------- helpers


def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False
    return keep[labels]


def _fill_small_holes(binary: np.ndarray, max_size: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~binary).astype(np.uint8), 4)
    small = stats[:, cv2.CC_STAT_AREA] < max_size
    small[0] = False
    return binary | small[labels]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


# --------------------------------------------------------------------------- steps


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def normalise(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Zero mean, unit variance (statistics taken inside mask if given)."""
    img = gray.astype(np.float32)
    vals = img[mask] if mask is not None and mask.any() else img
    out = (img - vals.mean()) / (vals.std() + 1e-6)
    if mask is not None:
        out[~mask] = 0.0
    return out


def roi_mask(
    norm: np.ndarray,
    block: int = 16,
    std_thresh: float = 0.1,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Region with ridge texture: local standard deviation above a threshold."""
    mean = cv2.boxFilter(norm, cv2.CV_32F, (block, block))
    sq = cv2.boxFilter(norm * norm, cv2.CV_32F, (block, block))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    mask = (std > std_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0
    if base_mask is not None:
        mask &= base_mask
    return _largest_component(mask)


def orientation_field(
    norm: np.ndarray,
    grad_sigma: float = 1.0,
    block_sigma: float = 7.0,
    smooth_sigma: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge orientation in [0, pi) per pixel, and coherence in [0, 1]."""
    smoothed = cv2.GaussianBlur(norm, (0, 0), grad_sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)  # d/dx (along columns)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)  # d/dy (along rows)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), block_sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), block_sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), block_sigma)

    sin2 = 2 * gxy
    cos2 = gxx - gyy
    mag = np.sqrt(sin2 * sin2 + cos2 * cos2)
    coherence = mag / (gxx + gyy + 1e-6)
    sin2 = cv2.GaussianBlur(sin2 / (mag + 1e-6), (0, 0), smooth_sigma)
    cos2 = cv2.GaussianBlur(cos2 / (mag + 1e-6), (0, 0), smooth_sigma)

    gradient_dir = 0.5 * np.arctan2(sin2, cos2)  # dominant gradient direction
    ridge_dir = np.mod(gradient_dir + np.pi / 2, np.pi)  # ridges run perpendicular
    return ridge_dir.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


def ridge_wavelength(
    norm: np.ndarray,
    mask: np.ndarray,
    block: int = 32,
    min_wavelength: float = 4.0,
    max_wavelength: float = 16.0,
    min_coverage: float = 0.75,
    fft_size: int = 128,
) -> float:
    """Median ridge period (pixels) from the spectral peak of overlapping blocks.

    Returns NaN if no block had enough coverage or a clear peak.
    """
    fft_size = max(fft_size, block)
    window = np.outer(np.hanning(block), np.hanning(block)).astype(np.float32)
    f = np.fft.fftfreq(fft_size)
    radius = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    band = (radius >= 1.0 / max_wavelength) & (radius <= 1.0 / min_wavelength)

    h, w = norm.shape
    step = max(1, block // 2)
    wavelengths = []
    for y in range(0, h - block + 1, step):
        for x in range(0, w - block + 1, step):
            if mask[y : y + block, x : x + block].mean() < min_coverage:
                continue
            blk = norm[y : y + block, x : x + block]
            blk = (blk - blk.mean()) * window
            spec = np.abs(np.fft.fft2(blk, s=(fft_size, fft_size)))
            spec[~band] = 0
            peak = int(np.argmax(spec))
            if spec.flat[peak] <= 0:
                continue
            wavelengths.append(1.0 / radius.flat[peak])
    return float(np.median(wavelengths)) if wavelengths else float("nan")


def normalise_scale(
    gray: np.ndarray,
    mask: np.ndarray,
    target_wavelength: float = 9.0,
    search_max_wavelength: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resize so the ridge period is ~target_wavelength px. Returns (gray, mask, scale)."""
    norm = normalise(gray, mask)
    block = int(max(32, 2.5 * search_max_wavelength))
    wl = ridge_wavelength(
        norm, mask, block=block, min_wavelength=3.0,
        max_wavelength=search_max_wavelength, min_coverage=0.6, fft_size=256,
    )
    if not np.isfinite(wl):
        return gray, mask, 1.0
    scale = target_wavelength / wl
    size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, size, interpolation=interp)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return gray, mask, float(scale)


def gabor_kernel(theta: float, wavelength: float, kx: float = 0.65, ky: float = 0.65):
    """Even-symmetric Gabor kernel tuned to ridges running in direction theta."""
    sx, sy = kx * wavelength, ky * wavelength
    half = int(np.ceil(3 * max(sx, sy)))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    nx, ny = -np.sin(theta), np.cos(theta)  # unit normal to the ridges
    u = xx * nx + yy * ny  # across ridges
    v = -xx * ny + yy * nx  # along ridges
    kernel = np.exp(-0.5 * (u**2 / sx**2 + v**2 / sy**2)) * np.cos(2 * np.pi * u / wavelength)
    return (kernel - kernel.mean()).astype(np.float32)


def gabor_filter(
    norm: np.ndarray,
    orientation: np.ndarray,
    wavelength: float,
    kx: float = 0.65,
    ky: float = 0.65,
    n_angles: int = 36,
) -> np.ndarray:
    """Filter each pixel with the Gabor kernel matching its ridge orientation."""
    bins = np.round(orientation / np.pi * n_angles).astype(int) % n_angles
    out = np.zeros_like(norm, dtype=np.float32)
    for k in range(n_angles):
        sel = bins == k
        if not sel.any():
            continue
        kernel = gabor_kernel(k * np.pi / n_angles, wavelength, kx, ky)
        response = cv2.filter2D(norm, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        out[sel] = response[sel]
    return out


def binarise(filtered: np.ndarray, mask: np.ndarray, cfg: EnhanceConfig) -> np.ndarray:
    """Ridges (dark after polarity correction) give a negative Gabor response."""
    ridges = (filtered < 0) & mask
    ridges = _remove_small_components(ridges, cfg.min_ridge_area)
    ridges = _fill_small_holes(ridges, cfg.max_hole_area)
    return ridges & mask


def thin(ridges: np.ndarray) -> np.ndarray:
    return skeletonize(ridges).astype(bool)


# --------------------------------------------------------------------------- pipeline


def enhance(
    gray: np.ndarray,
    base_mask: np.ndarray | None = None,
    cfg: EnhanceConfig = EnhanceConfig(),
) -> EnhancedPrint:
    """Run the full enhancement stage on a grayscale uint8 image."""
    mask0 = np.ones(gray.shape, bool) if base_mask is None else base_mask.astype(bool)
    if cfg.clahe:
        gray = apply_clahe(gray)

    scale = 1.0
    if cfg.target_wavelength:
        gray, mask0, scale = normalise_scale(gray, mask0, cfg.target_wavelength)

    norm = normalise(gray, mask0)
    if cfg.ridge_polarity == "bright":
        norm = -norm
    mask = roi_mask(norm, cfg.roi_block, cfg.roi_std_thresh, base_mask=mask0)
    norm = normalise(norm, mask)

    orientation, coherence = orientation_field(
        norm, cfg.grad_sigma, cfg.block_sigma, cfg.orient_smooth_sigma
    )
    wl = ridge_wavelength(norm, mask, cfg.freq_block, cfg.min_wavelength, cfg.max_wavelength)
    if not np.isfinite(wl):
        wl = cfg.target_wavelength or 9.0

    filtered = gabor_filter(norm, orientation, wl, cfg.gabor_kx, cfg.gabor_ky, cfg.gabor_angles)
    ridges = binarise(filtered, mask, cfg)
    skeleton = thin(ridges) & mask

    return EnhancedPrint(
        normalised=norm,
        orientation=orientation,
        wavelength=wl,
        filtered=filtered,
        ridges=ridges,
        skeleton=skeleton,
        mask=mask,
        scale=scale,
        extras={"coherence": coherence},
    )
