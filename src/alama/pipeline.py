"""End-to-end: image file -> template, for camera photos and contact scans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alama.capture import load_image, load_scan, to_grayscale
from alama.enhance import EnhanceConfig, enhance
from alama.minutiae import MinutiaeConfig, extract_template
from alama.models import Template
from alama.segment import SegmentConfig, segment_fingertips


@dataclass(frozen=True)
class PipelineConfig:
    # True when photos are already single-fingertip crops (e.g. public datasets);
    # False for raw phone photos that still need segmentation.
    photo_is_cropped: bool = True
    finger_index: int = 0  # which finger (left to right) to use from a raw photo
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    enhance_photo: EnhanceConfig = field(
        default_factory=lambda: EnhanceConfig(clahe=True, target_wavelength=9.0)
    )
    enhance_scan: EnhanceConfig = field(default_factory=EnhanceConfig)
    minutiae: MinutiaeConfig = field(default_factory=MinutiaeConfig)


def template_from_photo(path: str | Path, cfg: PipelineConfig = PipelineConfig()) -> Template:
    image = load_image(path)
    if cfg.photo_is_cropped:
        gray, mask = to_grayscale(image), None
    else:
        crops = segment_fingertips(image, cfg.segment)
        if cfg.finger_index >= len(crops):
            raise ValueError(f"{path}: found {len(crops)} fingers, wanted #{cfg.finger_index}")
        crop = crops[cfg.finger_index]
        gray, mask = to_grayscale(crop.image), crop.mask
    return extract_template(enhance(gray, mask, cfg.enhance_photo), cfg.minutiae)


def template_from_scan(path: str | Path, cfg: PipelineConfig = PipelineConfig()) -> Template:
    return extract_template(enhance(load_scan(path), None, cfg.enhance_scan), cfg.minutiae)
