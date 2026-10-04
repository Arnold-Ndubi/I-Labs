import cv2
import numpy as np

from alama.segment import segment_fingertips, skin_mask

SKIN_BGR = (120, 150, 200)


def synthetic_hand(tilts=(-15, 0, 20)) -> np.ndarray:
    """Dark background with skin-coloured ellipses standing in for fingers."""
    img = np.full((600, 800, 3), 40, np.uint8)
    for i, tilt in enumerate(tilts):
        centre = (150 + 250 * i, 300)
        cv2.ellipse(img, centre, (40, 140), tilt, 0, 360, SKIN_BGR, -1)
    return img


def test_skin_mask_separates_skin_from_background():
    mask = skin_mask(synthetic_hand())
    assert mask[300, 150] and mask[300, 400]
    assert not mask[20, 20]


def test_segments_fingers_left_to_right_and_straightens_them():
    tilts = (-15, 0, 20)
    crops = segment_fingertips(synthetic_hand(tilts))
    assert [c.index for c in crops] == [0, 1, 2]
    xs = [c.source_bbox[0] for c in crops]
    assert xs == sorted(xs)
    for crop, tilt in zip(crops, tilts, strict=True):
        assert abs(crop.rotation_deg - tilt) < 3
        h, w = crop.mask.shape
        assert 80 <= h <= 115  # ~35% of the 280 px finger length
        assert w <= 90  # finger width 80 px, upright
        assert crop.mask.mean() > 0.6


def test_ignores_small_blobs():
    img = synthetic_hand((0,))
    cv2.circle(img, (700, 50), 5, SKIN_BGR, -1)
    assert len(segment_fingertips(img)) == 1
