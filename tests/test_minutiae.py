import math

import numpy as np

from alama.minutiae import MinutiaeConfig, crossing_numbers, detect_minutiae
from alama.models import MinutiaType

CFG = MinutiaeConfig(border_margin=5, min_distance=0)


def y_skeleton() -> np.ndarray:
    """Horizontal ridge (row 50, cols 10..60) with a branch leaving up-right at col 36."""
    sk = np.zeros((100, 100), bool)
    sk[50, 10:61] = True
    for k in range(20):
        sk[49 - k, 36 + k] = True  # (49,36) ... (30,55)
    return sk


def by_kind(minutiae, kind):
    return [m for m in minutiae if m.kind is kind]


def test_crossing_number_of_line():
    sk = np.zeros((20, 20), bool)
    sk[10, 3:15] = True
    cn = crossing_numbers(sk)
    assert cn[10, 3] == 1 and cn[10, 14] == 1
    assert (cn[10, 4:14] == 2).all()
    assert cn[~sk].sum() == 0


def test_detects_endings_with_directions():
    found = detect_minutiae(y_skeleton(), cfg=CFG)
    endings = {(m.x, m.y): m.angle for m in by_kind(found, MinutiaType.ENDING)}
    assert set(endings) == {(10.0, 50.0), (60.0, 50.0), (55.0, 30.0)}
    assert math.isclose(endings[(60.0, 50.0)], 0.0, abs_tol=1e-6)
    assert math.isclose(endings[(10.0, 50.0)], math.pi, abs_tol=1e-6)
    assert math.isclose(endings[(55.0, 30.0)], math.pi / 4, abs_tol=math.radians(5))


def test_detects_one_bifurcation_pointing_into_fork():
    bifs = by_kind(detect_minutiae(y_skeleton(), cfg=CFG), MinutiaType.BIFURCATION)
    assert len(bifs) == 1
    b = bifs[0]
    assert abs(b.x - 36) <= 1 and abs(b.y - 50) <= 1
    # Stem is the left part of the ridge, so the bifurcation points right (0 rad).
    assert min(b.angle, 2 * math.pi - b.angle) < math.radians(10)


def test_border_minutiae_are_dropped():
    sk = np.zeros((50, 50), bool)
    sk[25, 0:30] = True  # touches the left image edge
    found = detect_minutiae(sk, cfg=MinutiaeConfig(border_margin=10, min_distance=0))
    assert [(m.x, m.y) for m in found] == [(29.0, 25.0)]


def test_close_pairs_are_dropped():
    sk = np.zeros((60, 60), bool)
    sk[30, 10:28] = True
    sk[30, 31:50] = True  # 3 px gap: broken ridge -> two spurious endings
    found = detect_minutiae(sk, cfg=MinutiaeConfig(border_margin=5, min_distance=8))
    assert sorted((m.x, m.y) for m in found) == [(10.0, 30.0), (49.0, 30.0)]
