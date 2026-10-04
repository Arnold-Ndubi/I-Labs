import math

import numpy as np
import pytest

from alama.match import BaselineMatcher, Bozorth3Matcher, SourceAfisMatcher, get_matcher
from alama.models import Minutia, MinutiaType, Template


def random_template(n=40, seed=0) -> Template:
    rng = np.random.default_rng(seed)
    kinds = [MinutiaType.ENDING, MinutiaType.BIFURCATION]
    return Template(
        [
            Minutia(float(x), float(y), float(a), kinds[k])
            for x, y, a, k in zip(
                rng.uniform(20, 280, n), rng.uniform(20, 280, n),
                rng.uniform(0, 2 * math.pi, n), rng.integers(0, 2, n), strict=True,
            )
        ],
        width=300, height=300,
    )


def transformed(t: Template, alpha: float, dx: float, dy: float, jitter=0.0, seed=1):
    """Same finger seen rotated CCW by alpha and shifted, with optional position noise."""
    rng = np.random.default_rng(seed)
    c, s = math.cos(alpha), math.sin(alpha)
    out = []
    for m in t.minutiae:
        x = c * m.x + s * m.y + dx + rng.normal(0, jitter)
        y = -s * m.x + c * m.y + dy + rng.normal(0, jitter)
        out.append(Minutia(x, y, (m.angle + alpha) % (2 * math.pi), m.kind))
    return Template(out, t.width, t.height)


def test_baseline_scores_same_finger_high_and_different_low():
    m = BaselineMatcher()
    probe = random_template(seed=0)
    genuine = m.score(probe, transformed(probe, math.radians(17), 20, -10, jitter=1.0))
    impostor = m.score(probe, random_template(seed=42))
    assert genuine > 0.6
    assert impostor < 0.1


def test_baseline_handles_empty():
    empty = Template([], 10, 10)
    assert BaselineMatcher().score(empty, random_template()) == 0.0


def test_get_matcher():
    assert get_matcher("baseline").name == "baseline"
    with pytest.raises(ValueError):
        get_matcher("nope")


@pytest.mark.nbis
@pytest.mark.skipif(not Bozorth3Matcher().available(), reason="bozorth3 not on PATH")
def test_bozorth3_genuine_beats_impostor():
    m = Bozorth3Matcher()
    probe = random_template()
    assert m.score(probe, transformed(probe, 0.2, 5, 5)) > m.score(probe, random_template(seed=9))


@pytest.mark.sourceafis
@pytest.mark.skipif(not SourceAfisMatcher().available(), reason="SourceAFIS CLI not built")
def test_sourceafis_genuine_beats_impostor():
    m = SourceAfisMatcher()
    probe = random_template()
    assert m.score(probe, transformed(probe, 0.2, 5, 5)) > m.score(probe, random_template(seed=9))
