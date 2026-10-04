"""Stage 5 — match: score a probe template against a gallery template.

Every matcher returns a similarity score (higher = more likely the same finger).
Scores are NOT comparable across matchers; thresholds come from evaluate.py.

Matchers:
- BaselineMatcher: small pure-Python matcher (Hough alignment + pairing). Only a
  sanity check so the pipeline runs end-to-end without external tools. Not a candidate.
- Bozorth3Matcher: NIST NBIS BOZORTH3 via subprocess (binary must be on PATH).
- SourceAfisMatcher: SourceAFIS (Java) via the CLI in java/sourceafis-cli,
  importing our ISO/IEC 19794-2 templates.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from alama.models import Template
from alama.template import to_iso19794_2, to_xyt


class Matcher(Protocol):
    name: str

    def score(self, probe: Template, gallery: Template) -> float: ...


# --------------------------------------------------------------------------- baseline


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rotate(xy: np.ndarray, alpha: float) -> np.ndarray:
    """Rotate pixel coords (y down) CCW on screen by alpha radians."""
    c, s = math.cos(alpha), math.sin(alpha)
    return np.column_stack([c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]])


@dataclass
class BaselineMatcher:
    name: str = "baseline"
    dist_tol: float = 12.0  # px
    angle_tol: float = math.radians(20)
    angle_bin: float = math.radians(10)
    shift_bin: float = 10.0  # px
    candidates: int = 5

    def score(self, probe: Template, gallery: Template) -> float:
        a, b = probe.as_array(), gallery.as_array()
        if len(a) == 0 or len(b) == 0:
            return 0.0
        # Vote over (rotation, dx, dy) from every same-type pair.
        ia, ib = np.nonzero(a[:, 3][:, None] == b[:, 3][None, :])
        if len(ia) == 0:
            return 0.0
        rot = _wrap(b[ib, 2] - a[ia, 2])
        c, s = np.cos(rot), np.sin(rot)
        ax, ay = a[ia, 0], a[ia, 1]
        tx = b[ib, 0] - (c * ax + s * ay)
        ty = b[ib, 1] - (-s * ax + c * ay)
        keys = np.column_stack([
            np.round(rot / self.angle_bin), np.round(tx / self.shift_bin),
            np.round(ty / self.shift_bin),
        ]).astype(int)
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()

        best = 0
        for k in np.argsort(counts)[::-1][: self.candidates]:
            sel = inverse == k
            alpha = float(np.angle(np.mean(np.exp(1j * rot[sel]))))
            shift = np.array([tx[sel].mean(), ty[sel].mean()])
            best = max(best, self._pair(a, b, alpha, shift))
        return best * best / (len(a) * len(b))

    def _pair(self, a: np.ndarray, b: np.ndarray, alpha: float, shift: np.ndarray) -> int:
        moved = _rotate(a[:, :2], alpha) + shift
        d = np.linalg.norm(moved[:, None, :] - b[None, :, :2], axis=-1)
        da = np.abs(_wrap(a[:, 2][:, None] + alpha - b[:, 2][None, :]))
        ok = (d <= self.dist_tol) & (da <= self.angle_tol)
        d = np.where(ok, d, np.inf)
        matched, used_a, used_b = 0, set(), set()
        for flat in np.argsort(d, axis=None):
            i, j = np.unravel_index(flat, d.shape)
            if not np.isfinite(d[i, j]):
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += 1
        return matched


# --------------------------------------------------------------------------- NBIS


@dataclass
class Bozorth3Matcher:
    """NIST NBIS BOZORTH3. Install NBIS and put `bozorth3` on PATH (or pass binary)."""

    name: str = "bozorth3"
    binary: str = "bozorth3"

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def score(self, probe: Template, gallery: Template) -> float:
        with tempfile.TemporaryDirectory(prefix="alama-") as tmp:
            p, g = Path(tmp, "probe.xyt"), Path(tmp, "gallery.xyt")
            p.write_text(to_xyt(probe))
            g.write_text(to_xyt(gallery))
            out = subprocess.run(
                [self.binary, str(p), str(g)], capture_output=True, text=True, check=True
            )
        return float(out.stdout.split()[0])


# --------------------------------------------------------------------------- SourceAFIS


@dataclass
class SourceAfisMatcher:
    """SourceAFIS via java/sourceafis-cli (build with `mvn -q package` in that folder)."""

    name: str = "sourceafis"
    jar: str = "java/sourceafis-cli/target/sourceafis-cli.jar"
    java: str = "java"

    def available(self) -> bool:
        return shutil.which(self.java) is not None and Path(self.jar).exists()

    def score(self, probe: Template, gallery: Template) -> float:
        with tempfile.TemporaryDirectory(prefix="alama-") as tmp:
            p, g = Path(tmp, "probe.iso"), Path(tmp, "gallery.iso")
            p.write_bytes(to_iso19794_2(probe))
            g.write_bytes(to_iso19794_2(gallery))
            out = subprocess.run(
                [self.java, "-jar", self.jar, "match-iso", str(p), str(g)],
                capture_output=True, text=True, check=True,
            )
        return float(out.stdout.strip())


MATCHERS = {"baseline": BaselineMatcher, "bozorth3": Bozorth3Matcher,
            "sourceafis": SourceAfisMatcher}


def get_matcher(name: str) -> Matcher:
    try:
        return MATCHERS[name]()
    except KeyError:
        raise ValueError(f"unknown matcher {name!r}; choose from {sorted(MATCHERS)}") from None
