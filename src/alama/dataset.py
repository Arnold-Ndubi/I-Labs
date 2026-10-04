"""Dataset manifests and comparison protocol.

A manifest is a CSV kept under data/ (never committed) with columns:
    subject_id, finger, modality, session, path
- subject_id: pseudonymous ID (never a name or ID number)
- finger:     ISO finger position 1..10 (1 = right thumb ... 10 = left little)
- modality:   "contactless" (phone/camera photo) or "contact" (scanner)
- session:    capture session label
- path:       image path, relative to the manifest file

Protocol: every contactless sample (probe) is compared with every contact sample
(gallery). Genuine = same subject and same finger; everything else is impostor.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("subject_id", "finger", "modality", "session", "path")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    subject_id: str
    finger: int
    modality: str
    session: str
    path: Path

    @property
    def identity(self) -> tuple[str, int]:
        return self.subject_id, self.finger


def load_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    samples = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            modality = row["modality"].strip().lower()
            if modality not in ("contactless", "contact"):
                raise ValueError(f"row {i + 2}: modality must be contactless or contact")
            samples.append(
                Sample(
                    sample_id=f"s{i:05d}",
                    subject_id=row["subject_id"].strip(),
                    finger=int(row["finger"]),
                    modality=modality,
                    session=row["session"].strip(),
                    path=(path.parent / row["path"].strip()).resolve(),
                )
            )
    return samples


def cross_modal_pairs(samples: list[Sample]) -> Iterator[tuple[Sample, Sample, bool]]:
    """Yield (probe=contactless, gallery=contact, is_genuine)."""
    probes = [s for s in samples if s.modality == "contactless"]
    gallery = [s for s in samples if s.modality == "contact"]
    for p in probes:
        for g in gallery:
            yield p, g, p.identity == g.identity
