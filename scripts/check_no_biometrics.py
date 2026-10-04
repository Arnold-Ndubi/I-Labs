"""Pre-commit guard: refuse to commit anything that could be biometric data.

Backstop for .gitignore (which `git add -f` bypasses). Exits non-zero if any
staged path is under data/ (other than data/README.md) or has an image or
template extension.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

BLOCKED_SUFFIXES = {
    # images
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".heic",
    ".heif", ".dng", ".raw", ".pgm", ".ppm", ".pnm", ".jp2", ".wsq",
    # templates / minutiae
    ".xyt", ".min", ".ist", ".fmr", ".iso", ".ansi", ".cbor", ".template",
    # array dumps
    ".npy", ".npz", ".pkl", ".pickle", ".h5", ".hdf5",
}
ALLOWED = {"data/README.md"}


def blocked(path: str) -> bool:
    p = PurePosixPath(path.replace("\\", "/"))
    if str(p) in ALLOWED:
        return False
    if p.parts and p.parts[0] == "data":
        return True
    return p.suffix.lower() in BLOCKED_SUFFIXES


def main(paths: list[str]) -> int:
    bad = [p for p in paths if blocked(p)]
    for p in bad:
        print(f"BLOCKED (possible biometric data): {p}", file=sys.stderr)
    if bad:
        print(
            "\nBiometric data must never be committed (Kenya DPA 2019). "
            "Unstage these files with `git restore --staged <file>`.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
