# Third-party licence register

Rule: check the licence of every library, model, tool and dataset **before** adding
it. It must allow commercial use, or be clearly marked research-only below.
Datasets are tracked in [`data/README.md`](data/README.md).

| Component | Used for | Licence | Commercial use | Notes |
|---|---|---|---|---|
| NumPy | arrays | BSD-3-Clause | Yes | |
| SciPy | stats (DET axes) | BSD-3-Clause | Yes | |
| OpenCV (`opencv-python-headless` ≥ 4.5) | image processing | Apache-2.0 | Yes | OpenCV < 4.5 was BSD-3; also fine |
| scikit-image | skeletonisation | BSD-3-Clause | Yes | |
| Matplotlib | ROC/DET plots | Matplotlib licence (PSF-based, BSD-compatible) | Yes | |
| pytest, pytest-cov, ruff, pre-commit, nbstripout | dev tooling | MIT | Yes | Not shipped |
| JupyterLab | exploration | BSD-3-Clause | Yes | Not shipped |
| SourceAFIS for Java (`com.machinezoo.sourceafis`) | candidate matcher | Apache-2.0 | Yes | Verify transitive Maven deps when bumping the version |
| NIST NBIS (MINDTCT, BOZORTH3) | candidate matcher | US Government work, public domain in the US | Yes, to be confirmed | Confirm current export-control notes on the NIST download page before distributing binaries |

Last reviewed: 2026-10-04 (initial setup — re-verify each entry against the
upstream licence file before the first external release).
