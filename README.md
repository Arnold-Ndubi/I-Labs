# I-Labs
Identification program Using Finger Prints

**Alama** (working name) — verify a bank customer's identity by photographing their
fingers with an ordinary phone camera and matching against the fingerprints captured
at account opening. Current phase: a 2–3 week **feasibility experiment** to measure
how well camera photos match contact-scanner prints. See [CLAUDE.md](CLAUDE.md) for
the full brief.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Capture | `alama.capture` | Load photos / scans |
| 2. Segment | `alama.segment` | Find each finger, straighten, crop the fingertip |
| 3. Enhance | `alama.enhance` | CLAHE, scale to 500 ppi ridge period, orientation field, Gabor, binarise, thin |
| 4. Template | `alama.minutiae`, `alama.template` | Endings and bifurcations to ISO/IEC 19794-2:2005 and NBIS `.xyt` |
| 5. Match | `alama.match` | Baseline (sanity check), NIST BOZORTH3, SourceAFIS |
| 6. Evaluate | `alama.evaluate` | FAR, FRR, EER, ROC/DET with sample sizes |

`alama.pipeline` chains the stages; `alama-eval` runs the whole experiment on a manifest.

## Setup (Windows)

1. Install Python 3.11+ from python.org (tick "Add python.exe to PATH").
2. From this folder:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   pytest
   ```
3. Optional matchers:
   - **NBIS**: build/install NIST NBIS and put `mindtct` and `bozorth3` on PATH.
   - **SourceAFIS**: install JDK 17+ and Maven, then `cd java/sourceafis-cli && mvn -q package`.

## Running an experiment

```bash
alama-eval --manifest data/founder/manifest.csv --dataset-name "founder v1" --matcher baseline --out results/2026-10-founder-baseline
```

Add `--raw-photos` for uncropped phone photos and `--photo-polarity bright` if ridges
appear lighter than valleys under flash.

## Data rules

- **Never commit biometric data.** `data/`, image files and template files are gitignored,
  and a pre-commit hook blocks them even with `git add -f`.
- Record every dataset licence in [data/README.md](data/README.md) and every library in
  [LICENSES.md](LICENSES.md) before use.
- Always report FAR/FRR together with dataset, sample sizes and threshold
  (see [results/README.md](results/README.md)).
