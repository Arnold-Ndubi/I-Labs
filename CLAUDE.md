# I-Labs: camera-based fingerprint verification

## What we're building
A prototype that lets a bank verify a customer's identity from any smartphone by
photographing their fingers with the phone camera and matching the result against
fingerprints the bank captured at account opening.

Goal: replace SIM/OTP-based mobile banking recovery in Kenya, which SIM-swap fraudsters
exploit, and remove the branch visit customers need when they change phone or SIM.

Phones' built-in fingerprint sensors and Face ID cannot be used for this: Android and
iOS never expose fingerprint images or templates to apps. So we capture fingerprints
with the ordinary camera instead.

Full product context (PRD): https://claude.ai/code/artifact/1d3a8eb3-8c8c-47ed-af18-acc4a8e67531

## Current phase: feasibility experiment (2–3 weeks)
Prove that a phone-camera finger photo can be matched reliably against a contact-scanner
print of the same finger. Nothing else matters until we have these numbers.

Pipeline to build:
1. Capture: load a finger photo (later: guided capture in an app with flash on).
2. Segment: find each fingertip in the image, crop and straighten it.
3. Enhance: convert to a ridge image comparable to a scanned print
   (grayscale, normalise, orientation field, Gabor filtering, binarise, thin).
4. Template: extract minutiae (ridge endings and bifurcations) into a standard
   template format (ISO/IEC 19794-2 or ANSI 378 where possible).
5. Match: compare against a template from a contact scan. Candidates: SourceAFIS,
   NIST NBIS (MINDTCT + BOZORTH3). Compare both.
6. Evaluate: false accept rate (FAR), false reject rate (FRR), equal error rate (EER),
   with ROC/DET curves.

Test data:
- The founder's own fingers: phone-camera photos plus scans from a cheap USB reader.
- Public contactless-to-contact datasets (e.g. Hong Kong PolyU). Check each licence
  before use and record it in `data/README.md`.

Later (not now): finger liveness / spoof detection, mobile capture app, server API,
integration with the face-verification and risk-scoring platform described in the PRD.

## Stack
- Python 3.11+
- OpenCV, NumPy, scikit-image for image processing
- pytest for tests
- Jupyter notebooks only for exploration; real logic lives in `src/`

## Suggested layout
```
src/alama/        capture, segment, enhance, minutiae, match, evaluate modules
notebooks/        exploration
tests/
data/             gitignored; never committed
results/          evaluation outputs (metrics, plots)
```

## Rules
- Never commit biometric data (fingerprint images, templates, scans). `data/` and any
  image files must be in `.gitignore`. Biometric data is sensitive personal data under
  Kenya's Data Protection Act 2019.
- Check the licence of every model, library and dataset before adding it; it must
  allow commercial use, or be clearly marked as research-only.
- Report accuracy honestly: always state dataset, sample size and thresholds with any
  FAR/FRR figure.
- Keep each pipeline stage a separate, testable function so stages can be swapped.

## Naming
Repo: I-Labs. Working product name: Alama (Swahili for "mark"; alama za vidole =
fingerprints). Not final.
