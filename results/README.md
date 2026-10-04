# results/

One folder per evaluation run, e.g. `results/2026-10-12-polyu-bozorth3/`, as written by
`alama-eval`:

- `report.md` / `report.json` — EER, FRR at target FARs, thresholds
- `scores.csv` — one row per comparison (pseudonymous sample IDs only)
- `roc.svg`, `det.svg` — curves (vector only; raster images are gitignored)

## Reporting rules

Every FAR/FRR/EER figure — here, in slides, in messages — must state:

1. Dataset (name + version) and which modality was probe vs gallery
2. Number of subjects, genuine comparisons and impostor comparisons
3. Matcher and the decision threshold
4. Pipeline version (git commit)

Never quote a FAR smaller than about 3 / (number of impostor comparisons); the
report marks such targets as "not measurable".
