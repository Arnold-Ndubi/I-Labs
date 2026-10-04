# data/ — biometric data (never committed)

Everything in this folder except this README is gitignored. Fingerprint images,
scans and templates are **sensitive personal data** under Kenya's Data Protection
Act 2019. Do not copy them anywhere else (chat, cloud drives, notebooks with
outputs, issue trackers).

## Layout

```
data/
  founder/            founder's own fingers (phone photos + USB scanner)
    manifest.csv
    photos/
    scans/
  polyu/              Hong Kong PolyU contactless-to-contact database (if licensed)
    manifest.csv
    ...
```

Manifest format: see `src/alama/dataset.py` (`subject_id, finger, modality, session, path`).
Use pseudonymous subject IDs (`p01`, `p02`, …); never names or ID numbers.

## Dataset licence register

Fill in **before** a dataset is downloaded or used. A dataset may be used only if
its licence allows commercial use, or it is clearly marked research-only and kept
out of anything shipped.

| Dataset | Source | Licence / terms | Commercial use? | Status | Checked by / date |
|---|---|---|---|---|---|
| Founder's own fingers | Captured in-house | Own data; written self-consent on file | Yes | Not yet captured | |
| PolyU Contactless 2D to Contact-based 2D Fingerprint Images Database | The Hong Kong Polytechnic University | Licence agreement required; terms to be read and recorded here | **Assume research-only until confirmed** | Not requested | |

## Capture log (founder data)

| Session | Date | Device (phone model / scanner model) | Lighting / flash | Fingers | Notes |
|---|---|---|---|---|---|
| | | | | | |
