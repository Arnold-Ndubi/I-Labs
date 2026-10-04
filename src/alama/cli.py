"""alama-eval: run the full experiment on a manifest and write results/.

    alama-eval --manifest data/polyu/manifest.csv --dataset-name "PolyU contactless-contact v?" \
               --matcher baseline --out results/2026-10-polyu-baseline

Outputs: scores.csv (pseudonymous sample IDs only), report.json, report.md, roc.svg, det.svg.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from alama.dataset import cross_modal_pairs, load_manifest
from alama.enhance import EnhanceConfig
from alama.evaluate import evaluate, plot_curves
from alama.match import get_matcher
from alama.pipeline import PipelineConfig, template_from_photo, template_from_scan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alama-eval", description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dataset-name", required=True, help="dataset name + version, for the report")
    ap.add_argument("--matcher", default="baseline", choices=["baseline", "bozorth3", "sourceafis"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--raw-photos", action="store_true", help="photos need finger segmentation")
    ap.add_argument("--photo-polarity", choices=["dark", "bright"], default="dark")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    cfg = PipelineConfig(
        photo_is_cropped=not args.raw_photos,
        enhance_photo=replace(
            EnhanceConfig(clahe=True, target_wavelength=9.0), ridge_polarity=args.photo_polarity
        ),
    )
    samples = load_manifest(args.manifest)
    matcher = get_matcher(args.matcher)

    templates = {}
    for s in samples:
        make = template_from_photo if s.modality == "contactless" else template_from_scan
        templates[s.sample_id] = make(s.path, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    genuine, impostor = [], []
    with (args.out / "scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["probe_id", "gallery_id", "genuine", "score"])
        for probe, gallery, same in cross_modal_pairs(samples):
            score = matcher.score(templates[probe.sample_id], templates[gallery.sample_id])
            (genuine if same else impostor).append(score)
            w.writerow([probe.sample_id, gallery.sample_id, int(same), f"{score:.6g}"])

    report = evaluate(
        genuine, impostor, dataset=args.dataset_name, matcher=matcher.name,
        n_subjects=len({s.subject_id for s in samples}), notes=args.notes,
    )
    (args.out / "report.json").write_text(report.to_json())
    (args.out / "report.md").write_text(report.to_markdown())
    plot_curves(genuine, impostor, args.out, f"{matcher.name} / {args.dataset_name}")
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
