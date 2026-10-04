"""Stage 6 — evaluate: FAR, FRR, EER and ROC/DET curves from comparison scores.

Convention: scores are similarities; a comparison is ACCEPTED when score >= threshold.
    FAR(t) = share of impostor scores >= t
    FRR(t) = share of genuine scores  <  t

Honest reporting (CLAUDE.md): every figure is published together with the dataset,
the number of genuine/impostor comparisons and subjects, and the threshold.
FRR at a target FAR is only reported when there are enough impostor comparisons to
observe that FAR (rule of 3: n_impostor >= 3 / FAR).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def error_rates(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, FAR, FRR), thresholds ascending."""
    genuine = np.sort(np.asarray(genuine, dtype=float))
    impostor = np.sort(np.asarray(impostor, dtype=float))
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("need at least one genuine and one impostor score")
    if thresholds is None:
        thresholds = np.unique(np.concatenate([genuine, impostor, [np.inf]]))
    far = 1.0 - np.searchsorted(impostor, thresholds, side="left") / len(impostor)
    frr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)
    return thresholds, far, frr


def equal_error_rate(genuine, impostor) -> tuple[float, float]:
    """(EER, threshold) at the threshold where |FAR - FRR| is smallest."""
    t, far, frr = error_rates(genuine, impostor)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(t[i])


def frr_at_far(genuine, impostor, target_far: float) -> tuple[float, float, float]:
    """Lowest FRR whose FAR <= target. Returns (FRR, actual FAR, threshold)."""
    t, far, frr = error_rates(genuine, impostor)
    ok = np.flatnonzero(far <= target_far)
    i = ok[0]  # thresholds ascending -> first ok has the lowest FRR
    return float(frr[i]), float(far[i]), float(t[i])


@dataclass
class OperatingPoint:
    target_far: float
    far: float
    frr: float
    threshold: float


@dataclass
class EvaluationReport:
    dataset: str
    matcher: str
    n_subjects: int
    n_genuine: int
    n_impostor: int
    eer: float
    eer_threshold: float
    operating_points: list[OperatingPoint] = field(default_factory=list)
    skipped_fars: list[float] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation: {self.matcher} on {self.dataset}",
            "",
            f"- Subjects: {self.n_subjects}",
            f"- Genuine comparisons: {self.n_genuine}",
            f"- Impostor comparisons: {self.n_impostor}",
            f"- EER: {self.eer:.2%} at threshold {self.eer_threshold:g}",
            "",
            "| Target FAR | Observed FAR | FRR | Threshold |",
            "|---|---|---|---|",
        ]
        for op in self.operating_points:
            lines.append(
                f"| {op.target_far:g} | {op.far:.4%} | {op.frr:.2%} | {op.threshold:g} |"
            )
        for far in self.skipped_fars:
            lines.append(f"| {far:g} | not measurable: needs >= {int(np.ceil(3 / far))} "
                         "impostor comparisons | | |")
        if self.notes:
            lines += ["", self.notes]
        return "\n".join(lines) + "\n"


def evaluate(
    genuine,
    impostor,
    *,
    dataset: str,
    matcher: str,
    n_subjects: int,
    target_fars: tuple[float, ...] = (1e-2, 1e-3, 1e-4),
    notes: str = "",
) -> EvaluationReport:
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    eer, eer_t = equal_error_rate(genuine, impostor)
    ops, skipped = [], []
    for target in target_fars:
        if len(impostor) < 3 / target:
            skipped.append(target)
            continue
        frr, far, t = frr_at_far(genuine, impostor, target)
        ops.append(OperatingPoint(target, far, frr, t))
    return EvaluationReport(
        dataset=dataset, matcher=matcher, n_subjects=n_subjects,
        n_genuine=len(genuine), n_impostor=len(impostor),
        eer=eer, eer_threshold=eer_t, operating_points=ops, skipped_fars=skipped, notes=notes,
    )


def plot_curves(genuine, impostor, out_dir: str | Path, title: str) -> list[Path]:
    """Write roc.svg and det.svg (vector plots only; raster images are gitignored)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, far, frr = error_rates(genuine, impostor)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(far, 1 - frr)
    ax.set(xscale="log", xlabel="False accept rate", ylabel="True accept rate",
           title=f"ROC — {title}", xlim=(1e-5, 1), ylim=(0, 1))
    ax.grid(True, which="both", alpha=0.3)
    roc = out_dir / "roc.svg"
    fig.savefig(roc, bbox_inches="tight")
    plt.close(fig)

    eps = 1e-6
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(norm.ppf(np.clip(far, eps, 1 - eps)), norm.ppf(np.clip(frr, eps, 1 - eps)))
    ticks = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5])
    ax.set_xticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
    ax.set(xlabel="False accept rate", ylabel="False reject rate", title=f"DET — {title}")
    ax.grid(True, alpha=0.3)
    det = out_dir / "det.svg"
    fig.savefig(det, bbox_inches="tight")
    plt.close(fig)
    return [roc, det]
