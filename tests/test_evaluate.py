import json

import numpy as np
import pytest

from alama.evaluate import equal_error_rate, error_rates, evaluate, frr_at_far, plot_curves


def test_error_rates_definitions():
    t, far, frr = error_rates([3, 4, 5], [1, 2, 3], thresholds=np.array([3.0]))
    assert far[0] == pytest.approx(1 / 3)  # impostor 3 >= 3 accepted
    assert frr[0] == 0.0  # no genuine < 3


def test_rates_are_monotonic():
    rng = np.random.default_rng(0)
    _, far, frr = error_rates(rng.normal(2, 1, 500), rng.normal(0, 1, 5000))
    assert np.all(np.diff(far) <= 0)
    assert np.all(np.diff(frr) >= 0)
    assert far[-1] == 0.0


def test_eer_perfect_separation():
    eer, t = equal_error_rate([5, 6, 7], [1, 2, 3])
    assert eer == 0.0
    assert 3 < t <= 5


def test_eer_identical_distributions():
    rng = np.random.default_rng(1)
    eer, _ = equal_error_rate(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000))
    assert eer == pytest.approx(0.5, abs=0.03)


def test_frr_at_far_respects_target():
    rng = np.random.default_rng(2)
    gen, imp = rng.normal(3, 1, 1000), rng.normal(0, 1, 10000)
    frr, far, _ = frr_at_far(gen, imp, 1e-2)
    assert far <= 1e-2
    assert 0 < frr < 0.5


def test_report_skips_unmeasurable_fars_and_states_sample_sizes(tmp_path):
    rng = np.random.default_rng(3)
    report = evaluate(
        rng.normal(3, 1, 50),
        rng.normal(0, 1, 500),
        dataset="synthetic",
        matcher="test",
        n_subjects=5,
    )
    assert report.n_genuine == 50 and report.n_impostor == 500
    assert [op.target_far for op in report.operating_points] == [1e-2]
    assert report.skipped_fars == [1e-3, 1e-4]
    md = report.to_markdown()
    assert "synthetic" in md and "Impostor comparisons: 500" in md
    assert json.loads(report.to_json())["dataset"] == "synthetic"

    paths = plot_curves(rng.normal(3, 1, 50), rng.normal(0, 1, 500), tmp_path, "t")
    assert all(p.suffix == ".svg" and p.stat().st_size > 0 for p in paths)
