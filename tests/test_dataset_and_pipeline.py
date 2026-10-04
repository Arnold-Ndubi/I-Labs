import subprocess
import sys
from pathlib import Path

import cv2
import pytest
from synthetic import stripes

from alama.dataset import cross_modal_pairs, load_manifest
from alama.models import Template
from alama.pipeline import template_from_photo, template_from_scan

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(tmp_path: Path) -> Path:
    rows = ["subject_id,finger,modality,session,path"]
    for subj in ("p01", "p02"):
        for finger in (2, 3):
            for modality in ("contactless", "contact"):
                name = f"{subj}_{finger}_{modality}.png"
                cv2.imwrite(str(tmp_path / name), stripes(theta=0.3 * finger))
                rows.append(f"{subj},{finger},{modality},s1,{name}")
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_manifest_and_cross_modal_protocol(tmp_path):
    samples = load_manifest(write_manifest(tmp_path))
    assert len(samples) == 8
    pairs = list(cross_modal_pairs(samples))
    assert len(pairs) == 4 * 4
    assert sum(same for _, _, same in pairs) == 4
    assert all(p.modality == "contactless" and g.modality == "contact" for p, g, _ in pairs)


def test_manifest_rejects_bad_modality(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("subject_id,finger,modality,session,path\np01,2,selfie,s1,x.png\n")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_pipeline_runs_end_to_end_on_synthetic_images(tmp_path):
    img = tmp_path / "synthetic.png"
    cv2.imwrite(str(img), stripes(shape=(300, 300), theta=0.7, noise=0.3))
    for make in (template_from_scan, template_from_photo):
        t = make(img)
        assert isinstance(t, Template)
        assert t.width > 0 and t.height > 0


def test_biometric_guard_blocks_images_and_data():
    script = ROOT / "scripts" / "check_no_biometrics.py"
    ok = subprocess.run([sys.executable, str(script), "src/alama/enhance.py", "data/README.md"])
    assert ok.returncode == 0
    for bad in ("data/polyu/manifest.csv", "notebooks/finger.JPG", "x/probe.xyt"):
        res = subprocess.run([sys.executable, str(script), bad], capture_output=True)
        assert res.returncode == 1, bad
