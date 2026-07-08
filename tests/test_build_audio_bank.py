import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_audio_bank import (  # noqa: E402
    calibration_percentiles,
    select_test_clips,
    select_train_normals,
)

from defectlens.audio.dataset import AudioRow


def test_calibration_percentiles_values():
    scores = np.arange(0, 101, dtype=float)  # 0..100 inclusive -> percentile p == value p
    calib = calibration_percentiles(scores)
    assert calib["p50"] == pytest.approx(50.0)
    assert calib["p90"] == pytest.approx(90.0)
    assert calib["p99"] == pytest.approx(99.0)
    assert calib["p50"] < calib["p90"] < calib["p99"]


def test_calibration_percentiles_accepts_list():
    calib = calibration_percentiles([1.0, 2.0, 3.0, 4.0])
    assert set(calib) == {"p50", "p90", "p99"}


def test_calibration_percentiles_empty_raises():
    with pytest.raises(ValueError, match="no test-normal scores"):
        calibration_percentiles(np.array([]))


def _rows(machine, split, n_normal, n_anomaly):
    # Emulate scan_machine_dir's sort: anomaly_ sorts before normal_ by filename.
    rows = [
        AudioRow(f"{machine}/{split}/anomaly_id_00_{i}.wav", machine, "00", split, "anomaly")
        for i in range(n_anomaly)
    ] + [
        AudioRow(f"{machine}/{split}/normal_id_00_{i}.wav", machine, "00", split, "normal")
        for i in range(n_normal)
    ]
    return sorted(rows, key=lambda r: r.path)


def test_limit_keeps_test_normals_despite_anomaly_first_sort():
    # Regression: a naive head(limit) on the sorted test rows selects only
    # anomalies (they sort first), leaving zero normals to calibrate from.
    def fake_scan(root, machine):
        return _rows(machine, "test", n_normal=50, n_anomaly=200)

    paths, labels = select_test_clips(fake_scan, Path("data/raw/audio"), ["fan"], limit=20)
    assert labels.count("normal") == 20
    assert labels.count("anomaly") == 20


def test_select_train_normals_excludes_anomaly_and_test():
    def fake_scan(root, machine):
        return _rows(machine, "train", n_normal=30, n_anomaly=0) + _rows(
            machine, "test", n_normal=10, n_anomaly=10
        )

    paths = select_train_normals(fake_scan, Path("data/raw/audio"), ["fan"], limit=None)
    assert len(paths) == 30
    assert all("train" in str(p) and "normal" in str(p) for p in paths)
