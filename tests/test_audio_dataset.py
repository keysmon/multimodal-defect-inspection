from pathlib import Path

from defectlens.audio.dataset import AudioRow, parse_wav_name, scan_machine_dir


def test_parse_wav_name_normal_train():
    row = parse_wav_name(Path("data/raw/audio/fan/train/normal_id_00_00000123.wav"), machine="fan")
    assert row == AudioRow(
        path="data/raw/audio/fan/train/normal_id_00_00000123.wav",
        machine="fan", machine_id="00", split="train", label="normal",
    )


def test_parse_wav_name_anomaly_test():
    row = parse_wav_name(Path("data/raw/audio/pump/test/anomaly_id_06_00000001.wav"), machine="pump")
    assert row.label == "anomaly" and row.split == "test" and row.machine_id == "06"


def test_scan_machine_dir(tmp_path):
    (tmp_path / "train").mkdir(); (tmp_path / "test").mkdir()
    (tmp_path / "train" / "normal_id_00_00000000.wav").touch()
    (tmp_path / "test" / "normal_id_00_00000001.wav").touch()
    (tmp_path / "test" / "anomaly_id_00_00000002.wav").touch()
    (tmp_path / "test" / "notes.txt").touch()  # ignored
    rows = scan_machine_dir(tmp_path, machine="fan")
    assert len(rows) == 3
    assert sorted({r.split for r in rows}) == ["test", "train"]
    assert sum(r.label == "anomaly" for r in rows) == 1
