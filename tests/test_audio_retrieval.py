from defectlens.audio.dataset import AudioRow
from defectlens.eval.audio_retrieval import (
    FAN_FAMILY,
    PUMP_FAMILY,
    card_matches_family,
    clip_is_correct,
    sample_test_clips,
)


def test_card_matches_family_and_overlap():
    assert card_matches_family(["fan_imbalance"], FAN_FAMILY)
    assert not card_matches_family(["pump_cavitation"], FAN_FAMILY)
    # overlap tags count for both families
    assert card_matches_family(["bearing_wear"], FAN_FAMILY)
    assert card_matches_family(["bearing_wear"], PUMP_FAMILY)


def test_clip_is_correct_threshold():
    # 3 of 5 carry a fan-family tag -> correct
    good = [
        ["fan_imbalance"], ["belt_drive"], ["airflow_restriction"],
        ["pump_cavitation"], ["pump_seal_leak"],
    ]
    assert clip_is_correct(good, FAN_FAMILY, threshold=3)
    # only 2 of 5 -> not correct (compressor_knock is neither family)
    bad = [
        ["fan_imbalance"], ["belt_drive"], ["pump_cavitation"],
        ["pump_seal_leak"], ["compressor_knock"],
    ]
    assert not clip_is_correct(bad, FAN_FAMILY, threshold=3)


def test_sample_test_clips_deterministic_and_test_only():
    rows = [
        AudioRow(f"fan/test/normal_id_00_{i}.wav", "fan", "00", "test", "normal")
        for i in range(100)
    ] + [
        AudioRow(f"fan/train/normal_id_00_{i}.wav", "fan", "00", "train", "normal")
        for i in range(20)
    ]
    s1 = sample_test_clips(rows, 10, seed=42)
    s2 = sample_test_clips(rows, 10, seed=42)
    assert [r.path for r in s1] == [r.path for r in s2]  # deterministic
    assert len(s1) == 10
    assert all(r.split == "test" for r in s1)  # train excluded


def test_sample_caps_at_available():
    rows = [
        AudioRow(f"fan/test/normal_id_00_{i}.wav", "fan", "00", "test", "normal")
        for i in range(5)
    ]
    assert len(sample_test_clips(rows, 50, seed=42)) == 5
