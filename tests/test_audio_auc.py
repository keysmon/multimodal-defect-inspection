from defectlens.eval.audio_auc import results_table


def test_results_table_shapes_and_beat_flags():
    ours = {"fan": {"00": 0.70}, "pump": {"00": 0.60}}
    baseline = {"fan": {"00": 0.65}, "pump": {"00": 0.65}}
    table = results_table(ours, baseline)
    assert table["fan"]["00"] == {"auc": 0.70, "baseline": 0.65, "beats_baseline": True}
    assert table["pump"]["00"]["beats_baseline"] is False
