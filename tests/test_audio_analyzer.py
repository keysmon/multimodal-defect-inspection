import pytest

from defectlens.serve.audio_analyzer import band_for_score, combine_severity


@pytest.mark.parametrize(
    "score,band,severity",
    [
        (0.10, "normal_operation", "cosmetic"),  # < p90
        (0.20, "atypical", "monitor"),           # == p90 -> monitor (not < p90)
        (0.25, "atypical", "monitor"),           # p90..p99
        (0.30, "atypical", "monitor"),           # == p99 -> monitor (not > p99)
        (0.35, "anomalous", "urgent"),           # > p99
    ],
)
def test_band_for_score_boundaries(score, band, severity):
    assert band_for_score(score, p90=0.20, p99=0.30) == (band, severity)


@pytest.mark.parametrize(
    "visual,audio,expected",
    [
        ("cosmetic", "cosmetic", "cosmetic"),
        ("monitor", "cosmetic", "monitor"),      # worst-of by rank
        ("cosmetic", "urgent", "urgent"),        # no escalation: visual below monitor
        ("monitor", "urgent", "structural"),     # escalation: monitor + audio urgent -> +1
        ("urgent", "urgent", "structural"),      # escalation: urgent + audio urgent -> +1
        ("urgent", "monitor", "urgent"),         # worst-of; audio not urgent -> no bump
        ("structural", "urgent", "structural"),  # bump capped at structural
    ],
)
def test_combine_severity(visual, audio, expected):
    assert combine_severity(visual, audio) == expected
