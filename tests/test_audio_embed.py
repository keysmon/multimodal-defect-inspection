import numpy as np

from defectlens.audio.embed import CLAP_MODEL, batched


def test_batched_covers_all_items_in_order():
    items = list(range(10))
    batches = list(batched(items, size=4))
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_clap_model_constant():
    assert CLAP_MODEL == "laion/clap-htsat-unfused"


def test_load_wav_48k_resamples_16k_to_48k(tmp_path):
    import soundfile as sf

    from defectlens.audio.embed import CLAP_SR, load_wav_48k

    sr = 16_000
    wav = tmp_path / "t.wav"
    sf.write(wav, np.sin(np.linspace(0, 440, sr)).astype("float32"), sr)
    out = load_wav_48k(wav)
    assert out.dtype == np.float32
    assert len(out) == CLAP_SR  # 1 second at 48k
