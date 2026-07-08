"""CLAP audio embeddings for anomaly scoring and (Phase 5.3) card retrieval.

laion/clap-htsat-unfused expects 48kHz mono input; DCASE wavs are 16kHz, so
we resample on load. Embeddings are L2-normalized so downstream cosine math
(kNN scorer, retrieval) can use plain dot products.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SR = 48_000


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_clap(device: str):
    import torch  # noqa: F401
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    return model, processor


def load_wav_48k(path: Path) -> np.ndarray:
    import torchaudio

    wave, sr = torchaudio.load(str(path))
    wave = wave.mean(dim=0, keepdim=True)  # mono
    if sr != CLAP_SR:
        wave = torchaudio.functional.resample(wave, sr, CLAP_SR)
    return wave.squeeze(0).numpy()


def embed_audio_files(model, processor, paths: list[Path], device: str, batch_size: int = 8) -> np.ndarray:
    import torch

    out = []
    for batch in batched(list(paths), batch_size):
        audios = [load_wav_48k(p) for p in batch]
        inputs = processor(audios=audios, sampling_rate=CLAP_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_audio_features(**inputs)
        if not isinstance(feats, torch.Tensor):  # transformers v5 output object
            feats = feats.pooler_output
        out.append(feats.cpu().numpy())
    embs = np.concatenate(out, axis=0)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)
