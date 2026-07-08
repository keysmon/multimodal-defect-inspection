"""DCASE 2020 Task 2 (MIMII) audio dataset scanning.

Filenames encode everything: <label>_id_<machine_id>_<clip_num>.wav under
<machine>/{train,test}/. Train contains normals only (DCASE unsupervised
protocol); test contains normals and anomalies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_WAV_RE = re.compile(r"^(normal|anomaly)_id_(\d\d)_\d+\.wav$")


@dataclass(frozen=True)
class AudioRow:
    path: str
    machine: str
    machine_id: str
    split: str
    label: str


def parse_wav_name(path: Path, machine: str) -> AudioRow:
    m = _WAV_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognized DCASE wav name: {path.name}")
    return AudioRow(
        path=str(path),
        machine=machine,
        machine_id=m.group(2),
        split=path.parent.name,
        label=m.group(1),
    )


def scan_machine_dir(root: Path, machine: str) -> list[AudioRow]:
    """All parseable wavs under <root>/{train,test}, sorted by path."""
    rows = []
    for split in ("train", "test"):
        for wav in sorted((root / split).glob("*.wav")):
            rows.append(parse_wav_name(wav, machine=machine))
    return rows
