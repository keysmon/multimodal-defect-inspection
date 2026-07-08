# Phase 5.2: Audio Anomaly Detection (ML core) - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DCASE-faithful unsupervised audio anomaly detection on MIMII fan+pump (CLAP embeddings + kNN density on normals only), with per-machine-ID AUC compared to the published DCASE 2020 Task 2 baseline.

**Architecture:** Pretrained CLAP audio embeddings (no training) + per-machine-ID k-nearest-neighbor distance scoring fit on NORMAL training clips only, per the DCASE protocol. Product integration (corpus cards, serving, UI) is the separate Phase 5.3 plan.

**Tech Stack:** transformers ClapModel (laion/clap-htsat-unfused), torchaudio (resample 16k->48k), numpy/sklearn (kNN + AUC), pytest.

**Branch:** `feat/phase5-audio` off `main`. Spec: `docs/superpowers/specs/2026-07-08-phase5-multimodal-aws-design.md` decisions 4-5.

**Money:** $0 (download + CPU/MPS embedding, ~20 min compute).

**Verified facts (2026-07-08, dcase.community + zenodo.org/records/3678171):**
- Data: DCASE 2020 Task 2 development set, `dev_data_fan.zip` 1.4GB MD5 649bdfc06263ae7a838963f43b6641e6, `dev_data_pump.zip` 1.0GB MD5 90e7091ef722b7238a7f1009365779cd, from `https://zenodo.org/records/3678171/files/<name>?download=1`.
- License CC BY-NC-SA 4.0 (attribution required in README; noncommercial OK for this project).
- Clips ~10s, 16kHz; machine IDs 0/2/4/6 per type; ~1000 normal train + 100-200 normal/anomaly test per ID.
- Baseline (autoencoder, Koizumi et al. 2020): fan avg AUC 65.83%, pump avg AUC 72.89%. Per-ID numbers: cite from arXiv:2006.05822 Table A.1 during Task 5 (fetch, do not trust memory).

---

### Task 1: Data fetch script + storage layout

**Files:**
- Create: `scripts/fetch_dcase_audio.sh`

No pytest for a download script; verification is MD5 + structure listing. Storage follows the existing repo pattern: real bytes live outside the repo in `~/datasets/dcase2020t2/`, the repo sees them via `data/raw/audio/` symlinks (data/ is gitignored except manifests).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Fetch DCASE 2020 Task 2 dev data (fan + pump) - MIMII subset, CC BY-NC-SA 4.0.
# Layout after run:
#   ~/datasets/dcase2020t2/{fan,pump}/{train,test}/*.wav
#   data/raw/audio/{fan,pump} -> symlinks into the above
set -euo pipefail

DEST="${HOME}/datasets/dcase2020t2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$DEST"

declare -A MD5S=(
  [dev_data_fan.zip]=649bdfc06263ae7a838963f43b6641e6
  [dev_data_pump.zip]=90e7091ef722b7238a7f1009365779cd
)

for NAME in dev_data_fan.zip dev_data_pump.zip; do
  ZIP="${DEST}/${NAME}"
  if [[ ! -f "$ZIP" ]]; then
    echo "== downloading ${NAME} =="
    curl -L -o "$ZIP" "https://zenodo.org/records/3678171/files/${NAME}?download=1"
  fi
  echo "== verifying ${NAME} =="
  GOT=$(md5 -q "$ZIP")
  [[ "$GOT" == "${MD5S[$NAME]}" ]] || { echo "MD5 MISMATCH for ${NAME}: ${GOT}" >&2; exit 1; }
  MACHINE="${NAME#dev_data_}"; MACHINE="${MACHINE%.zip}"
  if [[ ! -d "${DEST}/${MACHINE}" ]]; then
    echo "== extracting ${NAME} =="
    unzip -q "$ZIP" -d "$DEST"
  fi
done

mkdir -p "${REPO_ROOT}/data/raw/audio"
for MACHINE in fan pump; do
  ln -sfn "${DEST}/${MACHINE}" "${REPO_ROOT}/data/raw/audio/${MACHINE}"
done
echo "== done =="
find "${REPO_ROOT}/data/raw/audio/" -name "*.wav" | head -3
for MACHINE in fan pump; do
  printf "%s: train=%s test=%s\n" "$MACHINE" \
    "$(ls "${DEST}/${MACHINE}/train" | wc -l | tr -d ' ')" \
    "$(ls "${DEST}/${MACHINE}/test" | wc -l | tr -d ' ')"
done
```

- [ ] **Step 2: Run it**

Run: `bash scripts/fetch_dcase_audio.sh` (~2.4GB download; resumable — re-run skips existing zips)
Expected: MD5 checks pass; counts printed (fan train ~3675, test ~1475; pump train ~3349, test ~856 — exact numbers may differ slightly; anything in that magnitude with BOTH normal_ and anomaly_ files in test/ is correct. If the zip extracts to a different directory shape than `<machine>/{train,test}`, STOP and adapt the symlink block to the actual shape, then note the deviation in the commit message.)

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch_dcase_audio.sh
git commit -m "feat: DCASE2020 T2 fan+pump fetch script (MD5-verified, symlink layout)"
```

---

### Task 2: Filename parsing + dataset scan (pure, TDD)

**Files:**
- Create: `src/defectlens/audio/__init__.py` (empty)
- Create: `src/defectlens/audio/dataset.py`
- Test: `tests/test_audio_dataset.py`

DCASE filenames: `train/normal_id_00_00000000.wav`, `test/anomaly_id_02_00000042.wav`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_audio_dataset.py -q` — FAIL (module not found)

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Full suite** — `pytest -q`, all pass.

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/audio/ tests/test_audio_dataset.py
git commit -m "feat: DCASE audio filename parsing + machine dir scan"
```

---

### Task 3: kNN anomaly scorer (pure numpy, TDD)

**Files:**
- Create: `src/defectlens/audio/anomaly.py`
- Test: `tests/test_audio_anomaly.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from defectlens.audio.anomaly import KNNAnomalyScorer


def _cluster(center, n=50, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    embs = center + rng.normal(0, scale, size=(n, len(center)))
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def test_far_points_score_higher_than_near_points():
    normal = _cluster(np.array([1.0, 0.0, 0.0]))
    scorer = KNNAnomalyScorer(k=5).fit(normal)
    near = _cluster(np.array([1.0, 0.0, 0.0]), n=10, seed=1)
    far = _cluster(np.array([0.0, 1.0, 0.0]), n=10, seed=2)
    assert scorer.score(far).min() > scorer.score(near).max()


def test_k_larger_than_fit_set_is_capped():
    normal = _cluster(np.array([1.0, 0.0, 0.0]), n=3)
    scorer = KNNAnomalyScorer(k=10).fit(normal)
    assert np.isfinite(scorer.score(normal)).all()


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNAnomalyScorer().score(np.zeros((1, 3)))
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_audio_anomaly.py -q`

- [ ] **Step 3: Implement**

```python
"""k-NN anomaly scoring over normalized embeddings (DCASE unsupervised protocol).

Fit on NORMAL clips only; score = mean cosine distance to the k nearest
normal embeddings. Higher = more anomalous. This embeddings+density shape is
the modern replacement for the DCASE AE-reconstruction baseline.
"""
from __future__ import annotations

import numpy as np


class KNNAnomalyScorer:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._bank: np.ndarray | None = None

    def fit(self, normal_embeddings: np.ndarray) -> "KNNAnomalyScorer":
        self._bank = np.asarray(normal_embeddings, dtype=np.float32)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self._bank is None:
            raise RuntimeError("fit() before score()")
        emb = np.asarray(embeddings, dtype=np.float32)
        # cosine distance on L2-normalized vectors: 1 - dot
        sims = emb @ self._bank.T                     # [n, bank]
        k = min(self.k, self._bank.shape[0])
        top = np.sort(sims, axis=1)[:, -k:]           # k most-similar normals
        return 1.0 - top.mean(axis=1)
```

- [ ] **Step 4: Full suite** — `pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/audio/anomaly.py tests/test_audio_anomaly.py
git commit -m "feat: kNN anomaly scorer over normalized audio embeddings"
```

---

### Task 4: CLAP embedding wrapper

**Files:**
- Create: `src/defectlens/audio/embed.py`
- Test: `tests/test_audio_embed.py` (pure parts only; no model download in tests)

- [ ] **Step 1: Write the failing test (resample math + batching contract)**

```python
import numpy as np

from defectlens.audio.embed import CLAP_MODEL, batched


def test_batched_covers_all_items_in_order():
    items = list(range(10))
    batches = list(batched(items, size=4))
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_clap_model_constant():
    assert CLAP_MODEL == "laion/clap-htsat-unfused"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_audio_embed.py -q`

- [ ] **Step 3: Implement**

```python
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
```

Check `pip show torchaudio` first; if missing: `pip install torchaudio` and add `torchaudio` to pyproject's dependencies (it is a real runtime dep of the audio path).

- [ ] **Step 4: Full suite** — `pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/audio/embed.py tests/test_audio_embed.py pyproject.toml
git commit -m "feat: CLAP audio embedding wrapper (48k resample, normalized)"
```

---

### Task 5: Per-machine-ID AUC eval vs DCASE baseline

**Files:**
- Create: `src/defectlens/eval/audio_auc.py`
- Test: `tests/test_audio_auc.py` (pure aggregation only)

- [ ] **Step 1: Fetch the official per-ID baseline numbers**

WebFetch (or read) `https://arxiv.org/abs/2006.05822` (DCASE2020 T2 baseline paper) Table A.1 dev-set AUC per machine ID for fan and pump. Record them as the `BASELINE_AUC` dict in the module below (the averages must come out to fan 65.83 / pump 72.89 as published on dcase.community — if they do not, re-check; do NOT invent numbers).

- [ ] **Step 2: Write the failing test**

```python
from defectlens.eval.audio_auc import results_table


def test_results_table_shapes_and_beat_flags():
    ours = {"fan": {"00": 0.70}, "pump": {"00": 0.60}}
    baseline = {"fan": {"00": 0.65}, "pump": {"00": 0.65}}
    table = results_table(ours, baseline)
    assert table["fan"]["00"] == {"auc": 0.70, "baseline": 0.65, "beats_baseline": True}
    assert table["pump"]["00"]["beats_baseline"] is False
```

- [ ] **Step 3: Implement the module**

```python
"""Per-machine-ID audio anomaly AUC on the DCASE 2020 T2 dev set.

Protocol (matches the challenge): per machine type + ID, fit the scorer on
that ID's TRAIN normals only, score its TEST clips, AUC against the
normal/anomaly labels. Compares against the published AE baseline
(Koizumi et al. 2020, arXiv:2006.05822).

Usage: python -m defectlens.eval.audio_auc            # both machines
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BASELINE_AUC: dict[str, dict[str, float]] = {
    # Filled from arXiv:2006.05822 Table A.1 in Task 5 Step 1 - per machine ID.
    # "fan": {"00": 0.5441, "02": 0.7340, "04": 0.6161, "06": 0.7392},
    # "pump": {"00": 0.6715, "02": 0.6153, "04": 0.8833, "06": 0.7455},
}


def results_table(ours: dict, baseline: dict) -> dict:
    table: dict = {}
    for machine, ids in ours.items():
        table[machine] = {}
        for mid, auc in ids.items():
            base = baseline.get(machine, {}).get(mid)
            table[machine][mid] = {
                "auc": auc,
                "baseline": base,
                "beats_baseline": (base is not None and auc > base),
            }
    return table


def main(argv: list[str] | None = None) -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    from defectlens.audio.anomaly import KNNAnomalyScorer
    from defectlens.audio.dataset import scan_machine_dir
    from defectlens.audio.embed import embed_audio_files, load_clap
    from defectlens.eval.clip_zeroshot import pick_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/audio"))
    parser.add_argument("--machines", nargs="+", default=["fan", "pump"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/audio_auc.json"))
    args = parser.parse_args(argv)

    device = pick_device()
    model, processor = load_clap(device)

    ours: dict = {}
    for machine in args.machines:
        rows = scan_machine_dir(args.audio_root / machine, machine=machine)
        by_id = defaultdict(lambda: {"train": [], "test": []})
        for r in rows:
            by_id[r.machine_id][r.split].append(r)
        ours[machine] = {}
        for mid, splits in tqdm(sorted(by_id.items()), desc=machine):
            train_paths = [Path(r.path) for r in splits["train"] if r.label == "normal"]
            test_rows = splits["test"]
            train_embs = embed_audio_files(model, processor, train_paths, device, args.batch_size)
            test_embs = embed_audio_files(
                model, processor, [Path(r.path) for r in test_rows], device, args.batch_size
            )
            scorer = KNNAnomalyScorer(k=args.k).fit(train_embs)
            scores = scorer.score(test_embs)
            y_true = np.array([r.label == "anomaly" for r in test_rows], dtype=int)
            ours[machine][mid] = float(roc_auc_score(y_true, scores))

    table = results_table(ours, BASELINE_AUC)
    payload = {
        "method": "CLAP (laion/clap-htsat-unfused) embeddings + kNN(k=%d) on train normals" % args.k,
        "protocol": "DCASE 2020 Task 2 dev set, per-machine-ID, unsupervised",
        "baseline": "AE baseline, Koizumi et al. 2020 (arXiv:2006.05822)",
        "results": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for machine, ids in table.items():
        aucs = [v["auc"] for v in ids.values()]
        print(f"{machine}: avg AUC {sum(aucs)/len(aucs):.4f}")
        for mid, v in sorted(ids.items()):
            beat = "BEATS" if v["beats_baseline"] else "below"
            print(f"  id_{mid}: {v['auc']:.4f} vs baseline {v['baseline']} ({beat})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Unit tests + full suite** — `pytest -q`.

- [ ] **Step 5: Commit the harness**

```bash
git add src/defectlens/eval/audio_auc.py tests/test_audio_auc.py
git commit -m "feat: per-machine-ID audio AUC eval vs DCASE2020 baseline"
```

---

### Task 6: Run the eval + README + merge (controller)

- [ ] **Step 1:** `python -m defectlens.eval.audio_auc` (~20-35 min MPS: ~9,300 clips embedded once). Check memory headroom first (`memory_pressure -Q`).
- [ ] **Step 2:** README: new "Audio anomaly detection (Phase 5.2)" section - method sentence (embeddings+kNN on normals, DCASE protocol), per-ID table from results/audio_auc.json, DCASE/MIMII attribution + CC BY-NC-SA note, honest framing sentence ("industrial-equipment audio benchmark, HVAC-motivated; real HVAC acoustics differ").
- [ ] **Step 3:** Full suites, commit, merge `feat/phase5-audio` to main per seam discipline.

## Self-review

Spec coverage: decision 4 fully (unsupervised, CLAP, kNN/GMM-class density, DCASE-faithful, honest framing); decision 5's product half deferred to Plan 5.3 by design. De-scope trigger armed: if avg AUC lands below baseline after one honest iteration (e.g., k sweep 2/5/10 on ONE machine - allowed as the "one iteration"), report as-is and move on.
Placeholders: BASELINE_AUC is deliberately fetch-then-fill with the verification rule stated (averages must reconcile to published 65.83/72.89) - not a TBD, a guarded data-entry step.
Type consistency: AudioRow fields used identically in dataset/eval; embed returns L2-normalized np arrays consumed by both scorer and (later) retrieval.
