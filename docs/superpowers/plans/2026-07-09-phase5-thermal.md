# Phase 5.6: BFDD Thermal Fourth Modality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thermal (IR) as DefectLens's fourth modality via a controlled three-way segmentation comparison (RGB-only vs IR-only vs RGB+IR early fusion) on the BFDD facade-defect dataset, demonstrating what the thermal channel contributes.

**Architecture:** Offline analysis phase (like Phases 1-2), NOT wired into serving. A small `src/defectlens/thermal/` module handles pair listing, a frozen seed-42 split, and mask semantics; one training script fine-tunes SegFormer-b0 per input variant on Apple-Silicon MPS; results land in `results/thermal_bfdd.json`, a comparison figure in `docs/images/`, and an honest README section.

**Tech Stack:** PyTorch MPS, HuggingFace `transformers` (SegformerForSemanticSegmentation, `nvidia/mit-b0`), PIL/numpy, matplotlib (figure), pytest.

**Dataset facts (verified in-session 2026-07-09):**
- Location: `~/datasets/bfdd/Dataset_1x/{RGB,IR,Label,Label_color,Label_backup_7classes_20260125}` — 838 files each.
- Tarball: `~/datasets/bfdd/bfdd.tar.gz`, sha256 `43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a` (matches the Mendeley API's recorded hash).
- `RGB/<stem>.JPG` (RGB-mode, 640×512), `IR/<stem>.png` (RGB-mode PNG, 640×512, pixel-aligned), `Label/<stem>.png` (L-mode, ids 0-5), `Label_color/` (colorized masks for legend verification). Stems match across folders (e.g. `DJI_20250624181809_0003`).
- Pixel distribution over all 838 labels: id0 91.96%, id1 1.70%, id2 0.59%, id3 1.11%, id4 2.15%, id5 2.49% — heavy imbalance; report per-class IoU, never accuracy.
- License CC BY 4.0. Paper class set: Cracks, Peeling, Hollow Areas, Stains, Erosion. **The id→name order is NOT yet verified** — Task 2 resolves it before any results are labeled.
- Ignore `Label_backup_7classes_20260125/` (superseded annotation round).

**Honest-reporting mandate:** whatever direction the IR-vs-RGB numbers go — including "IR helps nowhere" — gets reported as measured. The portfolio story is the controlled comparison, not a predetermined conclusion.

---

### Task 1: Dataset module — pairs, frozen split, mask loading

**Files:**
- Create: `src/defectlens/thermal/__init__.py` (empty)
- Create: `src/defectlens/thermal/bfdd.py`
- Create: `scripts/fetch_bfdd.sh`
- Test: `tests/test_bfdd_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
"""BFDD dataset module: pair listing, frozen split, mask loading.

Real-data tests run only when ~/datasets/bfdd exists (CI skips them);
split determinism is locked with a synthetic stem list so it always runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from defectlens.thermal.bfdd import (
    BFDD_ROOT,
    CLASS_IDS,
    BfddPair,
    list_pairs,
    split_stems,
)

HAVE_DATA = BFDD_ROOT.exists()


def test_split_stems_is_deterministic_and_disjoint():
    stems = [f"img_{i:04d}" for i in range(100)]
    a = split_stems(stems, seed=42)
    b = split_stems(list(reversed(stems)), seed=42)  # order-insensitive
    assert a == b
    assert len(a["test"]) == 15 and len(a["val"]) == 15 and len(a["train"]) == 70
    assert not (set(a["train"]) & set(a["val"])) and not (set(a["val"]) & set(a["test"]))
    assert not (set(a["train"]) & set(a["test"]))


def test_split_stems_regression_lock_first_members():
    """Frozen-split discipline (Phase 1 convention): the seed-42 split of the
    real 838 stems must never silently change. Locks the first member of each
    bucket computed at plan time."""
    stems = [f"s{i}" for i in range(838)]
    s = split_stems(stems, seed=42)
    # computed once at implementation time and hard-coded here:
    assert s["test"][0] == sorted(s["test"])[0]  # placeholder REPLACED in Step 3
    # implementer: run split_stems on the synthetic 838 stems, paste the
    # actual first-member literals for train/val/test into this test.


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_list_pairs_finds_838_complete_pairs():
    pairs = list_pairs()
    assert len(pairs) == 838
    p = pairs[0]
    assert isinstance(p, BfddPair)
    assert p.rgb.exists() and p.ir.exists() and p.label.exists()
    assert p.rgb.suffix == ".JPG" and p.ir.suffix == ".png"


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_load_mask_values_within_class_ids():
    import numpy as np

    from defectlens.thermal.bfdd import load_mask

    pairs = list_pairs()
    m = load_mask(pairs[0].label)
    assert m.dtype == np.int64 and m.shape == (512, 640)
    assert set(np.unique(m)) <= set(CLASS_IDS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bfdd_dataset.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'defectlens.thermal'`

- [ ] **Step 3: Implement the module**

```python
"""BFDD (Building Facade Defect Dataset) access: pairs, frozen split, masks.

BFDD: 838 pixel-aligned RGB+IR facade image pairs with 6-class (background +
5 defect) segmentation masks, 640x512, CC BY 4.0.
Source: https://data.mendeley.com/datasets/9ych7czvyg/1 (fetch via
scripts/fetch_bfdd.sh). CLASS_NAMES provenance is documented in
docs/datasets.md (verified against Label_color + the dataset description).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BFDD_ROOT = Path.home() / "datasets" / "bfdd" / "Dataset_1x"

CLASS_IDS = (0, 1, 2, 3, 4, 5)
# PROVISIONAL until Task 2 verifies the id->name order against Label_color
# and the Mendeley description; Task 2 replaces this dict and this comment.
CLASS_NAMES = {
    0: "background",
    1: "class1",
    2: "class2",
    3: "class3",
    4: "class4",
    5: "class5",
}

VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass(frozen=True)
class BfddPair:
    stem: str
    rgb: Path
    ir: Path
    label: Path


def list_pairs(root: Path = BFDD_ROOT) -> list[BfddPair]:
    """All complete RGB/IR/Label triples, sorted by stem (deterministic)."""
    pairs = []
    for lab in sorted((root / "Label").glob("*.png")):
        stem = lab.stem
        rgb = root / "RGB" / f"{stem}.JPG"
        ir = root / "IR" / f"{stem}.png"
        if rgb.exists() and ir.exists():
            pairs.append(BfddPair(stem=stem, rgb=rgb, ir=ir, label=lab))
    return pairs


def split_stems(stems: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Frozen 70/15/15 split. Sorts first so input order can't leak in."""
    ordered = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_test = round(n * TEST_FRAC)
    n_val = round(n * VAL_FRAC)
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test : n_test + n_val],
        "train": ordered[n_test + n_val :],
    }


def split_pairs(pairs: list[BfddPair], seed: int = 42) -> dict[str, list[BfddPair]]:
    buckets = split_stems([p.stem for p in pairs], seed=seed)
    member = {s: k for k, ss in buckets.items() for s in ss}
    out: dict[str, list[BfddPair]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        out[member[p.stem]].append(p)
    return out


def load_mask(path: Path) -> np.ndarray:
    """L-mode PNG -> int64 (H, W) class-id array."""
    return np.array(Image.open(path), dtype=np.int64)
```

Also fix the regression-lock test: run `split_stems([f"s{i}" for i in range(838)], seed=42)` once, paste the literal first members of train/val/test into `test_split_stems_regression_lock_first_members` (replacing the placeholder assert), so the frozen split is locked against accidental change.

`scripts/fetch_bfdd.sh` (bash-3.2-safe, no `declare -A`; `set -euo pipefail`):

```bash
#!/usr/bin/env bash
# Fetch BFDD (Building Facade Defect Dataset, CC BY 4.0) to ~/datasets/bfdd.
# https://data.mendeley.com/datasets/9ych7czvyg/1
set -euo pipefail

DEST="${HOME}/datasets/bfdd"
URL="https://data.mendeley.com/public-files/datasets/9ych7czvyg/files/c1c5144b-cb20-4687-b514-d0bbec12209e/file_downloaded"
SHA="43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a"

mkdir -p "${DEST}"
cd "${DEST}"
if [ ! -f bfdd.tar.gz ]; then
  echo "Downloading BFDD (~528 MB)..."
  curl -sL -o bfdd.tar.gz "${URL}"
fi
echo "${SHA}  bfdd.tar.gz" | shasum -a 256 -c -
[ -d Dataset_1x ] || tar -xzf bfdd.tar.gz
echo "OK: ${DEST}/Dataset_1x"
```

`chmod +x scripts/fetch_bfdd.sh`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bfdd_dataset.py -q`
Expected: 4 passed (or 2 passed + 2 skipped without local data)

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/thermal/ tests/test_bfdd_dataset.py scripts/fetch_bfdd.sh
git commit -m "feat: BFDD dataset module - pairs, frozen seed-42 split, masks"
```

---

### Task 2: Verify the class-id → name mapping (evidence, not assumption)

**Files:**
- Modify: `src/defectlens/thermal/bfdd.py` (CLASS_NAMES dict + comment)
- Modify: `docs/datasets.md` (BFDD section with the evidence)
- Test: `tests/test_bfdd_dataset.py` (name coverage)

- [ ] **Step 1: Gather evidence**

Fetch the dataset description (title, description text, any file manifest notes):
`curl -s "https://data.mendeley.com/public-api/datasets/9ych7czvyg/versions" ; curl -s "https://data.mendeley.com/datasets/9ych7czvyg/1" | python3 -c "import sys,html,re; t=sys.stdin.read(); m=re.search(r'<meta name=\"description\" content=\"([^\"]+)', t); print(html.unescape(m.group(1)) if m else t[:2000])"`

Then cross-check colors: for 3-4 images whose `Label` contains a given id, open the matching `Label_color` PNG and record which color that id renders as; match colors/regions against defect appearance in the RGB image (cracks are thin dark lines; peeling shows flaking paint; stains are broad discolorations; hollow areas are only visible in IR as thermal contrast; erosion shows material loss). Example probe:

```python
import glob, numpy as np
from PIL import Image
for f in sorted(glob.glob(str(BFDD_ROOT / "Label/*.png")))[:60]:
    ids = set(np.unique(np.array(Image.open(f)))) - {0}
    if ids:
        print(f, ids)  # pick images with a single defect id, view RGB+Label_color
```

View the picked images (Read tool renders PNGs/JPGs) and decide the mapping. If the evidence is ambiguous for an id, name it `"unverified_<id>"` and say so in docs — do NOT guess silently.

- [ ] **Step 2: Update CLASS_NAMES + document**

Replace the provisional `CLASS_NAMES` in `src/defectlens/thermal/bfdd.py` with the verified names (snake_case, e.g. `"hollow_area"`), and replace the PROVISIONAL comment with one line citing the evidence ("verified 2026-07-09 against Label_color legend + Mendeley description; see docs/datasets.md"). Add a BFDD section to `docs/datasets.md`: source URL, license, fetch script, the id→name table, and 1-2 sentences of evidence per id.

- [ ] **Step 3: Add the coverage test**

```python
def test_class_names_cover_all_ids_and_are_not_placeholders():
    from defectlens.thermal.bfdd import CLASS_IDS, CLASS_NAMES

    assert set(CLASS_NAMES) == set(CLASS_IDS)
    assert CLASS_NAMES[0] == "background"
    for cid in CLASS_IDS[1:]:
        assert not CLASS_NAMES[cid].startswith("class"), "provisional name left in"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_bfdd_dataset.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/thermal/bfdd.py docs/datasets.md tests/test_bfdd_dataset.py
git commit -m "feat: verified BFDD class-id mapping with documented evidence"
```

---

### Task 3: SegFormer training/eval script with rgb | ir | rgbir variants

**Files:**
- Create: `src/defectlens/thermal/train_seg.py`
- Test: `tests/test_thermal_train.py`

- [ ] **Step 1: Write the failing tests (pure logic only — no training)**

```python
"""Variant wiring for the BFDD SegFormer comparison. No real training here."""
from __future__ import annotations

import numpy as np
import torch

from defectlens.thermal.train_seg import (
    VARIANT_CHANNELS,
    build_model,
    compose_input,
    iou_from_confusion,
)


def test_variant_channels():
    assert VARIANT_CHANNELS == {"rgb": 3, "ir": 3, "rgbir": 6}


def test_compose_input_shapes_and_variant_selection():
    rgb = np.zeros((512, 640, 3), dtype=np.uint8)
    ir = np.full((512, 640, 3), 255, dtype=np.uint8)
    x_rgb = compose_input(rgb, ir, "rgb")
    x_ir = compose_input(rgb, ir, "ir")
    x_fused = compose_input(rgb, ir, "rgbir")
    assert x_rgb.shape == (3, 512, 640) and x_fused.shape == (6, 512, 640)
    # normalized IR (all-255) has strictly larger mean than all-0 rgb
    assert x_ir.mean() > x_rgb.mean()
    # fusion stacks rgb first, ir second
    assert torch.allclose(x_fused[:3], x_rgb) and torch.allclose(x_fused[3:], x_ir)


def test_build_model_in_channels_and_labels():
    m3 = build_model("ir", num_labels=6)
    m6 = build_model("rgbir", num_labels=6)
    assert m3.config.num_channels == 3 and m6.config.num_channels == 6
    assert m3.config.num_labels == 6
    x = torch.zeros(1, 6, 128, 160)
    out = m6(pixel_values=x)
    assert out.logits.shape[1] == 6  # (B, num_labels, H/4, W/4)


def test_iou_from_confusion_known_values():
    conf = np.array([[2, 1], [1, 2]], dtype=np.int64)
    ious = iou_from_confusion(conf)
    assert np.allclose(ious, [2 / 4, 2 / 4])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_thermal_train.py -q`
Expected: FAIL with `No module named 'defectlens.thermal.train_seg'`

- [ ] **Step 3: Implement**

```python
"""Fine-tune SegFormer-b0 on BFDD for the rgb | ir | rgbir comparison.

Usage (from repo root, MPS):
  .venv/bin/python -m defectlens.thermal.train_seg --variant ir \
      --epochs 25 --batch-size 4 --output-dir models/thermal_bfdd/ir

Writes <output-dir>/metrics.json: per-class IoU on the frozen test split,
plus config. Weights stay out of git (models/ is gitignored).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defectlens.thermal.bfdd import (
    CLASS_IDS,
    CLASS_NAMES,
    BfddPair,
    frozen_split_pairs,
    load_mask,
)

VARIANT_CHANNELS = {"rgb": 3, "ir": 3, "rgbir": 6}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _norm(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def compose_input(rgb: np.ndarray, ir: np.ndarray, variant: str) -> torch.Tensor:
    """(H,W,3) uint8 arrays -> normalized CHW float tensor per variant."""
    if variant == "rgb":
        x = _norm(rgb)
    elif variant == "ir":
        x = _norm(ir)
    elif variant == "rgbir":
        x = np.concatenate([_norm(rgb), _norm(ir)], axis=-1)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def build_model(variant: str, num_labels: int = len(CLASS_IDS)):
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_labels,
        num_channels=VARIANT_CHANNELS[variant],
        ignore_mismatched_sizes=True,  # 6-ch stem + fresh head re-init
    )


class BfddSegDataset(Dataset):
    def __init__(self, pairs: list[BfddPair], variant: str):
        self.pairs = pairs
        self.variant = variant

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        rgb = np.array(Image.open(p.rgb).convert("RGB"))
        ir = np.array(Image.open(p.ir).convert("RGB"))
        x = compose_input(rgb, ir, self.variant)
        y = torch.from_numpy(load_mask(p.label))
        return x, y


def iou_from_confusion(conf: np.ndarray) -> np.ndarray:
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - np.diag(conf)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


@torch.no_grad()
def evaluate(model, loader, device, num_labels: int) -> np.ndarray:
    model.eval()
    conf = np.zeros((num_labels, num_labels), dtype=np.int64)
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy().ravel()
        gt = y.numpy().ravel()
        conf += np.bincount(gt * num_labels + pred, minlength=num_labels**2).reshape(
            num_labels, num_labels
        )
    return conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANT_CHANNELS), required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="train on N pairs (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    buckets = frozen_split_pairs()  # authoritative committed manifest, NOT args.seed
    train_pairs = buckets["train"][: args.subset or None]
    num_labels = len(CLASS_IDS)

    model = build_model(args.variant, num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        BfddSegDataset(train_pairs, args.variant),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        BfddSegDataset(buckets["test"], args.variant), batch_size=args.batch_size
    )

    step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            logits = model(pixel_values=x.to(device)).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    conf = evaluate(model, test_loader, device, num_labels)
    ious = iou_from_confusion(conf)
    defect_ids = [c for c in CLASS_IDS if c != 0]
    metrics = {
        "variant": args.variant,
        "epochs": args.epochs,
        "steps": step,
        "train_pairs": len(train_pairs),
        "test_pairs": len(buckets["test"]),
        "per_class_iou": {CLASS_NAMES[c]: float(ious[c]) for c in CLASS_IDS},
        "mean_defect_iou": float(np.nanmean([ious[c] for c in defect_ids])),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save_pretrained(args.output_dir / "checkpoint")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_thermal_train.py -q`
Expected: 4 passed (downloads ~14MB mit-b0 weights on first run)

- [ ] **Step 5: Smoke run (2 minutes, proves the loop end-to-end)**

Run: `.venv/bin/python -m defectlens.thermal.train_seg --variant rgbir --subset 8 --max-steps 4 --epochs 1 --output-dir /tmp/bfdd-smoke`
Expected: loss prints, `/tmp/bfdd-smoke/metrics.json` exists with all 6 class names.

- [ ] **Step 6: Commit**

```bash
git add src/defectlens/thermal/train_seg.py tests/test_thermal_train.py
git commit -m "feat: SegFormer-b0 BFDD training with rgb/ir/rgbir variants"
```

---

### Task 4: The three full runs + consolidated results

**Files:**
- Create: `results/thermal_bfdd.json`
- Create: `scripts/run_thermal_comparison.sh`

- [ ] **Step 1: Runner script**

```bash
#!/usr/bin/env bash
# Full rgb / ir / rgbir comparison. ~30-45 min per run on M3 Pro MPS.
set -euo pipefail
cd "$(dirname "$0")/.."
for v in rgb ir rgbir; do
  echo "=== variant: $v ==="
  .venv/bin/python -m defectlens.thermal.train_seg \
    --variant "$v" --epochs 25 --batch-size 4 \
    --output-dir "models/thermal_bfdd/$v"
done
.venv/bin/python - <<'EOF'
import json, pathlib
out = {v: json.loads((pathlib.Path("models/thermal_bfdd")/v/"metrics.json").read_text())
       for v in ("rgb", "ir", "rgbir")}
pathlib.Path("results/thermal_bfdd.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF
```

`chmod +x scripts/run_thermal_comparison.sh`.

- [ ] **Step 2: Run it (foreground background-task, ~1.5-2.5h total)**

Run: `scripts/run_thermal_comparison.sh 2>&1 | tee /tmp/thermal_runs.log`
Watch for: loss decreasing within each run; metrics.json per variant; final consolidated JSON. If MPS OOMs, drop `--batch-size` to 2 (adjust the script) — do NOT silently downscale images.

- [ ] **Step 3: Sync weights + results to S3 (weights stay off GitHub)**

Run: `aws s3 sync models/thermal_bfdd s3://defectlens-phase3-ca-002559670021/phase5/thermal/ --profile defectlens --exclude "*" --include "*/metrics.json" --include "*/checkpoint/*"`
(S3 storage cost: pennies; same convention as the audio bank.)

- [ ] **Step 4: Commit results**

```bash
git add results/thermal_bfdd.json scripts/run_thermal_comparison.sh
git commit -m "feat: BFDD rgb/ir/rgbir comparison results"
```

---

### Task 5: Comparison figure

**Files:**
- Create: `scripts/make_thermal_figure.py`
- Create: `docs/images/thermal-comparison.png` (generated)

- [ ] **Step 1: Implement the figure script**

```python
"""Comparison figure: RGB | IR | ground truth | RGB-model pred | IR-model pred
for 2-3 test images that contain the class where IR and RGB diverge most
(per results/thermal_bfdd.json). Requires the trained checkpoints from Task 4.

Usage: .venv/bin/python scripts/make_thermal_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from defectlens.thermal.bfdd import CLASS_IDS, CLASS_NAMES, frozen_split_pairs, list_pairs, load_mask
from defectlens.thermal.train_seg import compose_input

PALETTE = np.array(
    [[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48]],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def predict(model_dir: Path, x: torch.Tensor, size) -> np.ndarray:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).eval()
    with torch.no_grad():
        logits = model(pixel_values=x.unsqueeze(0)).logits
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].numpy()


def main() -> None:
    results = json.loads(Path("results/thermal_bfdd.json").read_text())
    # pick the defect class with the largest |IR - RGB| IoU gap as the story class
    gaps = {
        name: abs(results["ir"]["per_class_iou"][name] - results["rgb"]["per_class_iou"][name])
        for name in results["ir"]["per_class_iou"]
        if name != "background"
    }
    story = max(gaps, key=gaps.get)
    story_id = next(c for c in CLASS_IDS if CLASS_NAMES[c] == story)
    print(f"story class: {story} (IoU gap {gaps[story]:.3f})")

    test_pairs = frozen_split_pairs()["test"]
    picks = [p for p in test_pairs if story_id in np.unique(load_mask(p.label))][:3]

    fig, axes = plt.subplots(len(picks), 5, figsize=(16, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(picks):
        rgb = np.array(Image.open(p.rgb).convert("RGB"))
        ir = np.array(Image.open(p.ir).convert("RGB"))
        gt = load_mask(p.label)
        pred_rgb = predict(Path("models/thermal_bfdd/rgb/checkpoint"), compose_input(rgb, ir, "rgb"), gt.shape)
        pred_ir = predict(Path("models/thermal_bfdd/ir/checkpoint"), compose_input(rgb, ir, "ir"), gt.shape)
        for c, (img, title) in enumerate(
            [
                (rgb, "RGB"),
                (ir, "IR (thermal)"),
                (colorize(gt), "ground truth"),
                (colorize(pred_rgb), "RGB-only pred"),
                (colorize(pred_ir), "IR-only pred"),
            ]
        ):
            axes[r, c].imshow(img)
            axes[r, c].set_title(title if r == 0 else "", fontsize=11)
            axes[r, c].axis("off")
    fig.suptitle(f"BFDD: where the thermal channel diverges ({story})", fontsize=13)
    fig.tight_layout()
    fig.savefig("docs/images/thermal-comparison.png", dpi=120, bbox_inches="tight")
    print("wrote docs/images/thermal-comparison.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and inspect**

Run: `.venv/bin/python scripts/make_thermal_figure.py`
Then view `docs/images/thermal-comparison.png` (Read tool) — the figure must be legible, rows show the story class, no matplotlib artifacts. Pixel-perfection bar applies.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_thermal_figure.py docs/images/thermal-comparison.png
git commit -m "feat: thermal comparison figure"
```

---

### Task 6: README section + suite green + merge prep

**Files:**
- Modify: `README.md` (new collapsible Phase 5.6 section + metrics strip if warranted)

- [ ] **Step 1: Write the README section**

Follow the existing collapsible `<details>` pattern (see the Phase 5 sections). Contents, in order: what was built (three-way controlled comparison, same split/model/epochs, only the input differs); the per-class IoU table for rgb/ir/rgbir; the figure; honest caveats verbatim-style: 838 images from one publication (limited scene diversity), id→name mapping provenance (cite docs/datasets.md), class imbalance (report per-class IoU, background 92%), and the measured direction of the IR effect WHATEVER it is — if IR helped nowhere, the section says so and the "fourth modality" framing becomes "evaluated with a negative result", which is still a valid portfolio story. Update the top-of-README modality sentence ("photo + note + audio" -> "+ thermal (offline comparison)") ONLY if IR or fusion produced a real win; otherwise leave the top intact and let the section speak.

- [ ] **Step 2: Full suite green**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (DB skips OK). Also `cd frontend && CI=true npx react-scripts test --watchAll=false` if any frontend file was touched (it should NOT be in this phase).

- [ ] **Step 3: Commit + push branch**

```bash
git add README.md
git commit -m "docs: Phase 5.6 thermal comparison section"
git push -u origin feat/phase5-thermal
```

Merge to main only with explicit user authorization (permission-gated).
