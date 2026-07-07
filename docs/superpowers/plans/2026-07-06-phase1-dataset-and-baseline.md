# Phase 1: Dataset Unification + CLIP Zero-Shot Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge four public building-defect datasets into one 9-class taxonomy with a frozen, committed test split, and measure the CLIP ViT-L/14 zero-shot baseline (macro top-1/top-3 + confusion matrix) on that split.

**Architecture:** A `src/defectlens` Python package: a versioned label-mapping table (`configs/label_mapping.yaml`) drives a generic folder-scan ingest into a CSV manifest; a seeded stratified splitter freezes train/test manifests (committed); a CLIP zero-shot evaluator with prompt ensembling produces the baseline metrics that Phase 3's fine-tune must beat. Raw images are normalized into a canonical `data/raw/<dataset>/<source_label>/` layout by a one-time script, so ingest code never depends on upstream zip structures.

**Tech Stack:** Python 3.11+, PyTorch (MPS), HuggingFace `transformers` (CLIP ViT-L/14), PyYAML, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-defect-lens-design.md` (§4 dataset, §5 baseline)

**Dataset sources (preflight-verified 2026-07-06):**

| Dataset | Source | License | Required? |
|---|---|---|---|
| CODEBRIM | https://zenodo.org/records/2620293 → `CODEBRIM_classification_dataset.zip` | non-commercial/educational | yes |
| BD3 | https://github.com/samy101/bd3-building-defects-detection-dataset | see repo | yes |
| SDNET2018 | https://digitalcommons.usu.edu/all_datasets/48/ → `SDNET2018.zip` (~504 MB) | CC-BY-4.0 | yes |
| Roboflow wall-defects | https://universe.roboflow.com/builddef2/building-defect-on-walls (free account + API key) | check page | **optional** — classes already covered by BD3 |

---

## Task 1: Repo scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/defectlens/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create the project files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "defectlens"
version = "0.1.0"
description = "Building-defect inspection assistant — dataset pipeline, eval, serving"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "transformers>=4.49",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "tqdm>=4.66",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
*.egg-info/
.DS_Store
data/raw/
review_grid/
```

(Note: `data/manifests/` and `results/` are **committed** — they are the frozen-split record and the reported numbers.)

`README.md`:

```markdown
# DefectLens

Building-defect inspection assistant: photo → fine-grained defect ID + severity
framing + retrieved remediation guidance and standards citations.

Design spec: `docs/superpowers/specs/2026-07-06-defect-lens-design.md`

## Status

Phase 1 (dataset unification + CLIP zero-shot baseline) — in progress.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
```

`src/defectlens/__init__.py`:

```python
"""DefectLens: building-defect dataset pipeline, evaluation, and serving."""
```

`tests/test_smoke.py`:

```python
def test_import():
    import defectlens  # noqa: F401
```

- [ ] **Step 2: Create venv, install, run tests**

Run:
```bash
cd /Users/hangruan/Documents/GitHub/defect-lens
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```
Expected: `test_import PASSED`, 1 passed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml .gitignore README.md src tests
git commit -m "chore: scaffold defectlens package"
```

---

## Task 2: Taxonomy + versioned label-mapping table

**Files:**
- Create: `configs/label_mapping.yaml`
- Create: `src/defectlens/taxonomy.py`
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_taxonomy.py`:

```python
from pathlib import Path

import pytest

from defectlens.taxonomy import UNIFIED_CLASSES, load_mapping, map_label

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MAPPING = REPO_ROOT / "configs" / "label_mapping.yaml"


def write_mapping(tmp_path, text):
    p = tmp_path / "mapping.yaml"
    p.write_text(text)
    return p


def test_unified_classes_are_nine():
    assert len(UNIFIED_CLASSES) == 9
    assert "no_defect" in UNIFIED_CLASSES


def test_load_and_map(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: biological growth
""",
    )
    mapping = load_mapping(p)
    assert map_label(mapping, "bd3", "algae") == "mold_algae"


def test_exclude_returns_none(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: junk
    unified_label: EXCLUDE
    rationale: not a defect class
""",
    )
    mapping = load_mapping(p)
    assert map_label(mapping, "bd3", "junk") is None


def test_unknown_unified_label_rejected(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: not_a_class
    rationale: typo
""",
    )
    with pytest.raises(ValueError, match="not_a_class"):
        load_mapping(p)


def test_duplicate_mapping_rejected(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: a
  - source_dataset: bd3
    source_label: algae
    unified_label: water_damage
    rationale: b
""",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_mapping(p)


def test_unmapped_label_raises(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: a
""",
    )
    mapping = load_mapping(p)
    with pytest.raises(KeyError):
        map_label(mapping, "bd3", "never_seen")


def test_real_mapping_file_is_valid_and_complete():
    mapping = load_mapping(REAL_MAPPING)
    expected_sources = {
        ("codebrim", "background"),
        ("codebrim", "crack"),
        ("codebrim", "spallation"),
        ("codebrim", "efflorescence"),
        ("codebrim", "exposed_bars"),
        ("codebrim", "corrosion_stain"),
        ("bd3", "algae"),
        ("bd3", "major_crack"),
        ("bd3", "minor_crack"),
        ("bd3", "peeling"),
        ("bd3", "spalling"),
        ("bd3", "stain"),
        ("bd3", "normal"),
        ("roboflow_walls", "crack"),
        ("roboflow_walls", "mold"),
        ("roboflow_walls", "peeling_paint"),
        ("roboflow_walls", "stairstep_crack"),
        ("roboflow_walls", "water_seepage"),
        ("sdnet2018", "cracked"),
        ("sdnet2018", "non_cracked"),
    }
    assert expected_sources.issubset(mapping.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defectlens.taxonomy'`

- [ ] **Step 3: Write the mapping config and taxonomy module**

`configs/label_mapping.yaml`:

```yaml
# DefectLens unified label-mapping table (versioned; every change gets a rationale).
# unified_label must be one of the 9 UNIFIED_CLASSES, or EXCLUDE to drop samples.
mappings:
  # ---- CODEBRIM (concrete bridge defects, single-label crops) ----
  - source_dataset: codebrim
    source_label: background
    unified_label: no_defect
    rationale: non-defective concrete background crops
  - source_dataset: codebrim
    source_label: crack
    unified_label: crack
    rationale: direct match
  - source_dataset: codebrim
    source_label: spallation
    unified_label: spalling
    rationale: same phenomenon, naming variant
  - source_dataset: codebrim
    source_label: efflorescence
    unified_label: efflorescence
    rationale: direct match
  - source_dataset: codebrim
    source_label: exposed_bars
    unified_label: exposed_rebar
    rationale: exposed reinforcement bars
  - source_dataset: codebrim
    source_label: corrosion_stain
    unified_label: corrosion_stain
    rationale: direct match

  # ---- BD3 (building facade defects) ----
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: biological growth; merged class per spec taxonomy
  - source_dataset: bd3
    source_label: major_crack
    unified_label: crack
    rationale: crack severity handled downstream, not as separate class
  - source_dataset: bd3
    source_label: minor_crack
    unified_label: crack
    rationale: crack severity handled downstream, not as separate class
  - source_dataset: bd3
    source_label: peeling
    unified_label: peeling_paint
    rationale: peeling paint/plaster on walls
  - source_dataset: bd3
    source_label: spalling
    unified_label: spalling
    rationale: direct match
  - source_dataset: bd3
    source_label: stain
    unified_label: water_damage
    rationale: BD3 stains are predominantly moisture/water staining; spot-check validates
  - source_dataset: bd3
    source_label: normal
    unified_label: no_defect
    rationale: non-defective wall surfaces

  # ---- Roboflow building-defect-on-walls (OPTIONAL dataset) ----
  - source_dataset: roboflow_walls
    source_label: crack
    unified_label: crack
    rationale: direct match
  - source_dataset: roboflow_walls
    source_label: mold
    unified_label: mold_algae
    rationale: merged class per spec taxonomy
  - source_dataset: roboflow_walls
    source_label: peeling_paint
    unified_label: peeling_paint
    rationale: direct match
  - source_dataset: roboflow_walls
    source_label: stairstep_crack
    unified_label: crack
    rationale: stair-step masonry crack is a crack subtype
  - source_dataset: roboflow_walls
    source_label: water_seepage
    unified_label: water_damage
    rationale: direct match

  # ---- SDNET2018 (concrete crack patches; C*=cracked, U*=uncracked) ----
  - source_dataset: sdnet2018
    source_label: cracked
    unified_label: crack
    rationale: binary crack dataset, crack side
  - source_dataset: sdnet2018
    source_label: non_cracked
    unified_label: no_defect
    rationale: abundant clean-concrete negatives; capped at ingest (configs/sampling.yaml)
```

`src/defectlens/taxonomy.py`:

```python
"""Unified defect taxonomy and the versioned source→unified label mapping."""
from __future__ import annotations

from pathlib import Path

import yaml

UNIFIED_CLASSES = [
    "crack",
    "spalling",
    "efflorescence",
    "exposed_rebar",
    "corrosion_stain",
    "mold_algae",
    "water_damage",
    "peeling_paint",
    "no_defect",
]

EXCLUDE = "EXCLUDE"

Mapping = dict[tuple[str, str], str]


def load_mapping(path: Path) -> Mapping:
    """Load and validate configs/label_mapping.yaml."""
    raw = yaml.safe_load(Path(path).read_text())
    mapping: Mapping = {}
    for entry in raw["mappings"]:
        key = (entry["source_dataset"], entry["source_label"])
        if key in mapping:
            raise ValueError(f"Duplicate mapping for {key}")
        unified = entry["unified_label"]
        if unified != EXCLUDE and unified not in UNIFIED_CLASSES:
            raise ValueError(f"Unknown unified label {unified!r} for {key}")
        mapping[key] = unified
    return mapping


def map_label(mapping: Mapping, source_dataset: str, source_label: str) -> str | None:
    """Return the unified label, or None if the sample is excluded.

    Raises KeyError for unmapped labels so new upstream labels surface loudly
    instead of being silently dropped.
    """
    key = (source_dataset, source_label)
    if key not in mapping:
        raise KeyError(
            f"No mapping for {key} — add it to configs/label_mapping.yaml"
        )
    unified = mapping[key]
    return None if unified == EXCLUDE else unified
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_taxonomy.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add configs/label_mapping.yaml src/defectlens/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: unified 9-class taxonomy with versioned label-mapping table"
```

---

## Task 3: Metrics (macro top-k accuracy, confusion matrix)

**Files:**
- Create: `src/defectlens/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
import math

from defectlens.metrics import (
    confusion_matrix,
    macro_topk_accuracy,
    per_class_topk_accuracy,
)

CLASSES = ["a", "b", "c"]


def test_per_class_top1():
    y_true = ["a", "a", "b"]
    ranked = [["a", "b", "c"], ["b", "a", "c"], ["b", "c", "a"]]
    per = per_class_topk_accuracy(y_true, ranked, CLASSES, k=1)
    assert per["a"] == 0.5
    assert per["b"] == 1.0
    assert math.isnan(per["c"])  # no samples of class c


def test_top3_hits_anywhere_in_top3():
    y_true = ["a"]
    ranked = [["c", "b", "a"]]
    per = per_class_topk_accuracy(y_true, ranked, CLASSES, k=3)
    assert per["a"] == 1.0


def test_macro_ignores_absent_classes():
    y_true = ["a", "a", "b"]
    ranked = [["a", "x", "x"], ["b", "x", "x"], ["b", "x", "x"]]
    # a: 1/2, b: 1/1, c: absent -> macro = (0.5 + 1.0) / 2
    assert macro_topk_accuracy(y_true, ranked, CLASSES, k=1) == 0.75


def test_confusion_matrix():
    y_true = ["a", "a", "b"]
    top1 = ["a", "b", "b"]
    m = confusion_matrix(y_true, top1, CLASSES)
    assert m[0][0] == 1  # a predicted a
    assert m[0][1] == 1  # a predicted b
    assert m[1][1] == 1  # b predicted b
    assert sum(sum(row) for row in m) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defectlens.metrics'`

- [ ] **Step 3: Implement metrics**

`src/defectlens/metrics.py`:

```python
"""Evaluation metrics: per-class / macro top-k accuracy, confusion matrix."""
from __future__ import annotations

import math
from collections import defaultdict


def per_class_topk_accuracy(
    y_true: list[str], ranked_preds: list[list[str]], classes: list[str], k: int
) -> dict[str, float]:
    """Accuracy per class; NaN for classes absent from y_true."""
    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for true, ranked in zip(y_true, ranked_preds, strict=True):
        totals[true] += 1
        if true in ranked[:k]:
            hits[true] += 1
    return {
        c: (hits[c] / totals[c]) if totals[c] else float("nan") for c in classes
    }


def macro_topk_accuracy(
    y_true: list[str], ranked_preds: list[list[str]], classes: list[str], k: int
) -> float:
    """Mean of per-class accuracies over classes that appear in y_true."""
    per = per_class_topk_accuracy(y_true, ranked_preds, classes, k)
    vals = [v for v in per.values() if not math.isnan(v)]
    return sum(vals) / len(vals)


def confusion_matrix(
    y_true: list[str], top1_preds: list[str], classes: list[str]
) -> list[list[int]]:
    """rows = true class, cols = predicted class, in `classes` order."""
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, top1_preds, strict=True):
        m[idx[t]][idx[p]] += 1
    return m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/metrics.py tests/test_metrics.py
git commit -m "feat: per-class/macro top-k accuracy and confusion matrix"
```

---

## Task 4: Raw-data normalizer + acquisition docs + verifier

Upstream zips all differ in structure. We normalize everything into one canonical
layout — `data/raw/<dataset>/<source_label>/*.jpg` (symlinks) — so ingest code
never touches upstream structure. Two normalization rules cover all four datasets:
generic label-directory matching, and SDNET2018's C/U prefix convention.

**Files:**
- Create: `scripts/normalize_raw.py`
- Create: `scripts/verify_raw.py`
- Create: `docs/datasets.md`
- Test: `tests/test_normalize_raw.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_normalize_raw.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_raw import (  # noqa: E402
    DATASET_LABELS,
    canon,
    match_label,
    normalize_generic,
    normalize_sdnet,
)


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")


def test_canon():
    assert canon("Major Crack") == "majorcrack"
    assert canon("peeling-paint") == "peelingpaint"
    assert canon("Exposed_Bars") == "exposedbars"


def test_match_label():
    labels = DATASET_LABELS["bd3"]
    assert match_label("Major Crack", labels) == "major_crack"
    assert match_label("ALGAE", labels) == "algae"
    assert match_label("random_junk", labels) is None


def test_normalize_generic(tmp_path):
    src = tmp_path / "bd3_clone"
    touch(src / "dataset" / "Major Crack" / "img1.jpg")
    touch(src / "dataset" / "Algae" / "img2.jpg")
    touch(src / "dataset" / "Algae" / "notes.txt")  # non-image ignored
    dest = tmp_path / "raw" / "bd3"
    n = normalize_generic(src, dest, DATASET_LABELS["bd3"])
    assert n == 2
    assert (dest / "major_crack").is_dir()
    linked = list((dest / "algae").iterdir())
    assert len(linked) == 1
    assert linked[0].is_symlink()


def test_normalize_sdnet(tmp_path):
    src = tmp_path / "SDNET2018"
    touch(src / "D" / "CD" / "c1.jpg")
    touch(src / "D" / "UD" / "u1.jpg")
    touch(src / "W" / "CW" / "c2.jpg")
    dest = tmp_path / "raw" / "sdnet2018"
    n = normalize_sdnet(src, dest)
    assert n == 3
    assert len(list((dest / "cracked").iterdir())) == 2
    assert len(list((dest / "non_cracked").iterdir())) == 1


def test_normalize_generic_no_collisions(tmp_path):
    src = tmp_path / "clone"
    touch(src / "train" / "Crack" / "img.jpg")
    touch(src / "test" / "Crack" / "img.jpg")  # same filename, different split dir
    dest = tmp_path / "raw" / "roboflow_walls"
    n = normalize_generic(src, dest, DATASET_LABELS["roboflow_walls"])
    assert n == 2
    assert len(list((dest / "crack").iterdir())) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalize_raw.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'normalize_raw'`

- [ ] **Step 3: Implement the normalizer**

`scripts/normalize_raw.py`:

```python
"""Normalize downloaded datasets into data/raw/<dataset>/<source_label>/ symlinks.

Usage:
  python scripts/normalize_raw.py --dataset bd3 --source /path/to/bd3_clone
  python scripts/normalize_raw.py --dataset sdnet2018 --source /path/to/SDNET2018
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Canonical source labels per dataset (must match configs/label_mapping.yaml).
DATASET_LABELS: dict[str, set[str]] = {
    "codebrim": {
        "background", "crack", "spallation", "efflorescence",
        "exposed_bars", "corrosion_stain",
    },
    "bd3": {
        "algae", "major_crack", "minor_crack", "peeling",
        "spalling", "stain", "normal",
    },
    "roboflow_walls": {
        "crack", "mold", "peeling_paint", "stairstep_crack", "water_seepage",
    },
    "sdnet2018": {"cracked", "non_cracked"},
}


def canon(name: str) -> str:
    """Lowercase and strip all non-alphanumeric chars for fuzzy dir matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_label(dirname: str, labels: set[str]) -> str | None:
    """Match a directory name against canonical labels, tolerant of case/spacing."""
    c = canon(dirname)
    for label in labels:
        if c == canon(label):
            return label
    return None


def _link_images(label_dir_src: Path, label_dir_dest: Path, rel_root: Path) -> int:
    label_dir_dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(label_dir_src.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            continue
        # Flatten relative path into the filename to avoid collisions.
        flat = "__".join(img.relative_to(rel_root).parts)
        dest = label_dir_dest / flat
        if not dest.exists():
            dest.symlink_to(img.resolve())
            n += 1
    return n


def normalize_generic(source: Path, dest: Path, labels: set[str]) -> int:
    """Find dirs anywhere under `source` whose name matches a known label;
    symlink their images into dest/<canonical_label>/."""
    n = 0
    for d in sorted(p for p in source.rglob("*") if p.is_dir()):
        label = match_label(d.name, labels)
        if label is None:
            continue
        n += _link_images(d, dest / label, rel_root=source)
    return n


def normalize_sdnet(source: Path, dest: Path) -> int:
    """SDNET2018: {D,P,W}/{C*,U*}/*.jpg — C=cracked, U=non_cracked."""
    n = 0
    for sub in sorted(p for p in source.rglob("*") if p.is_dir()):
        if sub.name.upper().startswith("C") and len(sub.name) == 2:
            label = "cracked"
        elif sub.name.upper().startswith("U") and len(sub.name) == 2:
            label = "non_cracked"
        else:
            continue
        n += _link_images(sub, dest / label, rel_root=source)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--dest-root", type=Path, default=Path("data/raw"),
        help="canonical raw root (default: data/raw)",
    )
    args = parser.parse_args()

    dest = args.dest_root / args.dataset
    if args.dataset == "sdnet2018":
        n = normalize_sdnet(args.source, dest)
    else:
        n = normalize_generic(args.source, dest, DATASET_LABELS[args.dataset])
    print(f"Linked {n} images into {dest}")
    if n == 0:
        raise SystemExit(
            "No images linked — check that --source points at the extracted dataset."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalize_raw.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write the verifier**

`scripts/verify_raw.py`:

```python
"""Verify canonical raw layout: print per-dataset/per-label counts.

Exits 1 if a REQUIRED dataset is missing or has an empty label dir.
Roboflow is optional (its classes are covered by BD3).
"""
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = ["codebrim", "bd3", "sdnet2018"]
OPTIONAL = ["roboflow_walls"]


def main() -> int:
    raw = Path("data/raw")
    failed = False
    for dataset in REQUIRED + OPTIONAL:
        ddir = raw / dataset
        required = dataset in REQUIRED
        if not ddir.is_dir():
            level = "MISSING (required)" if required else "absent (optional, OK)"
            print(f"{dataset}: {level}")
            failed |= required
            continue
        for label_dir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            count = sum(1 for f in label_dir.iterdir() if f.is_file())
            flag = "" if count > 0 else "  <-- EMPTY"
            print(f"{dataset}/{label_dir.name}: {count}{flag}")
            failed |= (count == 0 and required)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Write the acquisition doc**

`docs/datasets.md`:

```markdown
# Dataset Acquisition

All datasets are normalized into `data/raw/<dataset>/<source_label>/` (symlinks)
via `scripts/normalize_raw.py`. Downloads live outside the repo (e.g. `~/datasets/`).
`data/raw/` is gitignored.

## 1. CODEBRIM (required)

- Source: https://zenodo.org/records/2620293
- Download `CODEBRIM_classification_dataset.zip`, extract to `~/datasets/codebrim/`
- License: **non-commercial / educational use only** (fine for this portfolio project)
- Classes: background, crack, spallation, efflorescence, exposed_bars, corrosion_stain

    python scripts/normalize_raw.py --dataset codebrim --source ~/datasets/codebrim

Note: if the extracted structure does not contain per-class directories
(normalize prints "No images linked"), inspect with
`find ~/datasets/codebrim -maxdepth 3 -type d | head -30` and, if labels are
encoded some other way (e.g. metadata files), reorganize into per-class dirs
before re-running. Record whatever was needed in this file.

## 2. BD3 (required)

- Source: https://github.com/samy101/bd3-building-defects-detection-dataset
- `git clone https://github.com/samy101/bd3-building-defects-detection-dataset ~/datasets/bd3`
- Classes: algae, major_crack, minor_crack, peeling, spalling, stain, normal

    python scripts/normalize_raw.py --dataset bd3 --source ~/datasets/bd3

## 3. SDNET2018 (required)

- Source: https://digitalcommons.usu.edu/all_datasets/48/ (SDNET2018.zip, ~504 MB)
- License: CC-BY-4.0
- Structure: {D,P,W}/{C*,U*}/... where C=cracked, U=uncracked

    python scripts/normalize_raw.py --dataset sdnet2018 --source ~/datasets/SDNET2018

## 4. Roboflow building-defect-on-walls (OPTIONAL)

- Source: https://universe.roboflow.com/builddef2/building-defect-on-walls
- Requires a free Roboflow account; export the latest version in "Folder Structure"
  (classification) format, extract to `~/datasets/roboflow_walls/`
- If unavailable, skip — mold/water/peeling classes are covered by BD3.

    python scripts/normalize_raw.py --dataset roboflow_walls --source ~/datasets/roboflow_walls

## Verify

    python scripts/verify_raw.py

Prints per-label counts; exits non-zero if a required dataset is missing/empty.
```

- [ ] **Step 7: Download, normalize, verify (manual + scripted)**

Do the downloads per `docs/datasets.md` (CODEBRIM and SDNET2018 are browser
downloads; BD3 is a git clone; Roboflow optional). Then:

```bash
python scripts/normalize_raw.py --dataset codebrim --source ~/datasets/codebrim
python scripts/normalize_raw.py --dataset bd3 --source ~/datasets/bd3
python scripts/normalize_raw.py --dataset sdnet2018 --source ~/datasets/SDNET2018
python scripts/verify_raw.py
```

Expected: counts printed for every label dir of the three required datasets, exit 0.
Sanity expectations: sdnet2018 ≈ 8k cracked / 48k non_cracked; bd3 ≈ 3.9k total;
codebrim ≈ several hundred per defect class.

- [ ] **Step 8: Commit**

```bash
git add scripts/normalize_raw.py scripts/verify_raw.py docs/datasets.md tests/test_normalize_raw.py
git commit -m "feat: raw-dataset normalizer, verifier, and acquisition docs"
```

---

## Task 5: Ingest → unified manifest (with sampling caps)

**Files:**
- Create: `src/defectlens/ingest.py`
- Create: `configs/sampling.yaml`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`:

```python
from pathlib import Path

from defectlens.ingest import ManifestRow, apply_caps, scan_dataset, write_manifest, read_manifest
from defectlens.taxonomy import load_mapping

MAPPING_YAML = """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: t
  - source_dataset: bd3
    source_label: normal
    unified_label: no_defect
    rationale: t
"""


def make_raw(tmp_path: Path) -> Path:
    repo = tmp_path
    for label, names in {"algae": ["a1.jpg", "a2.jpg"], "normal": ["n1.jpg"]}.items():
        d = repo / "data" / "raw" / "bd3" / label
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(b"fake")
    (repo / "configs").mkdir()
    (repo / "configs" / "mapping.yaml").write_text(MAPPING_YAML)
    return repo


def test_scan_dataset(tmp_path):
    repo = make_raw(tmp_path)
    mapping = load_mapping(repo / "configs" / "mapping.yaml")
    rows = scan_dataset(repo, "bd3", mapping)
    assert len(rows) == 3
    assert rows[0].image_path.startswith("data/raw/bd3/")
    assert {r.unified_label for r in rows} == {"mold_algae", "no_defect"}


def test_scan_is_deterministic(tmp_path):
    repo = make_raw(tmp_path)
    mapping = load_mapping(repo / "configs" / "mapping.yaml")
    assert scan_dataset(repo, "bd3", mapping) == scan_dataset(repo, "bd3", mapping)


def test_apply_caps():
    rows = [
        ManifestRow(f"data/raw/x/l/{i}.jpg", "x", "l", "crack") for i in range(10)
    ]
    capped = apply_caps(rows, caps={"x": {"l": 4}}, seed=17)
    assert len(capped) == 4
    # deterministic
    assert capped == apply_caps(rows, caps={"x": {"l": 4}}, seed=17)
    # uncapped groups untouched
    assert apply_caps(rows, caps={}, seed=17) == sorted(rows, key=lambda r: r.image_path)


def test_manifest_roundtrip(tmp_path):
    rows = [ManifestRow("data/raw/x/l/0.jpg", "x", "l", "crack")]
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    assert read_manifest(out) == rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defectlens.ingest'`

- [ ] **Step 3: Implement ingest**

`configs/sampling.yaml`:

```yaml
# Deterministic per-(dataset, source_label) caps applied at ingest.
# Rationale: SDNET2018 would otherwise drown the taxonomy in crack/no_defect.
seed: 17
caps:
  sdnet2018:
    cracked: 4000
    non_cracked: 4000
```

`src/defectlens/ingest.py`:

```python
"""Scan canonical raw layout into the unified manifest CSV."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from defectlens.taxonomy import Mapping, load_mapping, map_label

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FIELDS = ["image_path", "source_dataset", "source_label", "unified_label"]


@dataclass(frozen=True)
class ManifestRow:
    image_path: str  # posix path relative to repo root
    source_dataset: str
    source_label: str
    unified_label: str


def scan_dataset(repo_root: Path, dataset_name: str, mapping: Mapping) -> list[ManifestRow]:
    dataset_dir = repo_root / "data" / "raw" / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} not found — run scripts/normalize_raw.py")
    rows: list[ManifestRow] = []
    for label_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        unified = map_label(mapping, dataset_name, label_dir.name)
        if unified is None:
            continue
        for img in sorted(label_dir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS and img.is_file():
                rows.append(
                    ManifestRow(
                        image_path=img.relative_to(repo_root).as_posix(),
                        source_dataset=dataset_name,
                        source_label=label_dir.name,
                        unified_label=unified,
                    )
                )
    return rows


def apply_caps(
    rows: list[ManifestRow], caps: dict[str, dict[str, int]], seed: int
) -> list[ManifestRow]:
    """Deterministically subsample groups exceeding their cap."""
    grouped: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.source_dataset, r.source_label)].append(r)
    rng = random.Random(seed)
    out: list[ManifestRow] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: r.image_path)
        cap = caps.get(key[0], {}).get(key[1])
        if cap is not None and len(group) > cap:
            group = rng.sample(group, cap)
        out.extend(group)
    return sorted(out, key=lambda r: r.image_path)


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as f:
        return [ManifestRow(**row) for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified manifest CSV.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/manifest.csv"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    mapping = load_mapping(repo / "configs" / "label_mapping.yaml")
    sampling = yaml.safe_load((repo / "configs" / "sampling.yaml").read_text())

    rows: list[ManifestRow] = []
    for dataset_dir in sorted((repo / "data" / "raw").iterdir()):
        if dataset_dir.is_dir():
            rows.extend(scan_dataset(repo, dataset_dir.name, mapping))
    rows = apply_caps(rows, sampling.get("caps", {}), sampling["seed"])
    write_manifest(rows, repo / args.out)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.unified_label] += 1
    print(f"Wrote {len(rows)} rows to {args.out}")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 4 passed.

- [ ] **Step 5: Build the real manifest**

Run:
```bash
python -m defectlens.ingest
```
Expected: `Wrote N rows to data/manifests/manifest.csv` with per-class counts;
crack ≈ 4k–6k (capped SDNET + others), no_defect ≈ 4k–6k, each defect class > 300.

- [ ] **Step 6: Commit**

```bash
git add src/defectlens/ingest.py configs/sampling.yaml tests/test_ingest.py data/manifests/manifest.csv
git commit -m "feat: unified manifest ingest with deterministic sampling caps"
```

---

## Task 6: Frozen stratified train/test split

**Files:**
- Create: `src/defectlens/split.py`
- Test: `tests/test_split.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_split.py`:

```python
from defectlens.ingest import ManifestRow
from defectlens.split import stratified_split


def make_rows(dataset: str, label: str, n: int) -> list[ManifestRow]:
    return [
        ManifestRow(f"data/raw/{dataset}/{label}/{i}.jpg", dataset, label, label)
        for i in range(n)
    ]


def test_split_is_disjoint_and_complete():
    rows = make_rows("d1", "crack", 100) + make_rows("d2", "spalling", 50)
    train, test = stratified_split(rows, test_fraction=0.2, seed=42)
    assert len(train) + len(test) == 150
    assert set(r.image_path for r in train).isdisjoint(r.image_path for r in test)


def test_split_is_stratified():
    rows = make_rows("d1", "crack", 100) + make_rows("d2", "spalling", 50)
    train, test = stratified_split(rows, test_fraction=0.2, seed=42)
    test_crack = sum(1 for r in test if r.unified_label == "crack")
    test_spall = sum(1 for r in test if r.unified_label == "spalling")
    assert test_crack == 20
    assert test_spall == 10


def test_split_is_deterministic():
    rows = make_rows("d1", "crack", 100)
    a = stratified_split(rows, test_fraction=0.2, seed=42)
    b = stratified_split(rows, test_fraction=0.2, seed=42)
    assert a == b


def test_small_groups_get_test_representation():
    rows = make_rows("d1", "efflorescence", 5)
    train, test = stratified_split(rows, test_fraction=0.15, seed=42)
    assert len(test) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defectlens.split'`

- [ ] **Step 3: Implement the splitter**

`src/defectlens/split.py`:

```python
"""Seeded stratified split; outputs are FROZEN once committed (spec §4)."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from defectlens.ingest import ManifestRow, read_manifest, write_manifest


def stratified_split(
    rows: list[ManifestRow], test_fraction: float, seed: int
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Stratify by (source_dataset, unified_label)."""
    grouped: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.source_dataset, r.unified_label)].append(r)
    rng = random.Random(seed)
    train: list[ManifestRow] = []
    test: list[ManifestRow] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: r.image_path)
        rng.shuffle(group)
        n_test = round(len(group) * test_fraction)
        if len(group) >= 4:
            n_test = max(1, n_test)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    key = lambda r: r.image_path  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the train/test split.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/manifest.csv"))
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    train, test = stratified_split(rows, args.test_fraction, args.seed)
    out_dir = args.manifest.parent
    write_manifest(train, out_dir / "train.csv")
    write_manifest(test, out_dir / "test.csv")

    def table(name: str, split_rows: list[ManifestRow]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for r in split_rows:
            counts[r.unified_label] += 1
        print(f"{name}: {len(split_rows)} rows")
        for label in sorted(counts):
            print(f"  {label}: {counts[label]}")

    table("train", train)
    table("test", test)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_split.py -v`
Expected: 4 passed.

- [ ] **Step 5: Freeze the real split and commit it**

Run:
```bash
python -m defectlens.split
```
Expected: train/test tables printed; every unified class present in test.

```bash
git add src/defectlens/split.py tests/test_split.py data/manifests/train.csv data/manifests/test.csv
git commit -m "feat: freeze stratified train/test split (seed=42, test=0.15)"
```

**⚠️ From this commit on, `data/manifests/test.csv` is frozen. All reported
numbers (baseline and fine-tuned) use this file. Never regenerate it without
explicit user sign-off — that invalidates every prior number.**

---

## Task 7: Spot-check protocol (spec §4 QA gate)

**Files:**
- Create: `scripts/spot_check.py`
- Test: `tests/test_spot_check.py`

- [ ] **Step 1: Write the failing test**

`tests/test_spot_check.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spot_check import sample_per_class  # noqa: E402

from defectlens.ingest import ManifestRow


def test_sample_per_class_deterministic_and_bounded():
    rows = [
        ManifestRow(f"data/raw/d/crack/{i}.jpg", "d", "crack", "crack") for i in range(100)
    ] + [
        ManifestRow(f"data/raw/d/algae/{i}.jpg", "d", "algae", "mold_algae") for i in range(3)
    ]
    picked = sample_per_class(rows, n_per_class=30, seed=7)
    by_class = {}
    for r in picked:
        by_class.setdefault(r.unified_label, []).append(r)
    assert len(by_class["crack"]) == 30
    assert len(by_class["mold_algae"]) == 3  # fewer than n -> take all
    assert picked == sample_per_class(rows, n_per_class=30, seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spot_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spot_check'`

- [ ] **Step 3: Implement**

`scripts/spot_check.py`:

```python
"""Sample N images per unified class into review_grid/ for manual QA.

Protocol (spec §4): review ~30 images per class; any class whose mapping looks
wrong (>10% of samples don't match their unified label) gets its mapping entry
revisited in configs/label_mapping.yaml BEFORE the split is trusted.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectlens.ingest import ManifestRow, read_manifest  # noqa: E402


def sample_per_class(
    rows: list[ManifestRow], n_per_class: int, seed: int
) -> list[ManifestRow]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[r.unified_label].append(r)
    rng = random.Random(seed)
    picked: list[ManifestRow] = []
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda r: r.image_path)
        k = min(n_per_class, len(group))
        picked.extend(rng.sample(group, k))
    return sorted(picked, key=lambda r: r.image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/manifest.csv"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("review_grid"))
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    picked = sample_per_class(rows, args.n, args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    for r in picked:
        dest_dir = args.out / f"{r.unified_label}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(r.image_path)
        dest = dest_dir / f"{r.source_dataset}__{src.name}"
        dest.symlink_to(src.resolve())
    print(f"Wrote review grid to {args.out}/ — open in Finder and review per class.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spot_check.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the spot-check and review (manual gate)**

Run:
```bash
python scripts/spot_check.py && open review_grid
```
Review ~30 images per class in Finder. **Gate:** if any class has >10%
mislabeled-looking samples, fix `configs/label_mapping.yaml` (with rationale),
rebuild manifest + split, and re-review before proceeding. Record the outcome
in the commit message.

- [ ] **Step 6: Commit**

```bash
git add scripts/spot_check.py tests/test_spot_check.py
git commit -m "feat: per-class spot-check protocol tooling (manual QA passed)"
```

---

## Task 8: CLIP zero-shot evaluator (prompt ensembling)

**Files:**
- Create: `configs/clip_prompts.yaml`
- Create: `src/defectlens/eval/__init__.py`
- Create: `src/defectlens/eval/clip_zeroshot.py`
- Test: `tests/test_clip_zeroshot.py`

- [ ] **Step 1: Write the failing tests** (pure functions only — the model path is exercised in Task 9's real run)

`tests/test_clip_zeroshot.py`:

```python
import numpy as np

from defectlens.eval.clip_zeroshot import expand_prompts, rank_from_similarity


def test_expand_prompts():
    phrases = {"crack": "a crack", "no_defect": "a clean wall"}
    templates = ["a photo of {}", "{}"]
    prompts = expand_prompts(phrases, templates)
    assert prompts["crack"] == ["a photo of a crack", "a crack"]
    assert prompts["no_defect"] == ["a photo of a clean wall", "a clean wall"]


def test_rank_from_similarity():
    classes = ["a", "b", "c"]
    # image 0 most similar to c, then a, then b
    sims = np.array([[0.2, 0.1, 0.9], [0.8, 0.7, 0.1]])
    ranked = rank_from_similarity(sims, classes)
    assert ranked[0] == ["c", "a", "b"]
    assert ranked[1] == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clip_zeroshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defectlens.eval'`

- [ ] **Step 3: Implement config + evaluator**

`configs/clip_prompts.yaml`:

```yaml
# Prompt-ensemble config for the CLIP zero-shot baseline (spec §5).
model: openai/clip-vit-large-patch14
templates:
  - "a photo of {}"
  - "a close-up photo of {}"
  - "an inspection photo of {}"
  - "{}"
class_phrases:
  crack: "a crack in a concrete or masonry surface"
  spalling: "spalling concrete with the surface flaking or broken away"
  efflorescence: "white efflorescence salt deposits on a concrete or brick wall"
  exposed_rebar: "exposed steel reinforcement bars in damaged concrete"
  corrosion_stain: "rust and corrosion stains on a concrete surface"
  mold_algae: "mold or algae growth on a building wall"
  water_damage: "water damage stains or moisture seepage on a wall"
  peeling_paint: "peeling or flaking paint on a wall"
  no_defect: "a clean undamaged building surface"
```

`src/defectlens/eval/__init__.py`:

```python
```

(empty file)

`src/defectlens/eval/clip_zeroshot.py`:

```python
"""CLIP zero-shot baseline on the frozen test split (spec §5).

Produces results/clip_zeroshot_baseline.json + confusion_matrix.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from defectlens.ingest import read_manifest
from defectlens.metrics import confusion_matrix, macro_topk_accuracy, per_class_topk_accuracy
from defectlens.taxonomy import UNIFIED_CLASSES


def expand_prompts(
    class_phrases: dict[str, str], templates: list[str]
) -> dict[str, list[str]]:
    return {
        cls: [t.format(phrase) for t in templates]
        for cls, phrase in class_phrases.items()
    }


def rank_from_similarity(sims: np.ndarray, classes: list[str]) -> list[list[str]]:
    """sims: [n_images, n_classes] -> per-image class ranking, best first."""
    order = np.argsort(-sims, axis=1)
    return [[classes[j] for j in row] for row in order]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_text_features(model, processor, prompts: dict[str, list[str]], device: str):
    feats = []
    for cls in UNIFIED_CLASSES:
        inputs = processor(text=prompts[cls], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.mean(dim=0))
    feats = torch.stack(feats)
    return feats / feats.norm(dim=-1, keepdim=True)


def main() -> None:
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/clip_prompts.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts = expand_prompts(cfg["class_phrases"], cfg["templates"])
    assert set(prompts) == set(UNIFIED_CLASSES), "prompt config must cover all classes"

    device = pick_device()
    print(f"Device: {device}; model: {cfg['model']}")
    model = CLIPModel.from_pretrained(cfg["model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg["model"])

    text_feats = build_text_features(model, processor, prompts, device)

    rows = read_manifest(args.manifest)
    y_true = [r.unified_label for r in rows]
    all_sims: list[np.ndarray] = []
    for i in tqdm(range(0, len(rows), args.batch_size), desc="images"):
        batch = rows[i : i + args.batch_size]
        images = [Image.open(r.image_path).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_sims.append((emb @ text_feats.T).cpu().numpy())
    sims = np.concatenate(all_sims)
    ranked = rank_from_similarity(sims, UNIFIED_CLASSES)
    top1 = [r[0] for r in ranked]

    results = {
        "model": cfg["model"],
        "manifest": str(args.manifest),
        "n_images": len(rows),
        "macro_top1": macro_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=1),
        "macro_top3": macro_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=3),
        "per_class_top1": per_class_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=1),
        "per_class_top3": per_class_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=3),
        "confusion_matrix": confusion_matrix(y_true, top1, UNIFIED_CLASSES),
        "classes": UNIFIED_CLASSES,
    }
    args.out_dir.mkdir(exist_ok=True)
    out_json = args.out_dir / "clip_zeroshot_baseline.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"macro top-1: {results['macro_top1']:.3f}  macro top-3: {results['macro_top3']:.3f}")
    print(f"Wrote {out_json}")

    # Confusion matrix figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(results["confusion_matrix"], dtype=float)
    m_norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(UNIFIED_CLASSES)), UNIFIED_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(UNIFIED_CLASSES)), UNIFIED_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CLIP zero-shot — row-normalized confusion")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out_dir / "clip_zeroshot_confusion.png", dpi=150)
    print(f"Wrote {args.out_dir / 'clip_zeroshot_confusion.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_clip_zeroshot.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (≈23).

- [ ] **Step 6: Commit**

```bash
git add configs/clip_prompts.yaml src/defectlens/eval tests/test_clip_zeroshot.py
git commit -m "feat: CLIP zero-shot evaluator with prompt ensembling"
```

---

## Task 9: Run the baseline, record the numbers

**Files:**
- Create: `results/clip_zeroshot_baseline.json` (generated)
- Create: `results/clip_zeroshot_confusion.png` (generated)
- Modify: `README.md`

- [ ] **Step 1: Run the baseline on the frozen test split**

Run:
```bash
python -m defectlens.eval.clip_zeroshot
```
First run downloads CLIP ViT-L/14 (~1.7 GB). Expected: a few minutes of batched
inference on MPS; prints macro top-1 / top-3 (spec expectation: roughly 0.40–0.60
macro top-1 — if it's >0.85, something is leaking; if <0.15, check prompts/images).

- [ ] **Step 2: Update README with the baseline table**

Append to `README.md`:

```markdown
## Results

| Model | Macro top-1 | Macro top-3 | Split |
|---|---|---|---|
| CLIP ViT-L/14 zero-shot (prompt ensemble) | <from results JSON> | <from results JSON> | frozen `data/manifests/test.csv` |
| Qwen2.5-VL-3B + LoRA (Phase 3) | — | — | same |

![CLIP zero-shot confusion matrix](results/clip_zeroshot_confusion.png)
```

Fill in the two numbers from `results/clip_zeroshot_baseline.json` (3 decimal places).

- [ ] **Step 3: Commit**

```bash
git add results/clip_zeroshot_baseline.json results/clip_zeroshot_confusion.png README.md
git commit -m "feat: CLIP zero-shot baseline on frozen test split"
```

- [ ] **Step 4: Phase 1 exit review**

Verify against spec §8 week-1 exit criteria:
- [ ] `data/manifests/test.csv` frozen and committed
- [ ] baseline macro top-1/top-3 recorded in README + results JSON
- [ ] all tests green (`pytest -v`)
- [ ] spot-check protocol run and outcome recorded (Task 7)

Then report results to the user before starting Phase 2 (RAG) planning.

---

## Self-review (done at plan-write time)

- **Spec coverage (§4, §5, §8-wk1):** mapping table → Task 2; frozen split → Task 6; macro metrics → Task 3; spot-check → Task 7; license review → Task 4 (`docs/datasets.md`); CLIP ViT-L/14 zero-shot + prompt ensembling + per-class confusion on the frozen split → Tasks 8–9. ✓
- **Placeholders:** the two `<from results JSON>` cells in Task 9 are measurement outputs by definition, filled at run time from a named file. No other TBDs. ✓
- **Type consistency:** `ManifestRow` fields, `read_manifest`/`write_manifest`, `Mapping` alias, and `UNIFIED_CLASSES` usage verified consistent across Tasks 2, 5, 6, 7, 8. ✓
- **Known execution-time risk (documented, not hidden):** CODEBRIM's extracted zip structure was not inspectable at plan time; Task 4 Step 7 + `docs/datasets.md` carry the inspection/adaptation instructions, and `verify_raw.py` + Task 5's count expectations catch a bad normalization immediately.
