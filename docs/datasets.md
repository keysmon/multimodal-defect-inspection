# Dataset Acquisition

All datasets are normalized into `data/raw/<dataset>/<source_label>/` (symlinks)
via `scripts/normalize_raw.py`. Downloads live outside the repo (in `~/datasets/`).
`data/raw/` is gitignored.

## 1. CODEBRIM (required)

- Source: https://zenodo.org/records/2620293
- Download `CODEBRIM_classification_dataset.zip` (**~7.9 GB**), extract to `~/datasets/codebrim/`
- Direct URL: https://zenodo.org/api/records/2620293/files/CODEBRIM_classification_dataset.zip/content
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
- `git clone --depth 1 https://github.com/samy101/bd3-building-defects-detection-dataset ~/datasets/bd3`
- Classes: algae, major_crack, minor_crack, peeling, spalling, stain, normal

    python scripts/normalize_raw.py --dataset bd3 --source ~/datasets/bd3

## 3. SDNET2018 (required)

- Source: https://digitalcommons.usu.edu/all_datasets/48/ (SDNET2018.zip, ~504 MB;
  ships as `DATA_Maguire_20180517_ALL.zip`)
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
