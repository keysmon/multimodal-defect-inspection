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
- **Extraction gotcha (verified 2026-07-06):** the Zenodo zip is a malformed
  zip64 (offsets past 4 GiB truncated) — macOS `unzip`, `tar`, and Python
  `zipfile` all fail or partially extract, and `zip -FF` repair also fails.
  **`ditto -xk <zip> <dest>` extracts it fully** (7,729 pngs).
- **Real structure (NOT per-class folders):**
  `classification_dataset/{train,val,test}/{background,defects}/*.png` plus
  `metadata/{defects.xml,background.xml}` carrying per-crop multi-label binary
  flags (Background/Crack/Spallation/Efflorescence/ExposedBars/CorrosionStain).
- **Prep step (before normalize_raw):** CODEBRIM is multi-label; our v1
  taxonomy is single-label (spec §11), so `scripts/prepare_codebrim.py` keeps
  only crops with exactly one flag set (multi-label crops are counted and
  skipped) and stages them into per-class dirs:

      python scripts/prepare_codebrim.py --source ~/datasets/codebrim/classification_dataset --out ~/datasets/codebrim_by_class
      python scripts/normalize_raw.py --dataset codebrim --source ~/datasets/codebrim_by_class

- Known count reconciliation: metadata has 7,735 entries vs 7,729 files on
  disk — 6 entries lack files; the script reports this (`metadata_without_file`)
  rather than crashing.

## 2. BD3 (required)

- Source: https://www.kaggle.com/datasets/praveenkottari/bd3-dataset-for-building-defect-detection
  (**discovered 2026-07-06:** both GitHub repos — samy101 and Praveenkottari — contain
  only README/code, NOT the images; the actual dataset is hosted on Kaggle)
- Requires a Kaggle account + API token (`~/.kaggle/kaggle.json`, from
  kaggle.com → Settings → API → Create New Token), then:

      kaggle datasets download praveenkottari/bd3-dataset-for-building-defect-detection -p ~/datasets/ --unzip

  (or download the zip in a logged-in browser and extract to `~/datasets/bd3/`)
- Classes: algae, major_crack, minor_crack, peeling, plain (no-defect; GitHub docs call it 'normal'), spalling, stain

    python scripts/normalize_raw.py --dataset bd3 --source ~/datasets/bd3

## 3. SDNET2018 (required)

- Source: https://digitalcommons.usu.edu/all_datasets/48/ (~504 MB)
- Direct URL: https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent
- **Nested-zip gotcha (verified 2026-07-06):** the download is
  `DATA_Maguire_20180517_ALL.zip`, which contains a ReadMe and an inner
  `SDNET2018.zip` — extract BOTH (inner one to `~/datasets/SDNET2018/`).
- License: CC-BY-4.0
- Structure: {D,P,W}/{C*,U*}/... where C=cracked, U=uncracked (56,092 images verified)

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

## 5. BFDD — RGB-IR thermal comparison (Phase 5.6, OFFLINE)

Separate from the taxonomy datasets above: BFDD is used only for the offline
thermal-modality segmentation comparison (`src/defectlens/thermal/`), not the
serving taxonomy or `normalize_raw.py`/`verify_raw.py` flow.

- Source: https://data.mendeley.com/datasets/9ych7czvyg/1
  (paper: "BFDD: A Pixel-Level Aligned RGB-IR Image Dataset for Building Façade
  Defect Segmentation", version 1, published 2026-04-10)
- License: **CC BY 4.0**
- Fetch: `scripts/fetch_bfdd.sh` (downloads ~528 MB tarball to `~/datasets/bfdd`,
  sha256 `43d06305bf3c913f59d52c3ffa10caa0e129b668b7b3c9d8f80d619c6e6e8a7a`,
  extracts `Dataset_1x/`)
- Layout: `Dataset_1x/{RGB,IR,Label,Label_color}/<stem>.{JPG,png,png,png}` —
  838 pixel-aligned triples, 640×512. `RGB` is RGB-mode JPG; `IR` is an
  RGB-mode (false-color) PNG; `Label` is L-mode ids 0-5; `Label_color` is a
  colorized mask used here for legend verification. Ignore
  `Label_backup_7classes_20260125/` (superseded 7-class annotation round).
- Pixel distribution over all 838 labels: id0 91.96%, id1 1.70%, id2 0.59%,
  id3 1.11%, id4 2.15%, id5 2.49% — heavy imbalance; report per-class IoU.

### Class-id → name mapping (verified 2026-07-09)

The paper names 5 defect classes (Cracks, Peeling, Hollow Areas, Stains,
Erosion) but does **not** publish the id→name order, so it was resolved from
evidence, not the paper's listing order.

| id | name         | Label_color RGB   | evidence |
|----|--------------|-------------------|----------|
| 0  | background   | (0, 0, 0)         | unlabeled facade |
| 1  | crack        | (61, 61, 245)     | mask traces thin dark hairlines in RGB; present in 833/838 images (near-ubiquitous), most single-defect examples |
| 2  | hollow_area  | (169, 36, 191)    | near-invisible in RGB (featureless wall) but clear thermal-contrast patches in IR at the masked regions — matches the description's "sub-surface delamination (hollows)… invisible in the RGB spectrum"; rarest class (188 images, 0.59% px) |
| 3  | peeling      | (174, 79, 13)     | masks rounded blisters / bumps where the render/paint coating bubbles up and detaches (material bulging outward); consistent across 2 inspected images |
| 4  | erosion      | (36, 179, 83)     | masks regions where the surface finish is worn/spalled away exposing the darker rough substrate (material loss); consistent across 3 inspected images |
| 5  | stain        | (203, 253, 0)     | masks broad dark vertical discoloration streaks (water/dirt run-off) with the surface otherwise intact |

**Evidence tiers / confidence.** The id→color RGB tuples above are reproducible
(the dominant `Label_color` color at pixels where `Label==id`, consistent across
every image containing that id). crack, hollow_area, and stain are corroborated
by the Mendeley description and IR behavior and are high-confidence. peeling vs
erosion rests on visual inference — the discriminator is *material bulging
outward* (blisters → peeling, id3) vs *material removed* (worn/spalled loss →
erosion, id4), which are opposite processes; consistent across multiple examples
but not confirmed against a published color legend (the linked article is
paywalled). No id was left `unverified_<id>`.

**Recorded discrepancies (do not affect the mapping):**
- The Mendeley description says **788** aligned pairs, but the shipped
  `Dataset_1x/` contains **838** complete RGB/IR/Label triples (all used).
- The dataset ships its own `train.txt`/`test.txt` split. The thermal comparison
  intentionally ignores these and uses a frozen seed-42 70/15/15 split
  (`split_stems`) so the three input variants (rgb/ir/rgbir) share one split;
  matching the paper's official partition is not a goal of this internal
  controlled comparison.
