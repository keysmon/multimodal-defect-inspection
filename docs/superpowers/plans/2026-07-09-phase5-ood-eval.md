# Phase 5.4: Cross-Dataset OOD Evaluation - Mini-Plan (controller-driven)

**Goal:** Measure the fine-tuned classifier's generalization gap on an
independently collected dataset, per spec decision 3 (amended: no
self-collected photos; cross-dataset framing).

**Source (verified 2026-07-09):** Ozgenel & Gonenc Sorguc, "Concrete Crack
Images for Classification", Mendeley Data 5y9wdsg2zt v2, CC BY 4.0.
20k crack + 20k negative 227x227 crops from 458 high-res photos of METU
campus buildings (Ankara) - independent of SDNET2018 (Utah State),
CODEBRIM (bridge decks), BD3. Covers exactly the crack/no_defect axis -
the dominant error cluster (65% of in-distribution errors).

**Scope honesty (stated in README):** this measures OOD generalization on
TWO of nine classes; independent labeled sources for the other seven do
not exist at usable scale. In-distribution reference points (full test
split): crack 0.818, no_defect 0.884 top-1.

**Steps:**
1. `scripts/fetch_ood_crack.sh` - download to ~/datasets/metu_crack,
   symlink data/raw/ood_crack/{Positive,Negative}, no MD5 published ->
   record sha256 on first download in the script comment.
2. `scripts/build_ood_manifest.py` - stratified sample 200/class
   (seed 42, sorted-path determinism) -> data/manifests/ood_test.csv
   (image_path, source_dataset=metu_crack, source_label, unified_label).
   Committed (paths are symlink-relative like other manifests).
3. Run `python -m defectlens.eval.vlm_topk --test-manifest
   data/manifests/ood_test.csv --adapter models/qwen25vl-lora-v1
   --out-name vlm_topk_ood.json` locally (bf16 MPS, ~1h for 400 images).
4. Report: OOD vs in-distribution table; the GAP is the headline.
   Recovery round (spec: recover >=50% of gap) only if the gap >= 5
   points - it needs a GPU fine-tune run (~$2.5, ASK USER before launch).
   If gap < 5 points: report "robust under this shift", no spend.

**De-scope trigger honored:** measure-and-report; no grinding.
