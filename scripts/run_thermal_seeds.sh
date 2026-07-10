#!/usr/bin/env bash
# Seed-replicated BFDD comparison: 3 seeds x 3 variants = 9 runs (~4.5-7.5h on
# M-series MPS; background it). Adds error bars to the Phase 5.6 comparison and
# tests the hybrid-stem fix for the fusion-init confound.
#
# ir is intentionally EXCLUDED: its single-seed mean-defect IoU (0.156 vs rgb
# 0.472) is a gap far too large to be seed noise, so replicating it would burn
# ~1.5h to confirm the obvious. rgb / rgbir / rgbir_hybrid are the variants where
# init + shuffle variance could plausibly matter to the conclusion.
#
# NOTE on the split: the train/val/test partition is frozen and manifest-backed
# (frozen_split_pairs) and is seed-INDEPENDENT by design. --seed varies only
# weight initialization and batch-shuffle order - that is exactly the source of
# variance this replication measures. (On MPS the backend is also non-
# deterministic, so the reported std is an upper bound on seed sensitivity, not a
# pure init-variance estimate.)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results models/thermal_seeds

for seed in 42 43 44; do
  for v in rgb rgbir rgbir_hybrid; do
    echo "=== variant $v seed $seed ==="
    .venv/bin/python -m defectlens.thermal.train_seg \
      --variant "$v" --seed "$seed" --epochs 25 --batch-size 4 \
      --output-dir "models/thermal_seeds/$v-s$seed"
  done
done

.venv/bin/python - <<'EOF'
import json, pathlib
import numpy as np

seeds = [42, 43, 44]
variants = ["rgb", "rgbir", "rgbir_hybrid"]
root = pathlib.Path("models/thermal_seeds")

out = {"run_config": {"epochs": 25, "batch_size": 4, "lr": 6e-5, "seeds": seeds}}
for v in variants:
    ms = {s: json.loads((root / f"{v}-s{s}" / "metrics.json").read_text()) for s in seeds}
    classes = list(ms[seeds[0]]["per_class_iou"].keys())
    per_class = {}
    for c in classes:
        # None (class absent) -> nan; nanmean/nanstd ignore it.
        vals = np.array([ms[s]["per_class_iou"][c] for s in seeds], dtype=float)
        per_class[c] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    md = np.array([ms[s]["mean_defect_iou"] for s in seeds], dtype=float)
    out[v] = {
        "per_class_iou": per_class,
        "mean_defect_iou": {"mean": float(np.nanmean(md)), "std": float(np.nanstd(md))},
        "final_train_loss_per_seed": {str(s): ms[s]["final_train_loss"] for s in seeds},
    }

pathlib.Path("results/thermal_bfdd_seeds.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF
