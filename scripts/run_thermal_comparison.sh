#!/usr/bin/env bash
# Full rgb / ir / rgbir comparison. ~30-45 min per run on M3 Pro MPS.
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure output dirs exist so a fresh clone doesn't fail at the consolidation
# write after ~2h of training (git doesn't track empty dirs).
mkdir -p results models/thermal_bfdd

for v in rgb ir rgbir; do
  echo "=== variant: $v ==="
  .venv/bin/python -m defectlens.thermal.train_seg \
    --variant "$v" --epochs 25 --batch-size 4 \
    --output-dir "models/thermal_bfdd/$v"
done
.venv/bin/python - <<'EOF'
import json, pathlib
# run_config pins the shared schedule (matches the per-variant train invocations
# above + train_seg defaults for lr/seed) so results are reproducible on their own.
out = {"run_config": {"epochs": 25, "batch_size": 4, "lr": 6e-5, "seed": 42}}
out.update({v: json.loads((pathlib.Path("models/thermal_bfdd")/v/"metrics.json").read_text())
            for v in ("rgb", "ir", "rgbir")})
pathlib.Path("results/thermal_bfdd.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF
