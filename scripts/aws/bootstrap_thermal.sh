#!/bin/bash
# DefectLens Phase 5.6 thermal seed-experiment bootstrap — runs as EC2 user-data
# on a Deep Learning AMI (PyTorch, Ubuntu) GPU box for the 9-run BFDD seed
# replication (rgb / rgbir / rgbir_hybrid x seeds 42/43/44) on CUDA.
#
# NOT run directly: scripts/aws/launch_gpu.sh with BOOTSTRAP_FILE pointing here
# prepends an `export` header (S3_BUCKET, AWS_REGION, IDLE_TIMEOUT_SEC, ...).
# NOTE: launch_gpu.sh always exports AWS_REGION (default us-east-1), so the
# launch MUST pass AWS_REGION=ca-central-1 for the -ca- bucket / SG to resolve.
#
# Expected env (from the launch header):
#   S3_BUCKET                 required, e.g. defectlens-phase3-ca-002559670021
#   AWS_REGION                default ca-central-1 (shadowed by the launch header)
#   IDLE_TIMEOUT_SEC          default 1800 - shutdown if no output progress this long
#   CHECKPOINT_SYNC_INTERVAL  reused as the S3 sync interval (default 120)
set -euxo pipefail

# cloud-init user-data runs with NO $HOME in the environment (learned live
# 2026-07-10: fetch_bfdd.sh died on "HOME: unbound variable" under set -u,
# ~$0.15 lesson). Everything downstream (fetch destination, Path.home() in
# bfdd.py, pip cache) expects root's home — set it explicitly.
export HOME=/root

: "${S3_BUCKET:?S3_BUCKET must be set by launch_gpu.sh}"
: "${AWS_REGION:=ca-central-1}"
: "${IDLE_TIMEOUT_SEC:=1800}"
: "${SYNC_INTERVAL:=${CHECKPOINT_SYNC_INTERVAL:-120}}"

WORKDIR=/opt/defectlens-thermal
LOG_FILE="${WORKDIR}/thermal_seeds.log"
S3_PREFIX="s3://${S3_BUCKET}/phase5/thermal_seeds"

mkdir -p "${WORKDIR}/models/thermal_seeds" "${WORKDIR}/results"

# Safety net: on ANY exit (success, crash, unhandled error under set -e), push
# the cloud-init log to S3 then power off, so a bug never leaves a paid GPU box
# running and there's always diagnosis output reachable. Set FIRST.
shutdown_now() {
  aws s3 cp /var/log/cloud-init-output.log "${S3_PREFIX}/cloud-init-output.log" \
    --region "$AWS_REGION" || true
  sudo shutdown -h now || true
}
trap shutdown_now EXIT

echo "== locating a python with torch (DLAMI env layout varies by AMI generation) =="
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch || true
fi
ls /opt || true   # in the -x trace: shows actual env layout for diagnosis
PY=""
for CAND in "$(command -v python3 || true)" \
    /opt/conda/envs/pytorch/bin/python \
    /opt/pytorch/bin/python /opt/pytorch/bin/python3; do
  [[ -n "$CAND" && -x "$CAND" ]] || continue
  if "$CAND" -c 'import torch' 2>/dev/null; then PY="$CAND"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "FATAL: no python with torch found on this AMI — aborting" >&2
  exit 1
fi
echo "using python: $PY"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available())'
nvidia-smi || echo "WARNING: nvidia-smi not found — GPU may not be attached/ready"

echo "== installing run deps (NOT torch — keep the AMI's CUDA torch 2.12) =="
# transformers/pillow/numpy don't pull torch; the AMI's CUDA build stays.
"$PY" -m pip install "transformers==5.13.*" pillow numpy

echo "== pulling code tarball from S3 =="
cd "$WORKDIR"
aws s3 cp "${S3_PREFIX}/code.tar.gz" "${WORKDIR}/code.tar.gz" --region "$AWS_REGION"
# Tarball root contains src/, scripts/, data/manifests/ (incl. bfdd_split.csv).
tar -xzf "${WORKDIR}/code.tar.gz" -C "${WORKDIR}"
export PYTHONPATH="${WORKDIR}/src"

echo "== fetching BFDD dataset (~528 MB, sha-verified) =="
# Installs to $HOME/datasets/bfdd; user-data runs as root so HOME=/root, which
# matches bfdd.BFDD_ROOT = Path.home()/datasets/bfdd/Dataset_1x. shasum-or-
# sha256sum fallback in the script handles Ubuntu (no shasum).
bash scripts/fetch_bfdd.sh

echo "== background S3 sync of run outputs (every ${SYNC_INTERVAL}s) =="
(
  while true; do
    sleep "$SYNC_INTERVAL"
    aws s3 sync "${WORKDIR}/models/thermal_seeds" "${S3_PREFIX}/models/" --region "$AWS_REGION" || true
  done
) &
SYNC_PID=$!

echo "== idle-safety watchdog (shutdown after ${IDLE_TIMEOUT_SEC}s with no progress) =="
# Watch BOTH the tee'd log (updated every 50 steps) and the model outputs
# (written at end-of-run, ~every 10 min) so a long run isn't mistaken for idle.
touch "$LOG_FILE"
(
  last_mtime=0
  idle_since=$(date +%s)
  while true; do
    sleep 60
    newest=$(find "$LOG_FILE" "${WORKDIR}/models/thermal_seeds" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    newest="${newest:-0}"
    now=$(date +%s)
    if [[ "$newest" != "$last_mtime" ]]; then
      last_mtime="$newest"
      idle_since="$now"
    elif (( now - idle_since > IDLE_TIMEOUT_SEC )); then
      echo "idle watchdog: no output progress for ${IDLE_TIMEOUT_SEC}s, shutting down"
      sudo shutdown -h now
    fi
  done
) &
WATCHDOG_PID=$!

echo "== running the 9-run seed experiment on CUDA =="
set +e
DEVICE=cuda PY="$PY" bash scripts/run_thermal_seeds.sh 2>&1 | tee "$LOG_FILE"
RUN_EXIT=${PIPESTATUS[0]}
set -e

kill "$WATCHDOG_PID" 2>/dev/null || true

echo "== final sync: models + results + log =="
aws s3 sync "${WORKDIR}/models/thermal_seeds" "${S3_PREFIX}/models/" --region "$AWS_REGION" || true
if [[ -f "${WORKDIR}/results/thermal_bfdd_seeds.json" ]]; then
  aws s3 cp "${WORKDIR}/results/thermal_bfdd_seeds.json" "${S3_PREFIX}/results/thermal_bfdd_seeds.json" --region "$AWS_REGION" || true
fi
aws s3 cp "$LOG_FILE" "${S3_PREFIX}/thermal_seeds.log" --region "$AWS_REGION" || true
kill "$SYNC_PID" 2>/dev/null || true

echo "== thermal seed experiment complete (run_exit=${RUN_EXIT}) =="
# trap EXIT runs shutdown_now from here.
