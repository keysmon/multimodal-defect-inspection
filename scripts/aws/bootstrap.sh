#!/bin/bash
# DefectLens Phase 3 GPU training bootstrap — runs as EC2 user-data on a
# Deep Learning AMI (PyTorch, Ubuntu, us-east-1) GPU spot instance.
#
# NOT meant to be run directly: scripts/aws/launch_gpu.sh prepends an
# `export` header (S3_BUCKET, AWS_REGION, TRAIN_ARGS) before this script's
# contents and passes the concatenation as --user-data.
#
# Expected env (set by the header launch_gpu.sh prepends):
#   S3_BUCKET                required, e.g. defectlens-phase3-002559670021
#   AWS_REGION                default us-east-1
#   TRAIN_ARGS                 extra CLI args forwarded to defectlens.train.qlora
#                               (e.g. "--max-steps 100 --save-steps 25")
#   IDLE_TIMEOUT_SEC           default 1800 (30 min) - shutdown if no
#                               checkpoint progress for this long
#   CHECKPOINT_SYNC_INTERVAL   default 120 - seconds between background
#                               checkpoint -> S3 syncs
set -euxo pipefail

: "${S3_BUCKET:?S3_BUCKET must be set by launch_gpu.sh}"
: "${TRAIN_ARGS:=}"
: "${AWS_REGION:=us-east-1}"
: "${IDLE_TIMEOUT_SEC:=1800}"
: "${CHECKPOINT_SYNC_INTERVAL:=120}"
# Per-run S3 checkpoint namespace. Distinct values isolate runs: the resume
# logic scans ONLY this prefix, so a smoke run can never resume a prior
# full run's checkpoints (the 2026-07-21 incident: a v2 smoke auto-resumed
# the completed v1 run at phase3/checkpoints/ and clobbered its adapter).
: "${CKPT_SUBDIR:=checkpoints}"
: "${EVAL_ARGS:=}"
: "${SMOKE_RESUME_ARGS:=}"

WORKDIR=/opt/defectlens-phase3
CKPT_DIR="${WORKDIR}/checkpoints"
LOG_FILE="${WORKDIR}/train.log"
S3_PREFIX="s3://${S3_BUCKET}/phase3"

mkdir -p "$WORKDIR" "$CKPT_DIR"

# Safety net: no matter how this script exits (success, training crash,
# unhandled error under `set -e`), always shut the box down so a bug never
# leaves a paid GPU instance running unattended.
# Also push the cloud-init log (all bootstrap output incl. early failures)
# to S3 before powering off — a bootstrap that dies before training would
# otherwise leave zero evidence anywhere reachable (learned 2026-07-07:
# first smoke attempt shut down in 4 min with nothing in S3 to diagnose).
shutdown_now() {
  aws s3 cp /var/log/cloud-init-output.log "${S3_PREFIX}/cloud-init-output.log" \
    --region "$AWS_REGION" || true
  sudo shutdown -h now || true
}
trap shutdown_now EXIT

echo "== syncing phase3 package from ${S3_PREFIX} =="
aws s3 sync "${S3_PREFIX}/" "${WORKDIR}/dist/" --region "$AWS_REGION"

echo "== extracting image tars + manifests/configs into a repo-shaped tree =="
mkdir -p "${WORKDIR}/repo/data/manifests" "${WORKDIR}/repo/configs"
tar -xf "${WORKDIR}/dist/train_images.tar" -C "${WORKDIR}/repo"
tar -xf "${WORKDIR}/dist/test_images.tar" -C "${WORKDIR}/repo"
# All packaged manifests (train/test + extras like test_v1_frozen.csv for
# the EVAL2_ARGS backward-compat pass).
cp "${WORKDIR}"/dist/manifests/*.csv "${WORKDIR}/repo/data/manifests/"
cp "${WORKDIR}/dist/configs/label_mapping.yaml" "${WORKDIR}/repo/configs/label_mapping.yaml"

echo "== activating DLAMI PyTorch environment =="
# DLAMI ships a conda env named "pytorch" (Ubuntu OSS-driver DLAMI, 2024+).
# Fall back to whatever python3 is on PATH if conda isn't where expected, so
# a future AMI naming change fails at pip-install time with a clear error
# rather than silently no-op-ing this block.
# DLAMI python env layout varies by AMI generation (2026 Ubuntu 24.04 image
# has NO /opt/conda — first smoke attempt died on `pip: command not found`).
# Don't assume a layout: activate conda if present, then probe candidate
# pythons for one that can import torch, and use ITS pip for everything.
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

echo "== installing project wheel + GPU-only deps =="
WHEEL=$(ls "${WORKDIR}/dist/"defectlens-*.whl | head -n1)
"$PY" -m pip install "$WHEEL"
"$PY" -m pip install bitsandbytes  # CUDA-only dep, not in pyproject.toml (lazy-imported)

# Spot-interruption self-healing: if a prior run of THIS training already
# synced checkpoints to S3, pull them down and resume instead of restarting.
# (Smoke-run artifacts live under a different prefix so they never match.)
if aws s3 ls "${S3_PREFIX}/${CKPT_SUBDIR}/" --region "$AWS_REGION" 2>/dev/null | grep -q "checkpoint-"; then
  echo "== prior checkpoints found in S3 — downloading and enabling --resume =="
  aws s3 sync "${S3_PREFIX}/${CKPT_SUBDIR}/" "$CKPT_DIR" --region "$AWS_REGION"
  TRAIN_ARGS="$TRAIN_ARGS --resume"
fi

echo "== starting background checkpoint sync (every ${CHECKPOINT_SYNC_INTERVAL}s) =="
(
  while true; do
    sleep "$CHECKPOINT_SYNC_INTERVAL"
    aws s3 sync "$CKPT_DIR" "${S3_PREFIX}/${CKPT_SUBDIR}/" --region "$AWS_REGION" || true
  done
) &
SYNC_PID=$!

echo "== starting idle-safety watchdog (shutdown after ${IDLE_TIMEOUT_SEC}s with no checkpoint progress) =="
(
  last_mtime=0
  idle_since=$(date +%s)
  while true; do
    sleep 60
    newest=$(find "$CKPT_DIR" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    newest="${newest:-0}"
    now=$(date +%s)
    if [[ "$newest" != "$last_mtime" ]]; then
      last_mtime="$newest"
      idle_since="$now"
    elif (( now - idle_since > IDLE_TIMEOUT_SEC )); then
      echo "idle watchdog: no checkpoint progress for ${IDLE_TIMEOUT_SEC}s, shutting down"
      sudo shutdown -h now
    fi
  done
) &
WATCHDOG_PID=$!

echo "== training =="
cd "${WORKDIR}/repo"
set +e
"$PY" -m defectlens.train.qlora \
  --quant 4bit \
  --train-manifest data/manifests/train.csv \
  --output-dir "$CKPT_DIR" \
  $TRAIN_ARGS \
  2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# Smoke-mode resume test: re-run training with --resume plus SMOKE_RESUME_ARGS
# (e.g. "--max-steps 110" — appended after TRAIN_ARGS so its --max-steps wins)
# to prove checkpoint-resume works on this stack before the full run relies
# on it for spot-interruption recovery.
if [[ -n "${SMOKE_RESUME_ARGS:-}" && "$TRAIN_EXIT" -eq 0 ]]; then
  echo "== smoke: resume-from-checkpoint test (${SMOKE_RESUME_ARGS}) =="
  set +e
  "$PY" -m defectlens.train.qlora \
    --quant 4bit \
    --train-manifest data/manifests/train.csv \
    --output-dir "$CKPT_DIR" \
    $TRAIN_ARGS $SMOKE_RESUME_ARGS --resume \
    2>&1 | tee -a "$LOG_FILE"
  TRAIN_EXIT=${PIPESTATUS[0]}
  set -e
fi

kill "$WATCHDOG_PID" 2>/dev/null || true

echo "== final checkpoint sync =="
aws s3 sync "$CKPT_DIR" "${S3_PREFIX}/${CKPT_SUBDIR}/" --region "$AWS_REGION" || true

if [[ "$TRAIN_EXIT" -eq 0 ]]; then
  echo "== training succeeded — running eval on frozen test split =="
  "$PY" -m defectlens.eval.vlm_topk \
    --test-manifest data/manifests/test.csv \
    --adapter "${CKPT_DIR}/adapter" \
    --out-dir "$WORKDIR" \
    --out-name eval_results.json \
    $EVAL_ARGS || echo "WARNING: eval step failed, see train.log for training result"
  if [[ -f "${WORKDIR}/eval_results.json" ]]; then
    aws s3 cp "${WORKDIR}/eval_results.json" "${S3_PREFIX}/eval_results.json" --region "$AWS_REGION"
  fi
  # Optional second eval on the same instance (e.g. the v1 backward-compat
  # pass over test_v1_frozen.csv - its images are a proven subset of
  # test.csv's, so they are already in the extracted tar).
  if [[ -n "${EVAL2_ARGS:-}" ]]; then
    echo "== running second eval (${EVAL2_ARGS}) =="
    "$PY" -m defectlens.eval.vlm_topk       --adapter "${CKPT_DIR}/adapter"       --out-dir "$WORKDIR"       $EVAL2_ARGS || echo "WARNING: second eval failed, see train.log"
    for f in "${WORKDIR}"/*.json "${WORKDIR}"/*.jsonl; do
      base=$(basename "$f")
      [[ "$base" == "eval_results.json" ]] && continue
      aws s3 cp "$f" "${S3_PREFIX}/${base}" --region "$AWS_REGION" || true
    done
  fi
else
  echo "== training FAILED (exit ${TRAIN_EXIT}) — skipping eval =="
fi

aws s3 cp "$LOG_FILE" "${S3_PREFIX}/train.log" --region "$AWS_REGION" || true
kill "$SYNC_PID" 2>/dev/null || true

echo "== bootstrap complete (train_exit=${TRAIN_EXIT}) =="
# trap EXIT runs shutdown_now from here.
