# Phase 3: Qwen2.5-VL-3B QLoRA Fine-Tune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** LoRA fine-tune Qwen2.5-VL-3B on the 15,004-image frozen train split so macro top-1 on the frozen test split beats the 0.472 CLIP zero-shot baseline (target ≥0.80), for ≤$10 total AWS spend.

**Strategy (user-mandated):** validate-locally-before-GPU; ask-before-every-launch; smoke run before full run. Instance: **g6.xlarge spot** (~$0.38/hr, us-east-1) per live price check — g5.xlarge fallback; re-price at launch (see `aws-gpu-training-budget` skill). Launch mechanics per official `launching-ec2-instance-with-best-practices` skill. Account `002559670021` (`--profile defectlens` ONLY), $10 budget alarms live.

**Budget ledger (plan):** S3 ~$0.10 · smoke run ~$0.30 · full run 4-6h ~$2.50 · GPU eval ~$0.50 → ~$3.50, reserve for one retry. HARD STOP at $10.

---

## Task 1: Training script (local-first design)

**Files:** `src/defectlens/train/__init__.py`, `src/defectlens/train/qlora.py`, `tests/test_train.py`

- Dataset: rows from `data/manifests/train.csv` → chat sample: user=[image + "What building defect is shown in this image? Answer with one of: crack, spalling, efflorescence, exposed rebar, corrosion stain, mold or algae, water damage, peeling paint, no defect."], assistant=humanized label. Labels masked to assistant tokens only.
- **Balanced sampling:** per-class weighted sampling (inverse frequency, capped at 20× oversample) — crack:6154 vs exposed_rebar:70 would otherwise bury rare classes; macro top-1 is the metric.
- Args: `--train-manifest --subset N --quant {4bit,none} --lora-r 16 --lora-alpha 32 --lr 1e-4 --epochs 1 --max-steps --batch-size --grad-accum --save-steps --output-dir --resume --seed 42`. `--quant 4bit` requires CUDA (bitsandbytes); `none` = plain LoRA fp16/fp32 for local MPS validation.
- LoRA targets: attention+MLP projections of the language model (freeze vision tower v1).
- Checkpointing: adapter + optimizer state via Trainer save_steps; must resume cleanly.
- TDD pure parts: prompt/label formatting, class-weight computation, manifest→dataset row mapping. Model paths validated in Task 3's local run.

## Task 2: VLM top-k eval harness

**Files:** `src/defectlens/eval/vlm_topk.py`, `tests/test_vlm_topk.py`

- Method (spec §5): per image, compute sequence log-likelihood of each of the 9 humanized class answers under the (fine-tuned or base) model with the same prompt; rank; reuse `metrics.py` macro top-1/top-3 + confusion matrix on frozen `test.csv`. NaN→null JSON like prior evals. `--adapter` (optional), `--subset`, `--out-name`.
- Batching: group by image, 9 scored continuations each (single forward per continuation is fine v1).
- TDD pure parts: answer-set construction, ranking from log-lik dict, output shape.

## Task 3: LOCAL validation (controller, free, MPS)

- `python -m defectlens.train.qlora --subset 96 --quant none --max-steps 20 --batch-size 1 --grad-accum 4` → loss decreases over 20 steps; checkpoint saves; `--resume` continues.
- `python -m defectlens.eval.vlm_topk --subset 24` (base model) → runs end-to-end, sane output.
- Fix anything found; only then AWS.
- **Outcome (2026-07-07):** gate passed at step 10 of 20 — loss 0.546→0.447, checkpoint-10
  saved with full resume state (adapter+optimizer+scheduler+RNG). Run stopped there at
  user's request (MPS ~10.4 min/step; steps 11-20 added no new information). Two fixes
  found by attempts 1-2: `--max-pixels` knob + bf16-on-MPS (fp32 3.75B weights OOM 18GB).
  **`--resume` verification moved to the Task 6 smoke run** (standard Trainer plumbing;
  ~$0.01 on L4 vs 35 min on MPS): smoke run must include a kill+resume-from-checkpoint
  step before it counts as passed. Local eval check ABANDONED after two macOS jetsam
  kills (kernel memory-pressure storm, ~1.5GB free, verified in system log 18:26:44) —
  the `vlm_topk --subset` plumbing check moves to the smoke run too (~$0.02 on L4).
  Lesson: check free-memory headroom before any local multi-GB run; `/usr/bin/log`
  (zsh shadows `log` with a builtin) shows jetsam kills.

## Task 4: Data packaging + S3

**Files:** `scripts/package_data.py`
- Tar exactly the manifest-referenced images (resolve symlinks): `train_images.tar` (~3-4GB), `test_images.tar` (~700MB) + manifests + configs → `s3://defectlens-phase3-002559670021/phase3/` (create bucket us-east-1, private, `--profile defectlens`).

## Task 5: Launch runbook + bootstrap

**Files:** `docs/aws/phase3-runbook.md`, `scripts/aws/launch_gpu.sh`, `scripts/aws/bootstrap.sh`
- Per official skill: least-privilege instance role (S3 bucket-scoped), SG with SSH from current IP only, DLAMI (PyTorch, us-east-1), encrypted gp3 100GB, tags, spot request g6.xlarge (re-price first; g5 fallback), IMDSv2.
- bootstrap.sh (user-data): pull repo tarball + data from S3, pip install project, run training with `--quant 4bit`, sync checkpoints to S3 every save, auto-shutdown on completion or 30min idle GPU (safety).
- **ASK USER before every launch. No exceptions.**

## Task 6: Smoke run (~$0.30, NEEDS USER OK)
- 100 steps 4bit on g6.xlarge spot; verify loss/throughput/checkpoint-to-S3; terminate; project full-run cost from measured steps/sec; report.

## Task 7: Full run + GPU eval (~$3, NEEDS USER OK)
- Full balanced epoch (max-steps from smoke throughput within budget); eval on test.csv on-GPU; upload adapter + results JSON to S3; terminate.

## Task 8: Land the result
- Download adapter; local eval spot-verify (subset) matches GPU numbers; README before/after table + confusion matrix; merge into serving as the classifier (replaces CLIP-fused ranking behind same API); rotate the IAM key (user reminder); memory update.

## Self-review
Spec §5 coverage ✓ (QLoRA r=16 attention+MLP, instruction format, log-lik top-k on frozen split, checkpoint/resume, merged-fp16 landing). Budget controls ✓. No launch without user OK ✓. Frozen split untouched ✓.
