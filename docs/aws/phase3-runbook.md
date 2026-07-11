# Phase 3 GPU Runbook — QLoRA Fine-Tune on AWS

Launch mechanics follow AWS EC2 launch best practices (least-privilege IAM,
hardened SG, encrypted gp3, IMDSv2, tagging); the cost/spot strategy is
documented inline below.

**Account:** `002559670021`, profile `defectlens` ONLY, region `us-east-1`.
**Budget:** ≤$10 total (S3 ~$0.10, smoke ~$0.30, full run ~$2.50, GPU eval
~$0.50 → ~$3.50 planned, reserve for one retry). HARD STOP at $10.
**Rule:** ask the user before every GPU launch. No exceptions. `--yes` on
`launch_gpu.sh` exists only for a controller to pass *after* that approval —
never as a default.

## Order of operations

1. **`scripts/aws/setup_iam.sh`** — creates `defectlens-gpu-role` (trust=ec2,
   inline policy scoped to `s3://defectlens-phase3-002559670021/*` only) +
   matching instance profile. Idempotent, $0.
2. **`scripts/aws/setup_sg.sh`** — creates `defectlens-gpu-sg` in the default
   VPC: SSH from your current public IP `/32` only, default (all) egress.
   Self-healing — re-run any time your IP changes; it revokes the stale rule.
   Idempotent, $0.
3. **`python scripts/package_data.py --upload`** — builds
   `dist/phase3/{train_images.tar,test_images.tar}` (exactly the images in
   `data/manifests/{train,test}.csv`, symlinks dereferenced), copies the
   manifests + `configs/label_mapping.yaml`, builds a `defectlens` wheel, and
   `aws s3 cp`'s the lot to `s3://defectlens-phase3-002559670021/phase3/`.
   Validate first with `--subset 20` (no `--upload`) — cheap correctness
   check before paying for the real ~4-5GB upload.
4. **Launch smoke run** (NEEDS USER OK — ask first):
   ```bash
   scripts/aws/launch_gpu.sh --train-args "--max-steps 100 --save-steps 25"
   ```
   ~100 steps, ~30 min, ~$0.20-0.30. Verifies throughput, loss decreasing,
   checkpoint-to-S3 sync. Confirms end-to-end before spending on a full run.
5. **Monitor:**
   ```bash
   aws ec2 get-console-output --instance-id <id> --region us-east-1 --profile defectlens --output text
   # or once SSH'd in (SG allows your IP):
   ssh ubuntu@<public-ip>   # DLAMI Ubuntu default user
   tail -f /opt/defectlens-phase3/train.log
   aws s3 ls s3://defectlens-phase3-002559670021/phase3/checkpoints/ --profile defectlens
   ```
6. **Terminate** (bootstrap.sh also auto-shuts-down on completion, failure,
   or 30 min of no checkpoint progress — this is a manual override / early
   stop):
   ```bash
   aws ec2 terminate-instances --instance-ids <id> --region us-east-1 --profile defectlens
   ```
7. **Launch full run** (NEEDS USER OK — ask again, project cost from the
   smoke run's measured steps/sec before asking):
   ```bash
   scripts/aws/launch_gpu.sh --train-args "--epochs 1 --save-steps 100"
   ```
8. Repeat monitor/terminate. On success, `bootstrap.sh` runs
   `defectlens.eval.vlm_topk` on the frozen test split automatically and
   uploads `eval_results.json` + `train.log` to
   `s3://defectlens-phase3-002559670021/phase3/`.

## Guard rails baked into the scripts

- `launch_gpu.sh` **refuses to run** if any instance tagged `Project=defectlens`
  is already `pending`/`running` — prevents double-billing.
- `launch_gpu.sh` re-checks the spot price for g6.xlarge vs g5.xlarge at
  launch time (prices move) and picks the cheaper one unless `--instance-type`
  / `$INSTANCE_TYPE` is set.
- `launch_gpu.sh` prints the full launch plan + estimated `$/hr` and requires
  an interactive `yes` unless `--yes` is passed.
- `bootstrap.sh` (user-data on the box) syncs checkpoints to S3 every 120s,
  has a 30-minute idle-safety shutdown (no checkpoint mtime change), and a
  `trap ... EXIT` that shuts the box down no matter how the script exits —
  a crash never leaves a paid instance running unattended.
- IAM role is scoped to `s3://defectlens-phase3-002559670021/*` only
  (`GetObject`/`PutObject`/`ListBucket`) — no other AWS access from the box.
- SSH is restricted to your current `/32`; all other inbound is closed.

## Cost table (2026-07-07 live spot prices, us-east-1)

| Item | Rate | Est. duration | Est. cost |
|---|---|---|---|
| g6.xlarge spot (preferred) | ~$0.45/hr | — | — |
| g5.xlarge spot (fallback) | ~$0.56/hr | — | — |
| S3 storage + transfer | pennies | — | ~$0.10 |
| Smoke run (100 steps) | g6 rate | ~30 min | ~$0.20-0.30 |
| Full run (1 epoch) | g6 rate | ~4-6h | ~$2.00-2.70 |
| GPU eval (frozen test split) | g6 rate | ~15-30 min | ~$0.10-0.25 |
| **Planned total** | | | **~$3.50** (reserve to $10) |

Re-run the price check before every launch — `launch_gpu.sh` does this
automatically and prints both prices before asking for confirmation.

## Teardown checklist

- [ ] `aws ec2 describe-instances --filters "Name=tag:Project,Values=defectlens" "Name=instance-state-name,Values=pending,running,stopping,stopped" --region us-east-1 --profile defectlens` — confirm **nothing** is left running or stopped (stopped instances still bill EBS).
- [ ] `aws ec2 terminate-instances --instance-ids <id> ...` for anything found.
- [ ] Confirm the adapter + `eval_results.json` + `train.log` landed in
      `s3://defectlens-phase3-002559670021/phase3/` before terminating (final
      sync happens in `bootstrap.sh`, but verify).
- [ ] Download the adapter locally (Task 8) once training is accepted.
- [ ] Optional cleanup once Phase 3 is fully landed: empty + delete the S3
      bucket (`aws s3 rm s3://defectlens-phase3-002559670021 --recursive --profile defectlens`
      then `aws s3api delete-bucket --bucket defectlens-phase3-002559670021 --profile defectlens`),
      and detach/delete `defectlens-gpu-role` / `defectlens-gpu-sg` if the
      project is done with GPU training.
- [ ] Rotate the `defectlens` IAM user's access key (reminder from Task 8 —
      out of scope here, don't forget it later).

## References

- Spot pricing strategy, instance selection (g6 vs g5 vs g4dn),
  smoke-before-full, and checkpoint-to-S3 are encoded in the scripts under
  `scripts/aws/` and explained where used.
- This runbook covers the Phase 3 fine-tune's infrastructure only (data
  packaging through adapter landing); training methodology is in the README.

## Known gotcha (found while testing `launch_gpu.sh --dry-run`)

`aws ec2 describe-spot-price-history` auto-paginates by default, and the AWS
CLI applies `--query` **per page** before concatenating `--output text` —
without `--no-paginate`, `SpotPriceHistory[0].SpotPrice` returns one line
*per page* (e.g. two prices instead of one) instead of a single scalar. Both
price lookups in `launch_gpu.sh` pass `--no-paginate` to avoid this; if you
add any other `describe-*`-with-`--query` call to these scripts, check
whether it needs the same flag.
