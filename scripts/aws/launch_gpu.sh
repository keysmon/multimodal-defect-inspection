#!/usr/bin/env bash
# Launch ONE spot GPU instance for DefectLens Phase 3 QLoRA training.
#
# Guard rails:
#   - refuses to run if any instance tagged Project=defectlens is already
#     pending/running (never double-bill)
#   - re-checks spot price at launch time (prices move) and picks the
#     cheaper of g6.xlarge / g5.xlarge unless $INSTANCE_TYPE / --instance-type
#     is set
#   - prints the launch plan + estimated $/hr and requires an interactive
#     "yes" UNLESS --yes is passed. --yes must only ever be used by a
#     controller AFTER a human has approved the launch — never as a default.
#
# Requires scripts/aws/setup_iam.sh and scripts/aws/setup_sg.sh to have been
# run first, and scripts/package_data.py --upload to have populated
# s3://<bucket>/phase3/.
#
# Usage:
#   scripts/aws/launch_gpu.sh --dry-run
#   scripts/aws/launch_gpu.sh --train-args "--max-steps 100 --save-steps 25"
#   scripts/aws/launch_gpu.sh --yes --train-args "..."     # controller, post-approval only
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-defectlens}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="${BUCKET:-defectlens-phase3-002559670021}"
ROLE_NAME="defectlens-gpu-role"
SG_NAME="defectlens-gpu-sg"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-100}"
IDLE_TIMEOUT_SEC="${IDLE_TIMEOUT_SEC:-1800}"
CHECKPOINT_SYNC_INTERVAL="${CHECKPOINT_SYNC_INTERVAL:-120}"

ASSUME_YES=false
DRY_RUN=false
TRAIN_ARGS="${TRAIN_ARGS:-}"
INSTANCE_TYPE="${INSTANCE_TYPE:-}"

usage() {
  cat <<USAGE
Usage: $0 [--yes] [--dry-run] [--instance-type TYPE] [--train-args "..."]

  --yes                 skip interactive confirmation. ONLY for a controller
                         to pass AFTER the user has approved this exact launch.
  --dry-run             append --dry-run to run-instances (no billing, no
                         instance created); expect "DryRunOperation" back.
  --instance-type TYPE  override instance type (default: cheaper of
                         g6.xlarge / g5.xlarge by current spot price).
  --train-args "..."    extra args forwarded to defectlens.train.qlora,
                         e.g. "--max-steps 100 --save-steps 25".
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) ASSUME_YES=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

echo "== guard rail: checking for already-running defectlens instances =="
RUNNING=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=defectlens" \
            "Name=instance-state-name,Values=pending,running" \
  --region "$REGION" --profile "$PROFILE" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n "$RUNNING" ]]; then
  echo "Refusing to launch: instance(s) already pending/running with tag Project=defectlens: ${RUNNING}" >&2
  echo "Terminate them first (this guard prevents double-billing)." >&2
  exit 1
fi
echo "none running — OK to proceed"

echo ""
echo "== spot price check (re-priced at launch time; prices move) =="
# --no-paginate is required: describe-spot-price-history auto-paginates by
# default, and the AWS CLI applies --query PER PAGE before concatenating
# text output — without it, SpotPriceHistory[0].SpotPrice comes back as one
# line PER PAGE (e.g. two prices instead of one), which breaks the `[[`
# comparison below.
G6_PRICE=$(aws ec2 describe-spot-price-history --instance-types g6.xlarge \
  --product-descriptions "Linux/UNIX" --region "$REGION" --profile "$PROFILE" --no-paginate \
  --query 'SpotPriceHistory[0].SpotPrice' --output text)
G5_PRICE=$(aws ec2 describe-spot-price-history --instance-types g5.xlarge \
  --product-descriptions "Linux/UNIX" --region "$REGION" --profile "$PROFILE" --no-paginate \
  --query 'SpotPriceHistory[0].SpotPrice' --output text)
[[ "$G6_PRICE" == "None" ]] && G6_PRICE=""
[[ "$G5_PRICE" == "None" ]] && G5_PRICE=""
echo "g6.xlarge spot: \$${G6_PRICE:-unknown}/hr"
echo "g5.xlarge spot: \$${G5_PRICE:-unknown}/hr"

if [[ -z "$INSTANCE_TYPE" ]]; then
  if [[ -n "$G6_PRICE" && ( -z "$G5_PRICE" || $(echo "${G6_PRICE} <= ${G5_PRICE}" | bc -l) -eq 1 ) ]]; then
    INSTANCE_TYPE="g6.xlarge"
    EST_PRICE="$G6_PRICE"
  else
    INSTANCE_TYPE="g5.xlarge"
    EST_PRICE="$G5_PRICE"
  fi
  echo "auto-selected: ${INSTANCE_TYPE} (cheaper current spot price)"
else
  if [[ "$INSTANCE_TYPE" == "g6.xlarge" ]]; then EST_PRICE="$G6_PRICE"; else EST_PRICE="$G5_PRICE"; fi
  echo "using explicit instance type: ${INSTANCE_TYPE}"
fi
EST_PRICE="${EST_PRICE:-unknown}"

echo ""
echo "== resolving latest DLAMI PyTorch (GPU, x86_64, Ubuntu) AMI =="
AMI_ID=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=x86_64" \
  --region "$REGION" --profile "$PROFILE" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  echo "No DLAMI PyTorch (GPU, x86_64, Ubuntu) AMI found in ${REGION}." >&2
  exit 1
fi
AMI_INFO=$(aws ec2 describe-images --image-ids "$AMI_ID" \
  --region "$REGION" --profile "$PROFILE" \
  --query 'Images[0].[Name,RootDeviceName]' --output text)
AMI_NAME=$(echo "$AMI_INFO" | cut -f1)
ROOT_DEVICE=$(echo "$AMI_INFO" | cut -f2)
echo "AMI: ${AMI_ID} (${AMI_NAME}), root device ${ROOT_DEVICE}"

echo ""
echo "== resolving network (default VPC + a subnet, existing SG/role) =="
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --region "$REGION" --profile "$PROFILE" --query 'Vpcs[0].VpcId' --output text)
# SUBNET_ID overridable: on InsufficientInstanceCapacity, retry with a subnet
# in an AZ the error message names as having capacity (AZ capacity is per-type
# and shifts hour to hour — us-east-1d had none for g6.xlarge on 2026-07-07).
if [[ -z "${SUBNET_ID:-}" ]]; then
  SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" \
    --region "$REGION" --profile "$PROFILE" \
    --query 'sort_by(Subnets, &AvailableIpAddressCount)[-1].SubnetId' --output text)
fi
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --region "$REGION" --profile "$PROFILE" --query 'SecurityGroups[0].GroupId' --output text)
if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  echo "Security group ${SG_NAME} not found — run scripts/aws/setup_sg.sh first." >&2
  exit 1
fi
if ! aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  echo "Instance profile ${ROLE_NAME} not found — run scripts/aws/setup_iam.sh first." >&2
  exit 1
fi
echo "VPC ${VPC_ID}, subnet ${SUBNET_ID}, SG ${SG_ID}, instance profile ${ROLE_NAME}"

echo ""
echo "== building user-data (bootstrap + injected env) =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BOOTSTRAP_FILE overrides which bootstrap script is appended (default: the
# Phase-3 QLoRA bootstrap — zero behavior change for phase-3). Point it at
# scripts/aws/bootstrap_thermal.sh for the Phase 5.6 thermal seed experiment.
BOOTSTRAP_FILE="${BOOTSTRAP_FILE:-${SCRIPT_DIR}/bootstrap.sh}"
if [[ ! -f "$BOOTSTRAP_FILE" ]]; then
  echo "BOOTSTRAP_FILE not found: ${BOOTSTRAP_FILE}" >&2
  exit 1
fi
USER_DATA_FILE=$(mktemp)
cleanup() { rm -f "$USER_DATA_FILE"; }
trap cleanup EXIT
{
  echo "#!/bin/bash"
  echo "export S3_BUCKET=\"${BUCKET}\""
  echo "export AWS_REGION=\"${REGION}\""
  echo "export TRAIN_ARGS=\"${TRAIN_ARGS}\""
  echo "export IDLE_TIMEOUT_SEC=\"${IDLE_TIMEOUT_SEC}\""
  echo "export EVAL_ARGS=\"${EVAL_ARGS:-}\""
  echo "export SMOKE_RESUME_ARGS=\"${SMOKE_RESUME_ARGS:-}\""
  echo "export CHECKPOINT_SYNC_INTERVAL=\"${CHECKPOINT_SYNC_INTERVAL}\""
  echo "export CKPT_SUBDIR=\"${CKPT_SUBDIR:-checkpoints}\""
  cat "${BOOTSTRAP_FILE}"
} > "$USER_DATA_FILE"

echo ""
echo "=================================================================="
echo " Launch plan"
echo "=================================================================="
printf " %-20s %s\n" "Instance type:" "${INSTANCE_TYPE} (spot, one-time)"
printf " %-20s %s\n" "Estimated price:" "\$${EST_PRICE}/hr"
printf " %-20s %s\n" "AMI:" "${AMI_ID} (${AMI_NAME})"
printf " %-20s %s\n" "Security group:" "${SG_ID} (${SG_NAME}, SSH from admin IP only)"
printf " %-20s %s\n" "Instance profile:" "${ROLE_NAME} (S3 ${BUCKET} only)"
printf " %-20s %s\n" "Root volume:" "${ROOT_VOLUME_GB}GB gp3, encrypted"
printf " %-20s %s\n" "IMDSv2:" "required"
printf " %-20s %s\n" "Region / subnet:" "${REGION} / ${SUBNET_ID}"
printf " %-20s %s\n" "Bootstrap:" "$(basename "${BOOTSTRAP_FILE}")"
printf " %-20s %s\n" "Train args:" "${TRAIN_ARGS:-<none>}"
printf " %-20s %s\n" "Idle-safety:" "shutdown after ${IDLE_TIMEOUT_SEC}s with no checkpoint progress"
echo "=================================================================="
echo " Tags: Project=defectlens, Phase=3, Name=defectlens-gpu"
echo "=================================================================="

if [[ "$DRY_RUN" != "true" && "$ASSUME_YES" != "true" ]]; then
  read -r -p "Proceed with this SPOT launch at ~\$${EST_PRICE}/hr? Type 'yes' to continue: " CONFIRM
  if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

RUN_ARGS=(
  ec2 run-instances
  --image-id "$AMI_ID"
  --instance-type "$INSTANCE_TYPE"
  --iam-instance-profile "Name=${ROLE_NAME}"
  --security-group-ids "$SG_ID"
  --subnet-id "$SUBNET_ID"
  --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"SpotInstanceType\":\"one-time\",\"InstanceInterruptionBehavior\":\"terminate\"}}"
  --block-device-mappings "[{\"DeviceName\":\"${ROOT_DEVICE}\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"Encrypted\":true,\"DeleteOnTermination\":true}}]"
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=defectlens},{Key=Phase,Value=3},{Key=Name,Value=defectlens-gpu}]" \
                        "ResourceType=volume,Tags=[{Key=Project,Value=defectlens},{Key=Phase,Value=3}]"
  --user-data "file://${USER_DATA_FILE}"
  --count 1
  --region "$REGION"
  --profile "$PROFILE"
)
if [[ "$DRY_RUN" == "true" ]]; then
  RUN_ARGS+=(--dry-run)
fi

echo ""
echo "== launching =="
set +e
OUTPUT=$(aws "${RUN_ARGS[@]}" 2>&1)
STATUS=$?
set -e
echo "$OUTPUT"

if [[ "$DRY_RUN" == "true" ]]; then
  if echo "$OUTPUT" | grep -q "DryRunOperation"; then
    echo ""
    echo "Dry-run OK: DryRunOperation means the call would have been authorized — permissions/params look sufficient."
    exit 0
  else
    echo ""
    echo "Dry-run did NOT return DryRunOperation — inspect the error above (permissions, quota, or malformed request)." >&2
    exit 1
  fi
fi

if [[ $STATUS -ne 0 ]]; then
  echo "Launch failed." >&2
  exit $STATUS
fi

INSTANCE_ID=$(echo "$OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])" 2>/dev/null || echo "<unknown>")
echo ""
echo "Instance launched: ${INSTANCE_ID}"
echo "Tail bootstrap/training progress (console output lags a few minutes):"
echo "  aws ec2 get-console-output --instance-id ${INSTANCE_ID} --region ${REGION} --profile ${PROFILE} --output text"
echo "Or once SSH'd in: tail -f /opt/defectlens-phase3/train.log"
echo "Terminate when done (also happens automatically on completion/idle):"
echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --profile ${PROFILE}"
