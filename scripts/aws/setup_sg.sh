#!/usr/bin/env bash
# Idempotent hardened security group for the DefectLens Phase 3 GPU training
# box: SSH from the CURRENT public IP only, all egress (default). Per
# ~/.claude/skills/launching-ec2-instance-with-best-practices — least
# privilege, no 0.0.0.0/0 SSH.
#
# Self-healing: if re-run from a different IP (e.g. laptop moved networks),
# revokes the stale rule and authorizes the new one, so this stays safe to
# re-run before every launch.
#
# Usage: scripts/aws/setup_sg.sh
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-defectlens}"
REGION="${AWS_REGION:-us-east-1}"
SG_NAME="defectlens-gpu-sg"

VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --region "$REGION" --profile "$PROFILE" --query 'Vpcs[0].VpcId' --output text)
if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
  echo "No default VPC found in ${REGION} — create one or pass an explicit VPC." >&2
  exit 1
fi
echo "== default VPC: ${VPC_ID} =="

MY_IP=$(curl -s https://checkip.amazonaws.com)
if [[ -z "$MY_IP" ]]; then
  echo "Failed to determine current public IP via checkip.amazonaws.com" >&2
  exit 1
fi
CIDR="${MY_IP}/32"
echo "== current public IP: ${CIDR} =="

SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --region "$REGION" --profile "$PROFILE" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "DefectLens Phase 3 GPU training box - SSH from admin IP only" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" --profile "$PROFILE" \
    --query 'GroupId' --output text)
  echo "created security group ${SG_ID}"
  aws ec2 create-tags --resources "$SG_ID" \
    --tags Key=Name,Value="$SG_NAME" Key=Project,Value=defectlens Key=Phase,Value=3 \
    --region "$REGION" --profile "$PROFILE"
else
  echo "security group already exists: ${SG_ID}"
fi

EXISTING_SSH_CIDRS=$(aws ec2 describe-security-groups --group-ids "$SG_ID" \
  --region "$REGION" --profile "$PROFILE" \
  --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\` && ToPort==\`22\`].IpRanges[].CidrIp" \
  --output text)

for old_cidr in $EXISTING_SSH_CIDRS; do
  if [[ "$old_cidr" != "$CIDR" ]]; then
    echo "revoking stale SSH rule for ${old_cidr} (IP changed since last run)"
    aws ec2 revoke-security-group-ingress --group-id "$SG_ID" \
      --protocol tcp --port 22 --cidr "$old_cidr" \
      --region "$REGION" --profile "$PROFILE"
  fi
done

if echo "$EXISTING_SSH_CIDRS" | grep -qF "$CIDR"; then
  echo "SSH ingress already allows ${CIDR}"
else
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$CIDR" \
    --region "$REGION" --profile "$PROFILE" >/dev/null
  echo "authorized SSH from ${CIDR}"
fi

echo ""
echo "SG setup complete: ${SG_NAME} (${SG_ID}) in VPC ${VPC_ID}, SSH allowed from ${CIDR} only, egress unrestricted (default)."
