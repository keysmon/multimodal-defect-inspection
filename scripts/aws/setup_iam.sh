#!/usr/bin/env bash
# Idempotent least-privilege IAM role + instance profile for the DefectLens
# Phase 3 GPU training box. Scoped to ONLY the phase3 S3 bucket:
# least privilege, data-to-box-via-S3 pattern.
#
# Usage: scripts/aws/setup_iam.sh
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-defectlens}"
REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="defectlens-gpu-role"
BUCKET="${BUCKET:-defectlens-phase3-002559670021}"

trust_policy() {
  cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON
}

inline_policy() {
  cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Phase3ObjectReadWrite",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    },
    {
      "Sid": "Phase3BucketList",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}"
    }
  ]
}
JSON
}

echo "== role: ${ROLE_NAME} =="
if aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  echo "already exists"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$(trust_policy)" \
    --description "Least-privilege role for DefectLens Phase 3 GPU training (S3 access to ${BUCKET} only)" \
    --tags Key=Project,Value=defectlens Key=Phase,Value=3 \
    --profile "$PROFILE" >/dev/null
  echo "created"
fi

echo "== inline policy: defectlens-phase3-s3 (scoped to ${BUCKET}) =="
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "defectlens-phase3-s3" \
  --policy-document "$(inline_policy)" \
  --profile "$PROFILE"
echo "applied (put-role-policy is idempotent — always overwrites to current bucket scope)"

echo "== instance profile: ${ROLE_NAME} =="
if aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  echo "already exists"
else
  aws iam create-instance-profile --instance-profile-name "$ROLE_NAME" --profile "$PROFILE" >/dev/null
  echo "created — waiting ~10s for IAM propagation before attaching role"
  sleep 10
fi

ATTACHED=$(aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" --profile "$PROFILE" \
  --query "InstanceProfile.Roles[?RoleName=='${ROLE_NAME}'] | length(@)" --output text)
if [[ "$ATTACHED" == "1" ]]; then
  echo "role already attached to instance profile"
else
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$ROLE_NAME" \
    --role-name "$ROLE_NAME" \
    --profile "$PROFILE"
  echo "attached role to instance profile"
fi

echo ""
echo "IAM setup complete: role=${ROLE_NAME} instance-profile=${ROLE_NAME} bucket=${BUCKET}"
