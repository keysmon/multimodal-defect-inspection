#!/usr/bin/env bash
# Idempotent 1-day S3 lifecycle expiration for the CPU async-job prefix
# (phase5/cpu-jobs/) on the shared artifacts bucket. The async /analyze path
# writes request payloads (base64 image + optional audio, up to ~13MB each) to
# cpu-jobs/in/ and results to cpu-jobs/out/ | cpu-jobs/err/; without expiry they
# accumulate unbounded at the submit rate. The bucket is NOT CDK-managed (it
# predates the CDK stacks and is imported by name), so this rule can't live in
# ApiStack - it lives here, version-controlled and reviewable, instead of being
# an invisible console click.
#
# Merge-aware: it fetches the bucket's current lifecycle config and UPSERTs the
# `expire-cpu-jobs` rule, so it never clobbers other rules. Requires jq.
#
# Usage: scripts/aws/put_cpu_jobs_lifecycle.sh
#   env: AWS_PROFILE_NAME (default defectlens), AWS_REGION (default ca-central-1),
#        BUCKET (default defectlens-phase3-ca-002559670021), PREFIX, DAYS
set -euo pipefail

PROFILE="${AWS_PROFILE_NAME:-defectlens}"
REGION="${AWS_REGION:-ca-central-1}"
BUCKET="${BUCKET:-defectlens-phase3-ca-002559670021}"
PREFIX="${PREFIX:-phase5/cpu-jobs/}"
DAYS="${DAYS:-1}"
RULE_ID="expire-cpu-jobs"

echo "Identity check:" >&2
aws sts get-caller-identity --profile "$PROFILE" >&2

# The rule to upsert (Filter by prefix; Expiration in DAYS days).
new_rule=$(jq -n --arg id "$RULE_ID" --arg prefix "$PREFIX" --argjson days "$DAYS" '{
  ID: $id, Status: "Enabled", Filter: {Prefix: $prefix}, Expiration: {Days: $days}
}')

# Fetch existing rules (tolerate "no lifecycle configuration" -> empty list),
# drop any prior rule with the same ID or prefix, then append ours.
existing=$(aws s3api get-bucket-lifecycle-configuration \
  --bucket "$BUCKET" --profile "$PROFILE" --region "$REGION" \
  --query 'Rules' --output json 2>/dev/null || echo '[]')

merged=$(jq -n --argjson existing "${existing:-[]}" --argjson rule "$new_rule" --arg prefix "$PREFIX" '
  {Rules: (($existing // [])
    | map(select(.ID != $rule.ID and (.Filter.Prefix // .Prefix // "") != $prefix))
    + [$rule])}')

echo "Applying lifecycle config to s3://$BUCKET (rule $RULE_ID: expire $PREFIX after ${DAYS}d):" >&2
echo "$merged" | jq . >&2

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET" --profile "$PROFILE" --region "$REGION" \
  --lifecycle-configuration "$merged"

echo "Done. Verify:" >&2
aws s3api get-bucket-lifecycle-configuration \
  --bucket "$BUCKET" --profile "$PROFILE" --region "$REGION" --output json
