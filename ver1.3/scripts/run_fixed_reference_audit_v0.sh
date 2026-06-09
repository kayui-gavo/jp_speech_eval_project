#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="${1:-data/audit/fixed_reference_manifest_v0.csv}"
OUT="${2:-outputs/fixed_reference_scoring_audit.csv}"
SUMMARY="${3:-reports/fixed_reference_scoring_audit_summary.md}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  echo "Create it first:" >&2
  echo "  cp data/audit/fixed_reference_manifest_template.csv data/audit/fixed_reference_manifest_v0.csv" >&2
  echo "Then replace placeholder rows with real audio paths before running the audit." >&2
  exit 2
fi

../.venv/bin/python scripts/audit_fixed_reference_scoring.py \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --summary-out "$SUMMARY"

echo "Audit CSV: $OUT"
echo "Audit summary: $SUMMARY"
