#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <namespace/space-name>" >&2
  exit 2
fi

repo_id="$1"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir="$(mktemp -d)"

cleanup() {
  rm -rf "${bundle_dir}"
}
trap cleanup EXIT

"${root_dir}/.venv/bin/hf" auth whoami >/dev/null

rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv-kanade/' \
  --exclude '.DS_Store' \
  --exclude 'lambda_tradeoff.*' \
  --exclude 'plot_lambda_tradeoff.py' \
  "${root_dir}/" "${bundle_dir}/"

cp "${root_dir}/deploy/hf-space/README.md" "${bundle_dir}/README.md"

"${root_dir}/.venv/bin/hf" repos create "${repo_id}" \
  --repo-type space \
  --space-sdk docker \
  --public \
  --exist-ok

"${root_dir}/.venv/bin/hf" upload "${repo_id}" "${bundle_dir}" . \
  --repo-type space \
  --commit-message "Publish Japanese speech evaluation demo"

echo "Published Space: https://huggingface.co/spaces/${repo_id}"
