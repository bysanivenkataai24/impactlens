#!/usr/bin/env bash
# Assemble a Hugging Face *Static* Space with only the built review UI (no API, no model).
#
#   scripts/prepare_static_space.sh build/static-space
#   hf upload spaces/<owner>/<space> build/static-space . --commit-message "Deploy UI <git sha>"
set -euo pipefail
out="${1:?usage: scripts/prepare_static_space.sh <output directory>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "error: $out exists and is not empty" >&2
  exit 1
fi
(cd "$root/frontend" && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null)
mkdir -p "$out"
cp -R "$root/frontend/dist/." "$out/"
cp "$root/deploy/huggingface/STATIC_SPACE_README.md" "$out/README.md"
echo "Static Space files written to $out (from git $(git -C "$root" rev-parse --short HEAD))"
