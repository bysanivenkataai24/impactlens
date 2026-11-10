#!/usr/bin/env bash
# Assemble the files a Hugging Face Docker Space needs into one directory.
#
#   scripts/prepare_space.sh build/space
#   hf upload spaces/<owner>/<space> build/space . --commit-message "Deploy <git sha>"
#
# Copies only what the Docker build uses; never copies .env, data, runs or
# checkpoints. The Space README (with the Spaces YAML header) comes from
# deploy/huggingface/SPACE_README.md.
set -euo pipefail

out="${1:?usage: scripts/prepare_space.sh <output directory>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "error: $out exists and is not empty" >&2
  exit 1
fi
mkdir -p "$out"

cd "$root"
cp Dockerfile .dockerignore pyproject.toml uv.lock LICENSE "$out/"
cp deploy/huggingface/SPACE_README.md "$out/README.md"
mkdir -p "$out/src" "$out/frontend" "$out/models" "$out/schemas"
cp -R src/impactlens "$out/src/"
cp -R schemas/. "$out/schemas/"
cp models/README.md "$out/models/"
# Frontend sources only: dependencies are installed from the lock file in the build.
for item in package.json package-lock.json .npmrc index.html vite.config.ts tsconfig.json tsconfig.app.json tsconfig.node.json src public; do
  cp -R "frontend/$item" "$out/frontend/"
done
find "$out" -name '__pycache__' -type d -prune -exec rm -rf {} +

if grep -rIl --exclude=uv.lock --exclude=package-lock.json -E '(hf_[A-Za-z0-9]{30,}|API_KEY=[^ ]{8,})' "$out" >/dev/null; then
  echo "error: something that looks like a credential was found in $out; aborting" >&2
  exit 1
fi
echo "Space files written to $out (from git $(git rev-parse --short HEAD 2>/dev/null || echo unknown))"
