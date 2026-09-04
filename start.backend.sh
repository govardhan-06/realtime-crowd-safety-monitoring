#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_ARGS=(--config configs/pipeline/dev.toml)
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is not set; starting the local ephemeral backend." >&2
  BACKEND_ARGS+=(--ephemeral)
fi

exec venv/bin/python -m crowd_safety serve-api "${BACKEND_ARGS[@]}" "$@"
