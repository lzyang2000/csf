#!/usr/bin/env bash
# Start the demo: scripts/serve.sh [port] [extra csf-demo args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-7860}"
shift || true
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Load LLM2Vec in-process. Set TEXT_ENCODER_MODE=auto (or api) and TEXT_ENCODER_URL
# to use a running kimodo text-encoder server instead.
export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
exec "$ROOT/.venv/bin/csf-demo" --port "$PORT" "$@"
