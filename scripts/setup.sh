#!/usr/bin/env bash
# Build the environment: submodules, a Python 3.12 venv, CUDA torch, kimodo, csf.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init third_party/kimodo third_party/ECHO_CODE \
  third_party/MotionHiFlow third_party/ardy

# CUDA 12.6 wheels; every install below re-pins them so no extra can upgrade torch.
TORCH=(torch==2.7.1 torchvision==0.22.1)
PIP=(uv pip install --python .venv/bin/python --torch-backend=cu126)

uv venv .venv --python 3.12
"${PIP[@]}" "${TORCH[@]}"
# kimodo and its demo extra (the viser fork the GUI is built on, LLM2Vec).
"${PIP[@]}" "${TORCH[@]}" -e "third_party/kimodo[demo]"
"${PIP[@]}" "${TORCH[@]}" -e ".[echo,mhf,fetch,dev]"

echo "Done. Start the demo with: scripts/serve.sh"
echo "Optional weights: scripts/fetch_echo_ckpts.sh, scripts/fetch_motionhiflow_ckpts.sh"
