#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the pretrained MotionHiFlow models, CLIP and HumanML3D statistics into
# checkpoints/mhf/, and link CLIP where the DiT config expects it.
#
# What it pulls:
#   1. The t2m VAE and DiT from Hugging Face (heng-li/MotionHiFlow) into
#      $DEST/checkpoints/t2m_{vae_agcn_16d,tmdit_16d}/, matching vae_dir and
#      dit_dir in configs/backbones/motionhiflow.yaml.
#   2. The CLIP text encoder openai/clip-vit-base-patch32 from Hugging Face into
#      $DEST/clip-vit-base-patch32, symlinked to
#      third_party/MotionHiFlow/deps/clip-vit-base-patch32 (the path the DiT
#      config names as ./deps/clip-vit-base-patch32).
#   3. The HumanML3D 263-d normalisation mean/std from the upstream evaluator
#      bundle (Google Drive, via gdown), extracted to
#      third_party/MotionHiFlow/deps/evaluators/t2m/Comp_v6_KLD005/meta/{mean,std}.npy,
#      where csf/backbones/motionhiflow/loader.py reads them.
#
# Requires: huggingface_hub, gdown and unzip (pip install huggingface_hub gdown).
# Idempotent: anything already present is skipped.
set -euo pipefail

# Anchor on this script rather than `git rev-parse --show-toplevel`, which
# would resolve to a submodule root when run from inside third_party/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/checkpoints/mhf"
MHF_DIR="$REPO_ROOT/third_party/MotionHiFlow"
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=python
META_DIR="$MHF_DIR/deps/evaluators/t2m/Comp_v6_KLD005/meta"
EVAL_GDRIVE_ID="19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM"   # EVALUATOR_T2M_URL in upstream prepare.sh

mkdir -p "$DEST"

# 1 + 2: models and CLIP from Hugging Face.
if [[ -f "$DEST/checkpoints/t2m_tmdit_16d/checkpoints/net_best_fid.tar" \
   && -f "$DEST/checkpoints/t2m_vae_agcn_16d/checkpoints/net_best_fid.tar" \
   && -d "$DEST/clip-vit-base-patch32" ]]; then
    echo "MotionHiFlow models and CLIP already present; skipping the Hugging Face download."
else
    echo "Downloading the t2m VAE, DiT and CLIP from Hugging Face ..."
    "$PY" - "$DEST" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
snapshot_download("heng-li/MotionHiFlow", local_dir=dest,
                  allow_patterns=["checkpoints/t2m_vae_agcn_16d/*",
                                  "checkpoints/t2m_tmdit_16d/*"])
snapshot_download("openai/clip-vit-base-patch32",
                  local_dir=dest + "/clip-vit-base-patch32",
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors"])
print("Hugging Face download done.")
PYEOF
fi

# CLIP link at the path the DiT config names.
CLIP_LINK="$MHF_DIR/deps/clip-vit-base-patch32"
if [[ ! -e "$CLIP_LINK" ]]; then
    mkdir -p "$MHF_DIR/deps"
    ln -sfn "$DEST/clip-vit-base-patch32" "$CLIP_LINK"
    echo "Linked $CLIP_LINK -> $DEST/clip-vit-base-patch32"
fi

# 3: HumanML3D 263-d mean/std.
if [[ -f "$META_DIR/mean.npy" && -f "$META_DIR/std.npy" ]]; then
    echo "HumanML3D mean/std already present; skipping the evaluator download."
else
    if ! "$PY" -c "import gdown" 2>/dev/null; then
        echo "ERROR: gdown is required for the HumanML3D mean/std. Install: pip install gdown" >&2
        exit 1
    fi
    echo "Downloading the t2m evaluator bundle (for the 263-d mean/std) from Google Drive ..."
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT INT TERM
    "$PY" -m gdown "$EVAL_GDRIVE_ID" -O "$TMP/t2m_evaluator.zip"
    unzip -oq "$TMP/t2m_evaluator.zip" -d "$TMP/extracted"
    SRC_META="$(dirname "$(find "$TMP/extracted" -name mean.npy -path '*Comp_v6*' | head -1)")"
    if [[ -z "$SRC_META" || ! -f "$SRC_META/mean.npy" ]]; then
        echo "ERROR: could not locate Comp_v6_KLD005/meta/mean.npy in the evaluator bundle." >&2
        exit 1
    fi
    mkdir -p "$META_DIR"
    cp "$SRC_META/mean.npy" "$SRC_META/std.npy" "$META_DIR/"
    echo "Placed mean/std at $META_DIR."
fi

echo "All MotionHiFlow assets present:"
echo "  models: $DEST/checkpoints/t2m_{vae_agcn_16d,tmdit_16d}/"
echo "  clip:   $CLIP_LINK"
echo "  stats:  $META_DIR/{mean,std}.npy"
