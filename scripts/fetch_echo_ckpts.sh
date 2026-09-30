#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the pretrained ECHO checkpoints from ModelScope into checkpoints/echo/ (~9.5 GB).
#
# Idempotent. The skip check is the weights file the loader actually opens:
# csf/backbones/echo/loader.py needs <ckpt_dir>/opt.txt and
# <ckpt_dir>/model/latest.tar, with ckpt_dir read from configs/backbones/echo.yaml.
# The clone goes to a temporary directory and is moved into place only once the
# weights are verified, so an interrupted run never leaves a partial tree that
# the check would accept. The clone's .git is removed afterwards so no nested
# repository ends up inside the (gitignored) checkpoint tree.
#
# Requires: git with git-lfs (all ECHO weights are LFS objects).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/checkpoints/echo"
ECHO_YAML="$ROOT/configs/backbones/echo.yaml"
REPO_URL="https://www.modelscope.cn/Hzzzz001/ECHO.git"

yaml_get() {  # yaml_get <key>: the first bare scalar, comments stripped
  sed -n "s/^$1:[[:space:]]*\([^[:space:]#]*\).*/\1/p" "$ECHO_YAML" | head -1
}
CKPT_REL="$(yaml_get ckpt_dir)"
DATA_REL="$(yaml_get data_root)"
[ -n "$CKPT_REL" ] || { echo "ERROR: no ckpt_dir in $ECHO_YAML" >&2; exit 1; }
case "$CKPT_REL" in
  checkpoints/echo/*) ;;
  *) echo "ERROR: ckpt_dir ($CKPT_REL) must live under checkpoints/echo/ for this script." >&2
     exit 1 ;;
esac
SENTINEL="$ROOT/$CKPT_REL/model/latest.tar"
STATS="$ROOT/${DATA_REL:-checkpoints/echo/robot_humanml_data_v2}/Mean_38d.npy"

if [ -s "$SENTINEL" ] && [ -s "$STATS" ]; then
  echo "ECHO checkpoints already present ($SENTINEL)"; exit 0
fi
if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "ERROR: $DEST exists but is incomplete (no $SENTINEL)." >&2
  echo "       Inspect it, then 'rm -rf $DEST' and re-run." >&2
  exit 1
fi

mkdir -p "$ROOT/checkpoints"
TMP="$(mktemp -d "$ROOT/checkpoints/.echo-fetch.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT INT TERM
echo "Cloning ModelScope Hzzzz001/ECHO into $DEST (~9.5 GB, LFS) ..."
git clone "$REPO_URL" "$TMP/echo"

# A clone without git-lfs (or with GIT_LFS_SKIP_SMUDGE=1) leaves ~130 B pointer files.
TMP_SENTINEL="$TMP/echo/${CKPT_REL#checkpoints/echo/}/model/latest.tar"
if [ ! -s "$TMP_SENTINEL" ] || [ "$(wc -c < "$TMP_SENTINEL")" -lt 1048576 ]; then
  echo "LFS payload missing; running 'git lfs pull' ..."
  git -C "$TMP/echo" lfs pull
fi
if [ ! -s "$TMP_SENTINEL" ] || [ "$(wc -c < "$TMP_SENTINEL")" -lt 1048576 ]; then
  echo "ERROR: $TMP_SENTINEL is still an LFS pointer. Install git-lfs and re-run." >&2
  exit 1
fi

rm -rf "$TMP/echo/.git"
# DEST is empty or absent (checked above); remove it so `mv` does not nest the clone inside.
if [ -d "$DEST" ]; then rmdir "$DEST"; fi
mv "$TMP/echo" "$DEST"
echo "Done. Weights: $SENTINEL"
echo "      Stats:   $STATS"
echo "      Variants under $DEST/checkpoints/robotv2/ (38d, 38d_lite, 38d_transformer)."
