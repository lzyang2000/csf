# SPDX-License-Identifier: Apache-2.0
"""Repository-anchored paths, so nothing depends on the working directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
BACKBONE_CONFIGS_DIR = CONFIGS_DIR / "backbones"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
