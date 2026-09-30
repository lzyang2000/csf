# SPDX-License-Identifier: Apache-2.0
"""Error paths of the MotionHiFlow loader and its import isolation (no checkpoint needed)."""
from __future__ import annotations

import sys
import types

import pytest

_CFG = {"cond_scale": 4.5, "time_steps": 12, "fps": 20, "joints_num": 22, "dataset": "t2m"}


def test_load_mhf_missing_dirs_raise(tmp_path):
    from csf.backbones.motionhiflow.loader import load_mhf_models

    cfg = dict(_CFG, vae_dir=str(tmp_path / "novae"), dit_dir=str(tmp_path / "nodit"),
               clip_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_motionhiflow_ckpts"):
        load_mhf_models(cfg, device="cpu")


def test_load_mhf_missing_config_raises(tmp_path):
    from csf.backbones.motionhiflow.loader import load_mhf_models

    (tmp_path / "vae").mkdir()
    (tmp_path / "dit").mkdir()
    cfg = dict(_CFG, vae_dir=str(tmp_path / "vae"), dit_dir=str(tmp_path / "dit"),
               clip_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_mhf_models(cfg, device="cpu")


def test_load_mhf_missing_checkpoint_raises(tmp_path):
    from csf.backbones.motionhiflow.loader import load_mhf_models

    for name in ("vae", "dit"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.yaml").write_text("{}\n")
    cfg = dict(_CFG, vae_dir=str(tmp_path / "vae"), dit_dir=str(tmp_path / "dit"),
               clip_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="net_best_fid.tar"):
        load_mhf_models(cfg, device="cpu")


def test_mhf_import_isolation_evicts_and_restores():
    from csf.backbones.motionhiflow.loader import _mhf_import_isolation

    outer = types.ModuleType("src")
    sys.modules["src"] = outer
    try:
        with _mhf_import_isolation():
            assert sys.modules.get("src") is not outer
            sys.modules["src.mhf_only"] = types.ModuleType("src.mhf_only")
        assert sys.modules.get("src") is outer
        assert "src.mhf_only" not in sys.modules
    finally:
        if sys.modules.get("src") is outer:
            del sys.modules["src"]
