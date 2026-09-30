# SPDX-License-Identifier: Apache-2.0
"""configs/backbones/motionhiflow.yaml carries what the loader, the registry and the fetch script read."""
import yaml

from csf.paths import BACKBONE_CONFIGS_DIR

CFG = BACKBONE_CONFIGS_DIR / "motionhiflow.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CFG.read_text())


def test_mhf_config_has_required_keys():
    cfg = _cfg()
    for key in ["vae_dir", "dit_dir", "clip_dir", "cond_scale", "time_steps", "fps",
                "joints_num", "dataset"]:
        assert key in cfg, f"missing key: {key}"


def test_mhf_config_motion_format_and_presets():
    cfg = _cfg()
    assert (cfg["fps"], cfg["joints_num"], cfg["dataset"]) == (20, 22, "t2m")
    assert (cfg["cond_scale"], cfg["time_steps"]) == (4.5, 12)


def test_mhf_checkpoint_paths_live_under_checkpoints_mhf():
    """The fetch script and existing checkpoint trees rely on these exact paths."""
    cfg = _cfg()
    assert cfg["vae_dir"] == "checkpoints/mhf/checkpoints/t2m_vae_agcn_16d"
    assert cfg["dit_dir"] == "checkpoints/mhf/checkpoints/t2m_tmdit_16d"
    assert cfg["clip_dir"] == "checkpoints/mhf/clip-vit-base-patch32"
