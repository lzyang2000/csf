# SPDX-License-Identifier: Apache-2.0
"""configs/backbones/echo.yaml carries what the loader, the registry and the fetch script read."""
import yaml

from csf.paths import BACKBONE_CONFIGS_DIR, REPO_ROOT

CFG = BACKBONE_CONFIGS_DIR / "echo.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CFG.read_text())


def test_echo_config_has_required_keys():
    cfg = _cfg()
    for key in ["variant", "ckpt_dir", "data_root", "mujoco_xml",
                "num_denoising_steps", "diffuser_name", "fps"]:
        assert key in cfg, f"missing {key}"
    assert cfg["fps"] == 50
    assert cfg["variant"] in {"lite", "full", "transformer"}


def test_echo_checkpoint_paths_live_under_checkpoints_echo():
    """The fetch script and existing checkpoint trees rely on these exact paths."""
    cfg = _cfg()
    assert cfg["ckpt_dir"] == "checkpoints/echo/checkpoints/robotv2/robotv2_38d"
    assert cfg["data_root"] == "checkpoints/echo/robot_humanml_data_v2"


def test_echo_mujoco_xml_is_kimodos_g1_mjcf():
    cfg = _cfg()
    xml = REPO_ROOT / cfg["mujoco_xml"]
    assert xml.name == "g1.xml" and "g1skel34" in cfg["mujoco_xml"]
    assert xml.exists(), f"G1 MJCF not found: {xml}"
