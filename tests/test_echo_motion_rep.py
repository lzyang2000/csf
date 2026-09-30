# SPDX-License-Identifier: Apache-2.0
"""EchoMotionRep: denormalised 38-d ECHO features to kimodo's 34-joint G1 motion dict."""
import numpy as np
import pytest

pytest.importorskip("scipy")                     # kimodo's G1Skeleton34 needs it

from kimodo.skeleton.registry import build_skeleton  # noqa: E402

CONTRACT_KEYS = ["local_rot_mats", "global_rot_mats", "posed_joints", "root_positions",
                 "smooth_root_pos", "foot_contacts", "global_root_heading"]
N_JOINTS = 34        # G1Skeleton34: 29 hinged links, the pelvis and kimodo's end effectors


def _rep(fps=50, mujoco_xml=None):
    from csf.backbones.echo.motion_rep import EchoMotionRep

    return EchoMotionRep(build_skeleton(34), fps=fps, mujoco_xml=mujoco_xml)


def _standing(T):
    """Standing still at 0.8 m with an identity root orientation (6-d = first two columns)."""
    x = np.zeros((T, 38), dtype=np.float32)
    x[:, 31] = 0.8
    x[:, 32] = 1.0
    x[:, 36] = 1.0
    return x


def test_inverse_returns_contract_keys_and_shapes():
    rep = _rep()
    assert rep.fps == 50 and rep.size_dict == {"pose": 38}
    T = 7
    out = rep.inverse(_standing(T), is_normalized=False, return_numpy=True)
    for key in CONTRACT_KEYS:
        assert key in out, f"missing key: {key}"
    assert out["posed_joints"].shape == (T, N_JOINTS, 3)
    assert out["root_positions"].shape == (T, 3)
    assert out["smooth_root_pos"].shape == (T, 3)
    assert out["global_rot_mats"].shape == (T, N_JOINTS, 3, 3)
    assert out["local_rot_mats"].shape == (T, N_JOINTS, 3, 3)
    assert out["foot_contacts"].shape == (T, 4)
    assert out["global_root_heading"].shape == (T, 2)
    for key in CONTRACT_KEYS:
        assert np.isfinite(out[key]).all(), key


def test_config_mjcf_gives_the_same_motion_as_kimodos_default():
    import yaml

    from csf.paths import BACKBONE_CONFIGS_DIR, REPO_ROOT

    xml = REPO_ROOT / yaml.safe_load((BACKBONE_CONFIGS_DIR / "echo.yaml").read_text())["mujoco_xml"]
    x = _standing(8)
    x[1:, 29] = 0.02
    a = _rep().inverse(x, return_numpy=True)
    b = _rep(mujoco_xml=str(xml)).inverse(x, return_numpy=True)
    np.testing.assert_allclose(a["posed_joints"], b["posed_joints"], atol=1e-6)


def test_forward_root_displacement_maps_to_kimodo_z():
    """MuJoCo +X (forward) is kimodo's +Z: 0.1 m per frame over 4 steps ends 0.4 m forward.

    Short clips (T <= 10) skip ECHO's static-start blend, so the integration is exact.
    """
    T = 5
    x = _standing(T)
    x[1:, 29] = 0.1
    root = _rep().inverse(x, return_numpy=True)["root_positions"]
    np.testing.assert_allclose(root[:, 2], 0.1 * np.arange(T), atol=1e-4)
    np.testing.assert_allclose(root[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(root[:, 1], 0.8, atol=1e-4)


def test_standing_still_touches_the_ground():
    out = _rep().inverse(_standing(8), return_numpy=True)
    assert out["foot_contacts"].sum() > 0


def test_only_single_clips_are_accepted():
    rep = _rep()
    with pytest.raises((AssertionError, ValueError)):
        rep.inverse(np.zeros((38,), dtype=np.float32))
    with pytest.raises((AssertionError, ValueError)):
        rep.inverse(np.zeros((2, 7, 38), dtype=np.float32))


def test_is_normalized_true_is_rejected():
    with pytest.raises(AssertionError):
        _rep().inverse(_standing(3), is_normalized=True, return_numpy=True)


def test_return_numpy_false_returns_torch():
    import torch

    out = _rep().inverse(_standing(6), is_normalized=False, return_numpy=False)
    for key in CONTRACT_KEYS:
        assert isinstance(out[key], torch.Tensor), key
