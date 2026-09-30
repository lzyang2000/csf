# SPDX-License-Identifier: Apache-2.0
"""MHFMotionRep: denormalised 263-d HumanML3D features to kimodo's motion dict (22 joints, Y-up)."""
import numpy as np
import pytest
import torch

from csf.backbones.motionhiflow.motion_rep import MHFMotionRep

CONTRACT_KEYS = ["local_rot_mats", "global_rot_mats", "posed_joints", "root_positions",
                 "smooth_root_pos", "foot_contacts", "global_root_heading"]
J = 22


def _fake_x263(T: int) -> np.ndarray:
    """Root at 0.9 m turning at 0.05 rad/frame, with fixed foot-contact flags."""
    x = np.zeros((T, 263), dtype=np.float32)
    x[:, 3] = 0.9
    x[:, 0] = 0.05
    x[:, 259] = 1.0
    x[:, 261] = 1.0
    return x


@pytest.fixture
def rep():
    return MHFMotionRep(joints_num=J, fps=20)


def test_attributes(rep):
    assert rep.fps == 20 and rep.joints_num == J and rep.size_dict == {"pose": 263}


def test_inverse_keys_and_shapes(rep):
    T = 6
    out = rep.inverse(_fake_x263(T), is_normalized=False, return_numpy=True)
    for key in CONTRACT_KEYS:
        assert key in out, f"missing key: {key}"
        assert np.isfinite(out[key]).all(), key
    assert out["posed_joints"].shape == (T, J, 3)
    assert out["root_positions"].shape == (T, 3)
    assert out["local_rot_mats"].shape == (T, J, 3, 3)
    assert out["global_rot_mats"].shape == (T, J, 3, 3)
    assert out["global_root_heading"].shape == (T,)


def test_root_height_is_the_y_axis(rep):
    x = np.zeros((4, 263), dtype=np.float32)
    x[:, 3] = 0.9
    np.testing.assert_allclose(rep.inverse(x, return_numpy=True)["root_positions"][:, 1], 0.9, atol=1e-5)


def test_foot_contacts_come_from_the_last_four_channels(rep):
    x = np.zeros((3, 263), dtype=np.float32)
    x[:, 259:263] = [1.0, 0.5, 0.0, 0.75]
    np.testing.assert_allclose(rep.inverse(x, return_numpy=True)["foot_contacts"][0],
                               [1.0, 0.5, 0.0, 0.75], atol=1e-6)


def test_rotations_are_identity_without_a_rest_pose(rep):
    out = rep.inverse(_fake_x263(3), return_numpy=True)
    np.testing.assert_allclose(out["local_rot_mats"], np.broadcast_to(np.eye(3), (3, J, 3, 3)), atol=1e-6)


def test_smooth_root_pos_equals_root_positions(rep):
    out = rep.inverse(_fake_x263(5), return_numpy=True)
    np.testing.assert_array_equal(out["smooth_root_pos"], out["root_positions"])


def test_return_types_and_tensor_input(rep):
    out = rep.inverse(_fake_x263(4), return_numpy=False)
    for key in CONTRACT_KEYS:
        assert isinstance(out[key], torch.Tensor), key
    x = torch.zeros(4, 263)
    x[:, 3] = 0.9
    assert rep.inverse(x, return_numpy=True)["posed_joints"].shape == (4, J, 3)


def test_only_single_denormalised_clips_are_accepted(rep):
    with pytest.raises((AssertionError, ValueError)):
        rep.inverse(np.zeros((263,), dtype=np.float32))
    with pytest.raises((AssertionError, ValueError)):
        rep.inverse(np.zeros((2, 5, 263), dtype=np.float32))
    with pytest.raises(AssertionError):
        rep.inverse(_fake_x263(3), is_normalized=True)


def test_heading_accumulates_the_yaw_velocity(rep):
    x = np.zeros((10, 263), dtype=np.float32)
    x[:, 0] = 0.1
    heading = rep.inverse(x, return_numpy=True)["global_root_heading"]
    np.testing.assert_allclose(heading, 0.1 * np.arange(10), atol=1e-5)


def test_xz_root_integration_without_rotation(rep):
    T = 6
    x = np.zeros((T, 263), dtype=np.float32)
    x[:, 1] = 0.1
    root = rep.inverse(x, return_numpy=True)["root_positions"]
    np.testing.assert_allclose(root[:, 0], 0.1 * np.arange(T), atol=1e-5)
    np.testing.assert_allclose(root[:, 2], 0.0, atol=1e-5)


# ------------------------------------------------------------ rotation recovery
def _smplx():
    from kimodo.skeleton.definitions import SMPLXSkeleton22

    return SMPLXSkeleton22(load=True)


def test_geometric_global_rotation_recovery():
    """G[j] @ (rest[c] - rest[j]) is parallel to the posed bone for every joint j with child c."""
    skel = _smplx()
    rest = skel.neutral_joints.float()
    parents = skel.joint_parents.long()
    mr = MHFMotionRep(joints_num=J, fps=20, rest_joints=rest, joint_parents=parents)

    # A valid pose: FK of the rest skeleton through random local rotations.
    torch.manual_seed(0)
    T = 8
    ax = 0.7 * torch.randn(T, J, 3)
    th = ax.norm(dim=-1, keepdim=True)
    axn = ax / (th + 1e-8)
    K = torch.zeros(T, J, 3, 3)
    K[..., 0, 1], K[..., 0, 2] = -axn[..., 2], axn[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axn[..., 2], -axn[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axn[..., 1], axn[..., 0]
    local = torch.eye(3) + torch.sin(th)[..., None] * K + (1 - torch.cos(th)[..., None]) * (K @ K)
    G_true = torch.eye(3).repeat(T, J, 1, 1)
    posed = torch.zeros(T, J, 3)
    for j in range(J):
        p = int(parents[j])
        if p < 0:
            G_true[:, j] = local[:, j]
        else:
            G_true[:, j] = G_true[:, p] @ local[:, j]
            posed[:, j] = posed[:, p] + torch.einsum("tij,j->ti", G_true[:, p], rest[j] - rest[p])

    G = mr._recover_global_rotations(posed)
    assert torch.det(G).min().item() > 0.99
    assert (G @ G.transpose(-1, -2) - torch.eye(3)).abs().max().item() < 1e-3
    for j in range(J):
        for c in mr._children[j]:
            pred = torch.einsum("tij,tj->ti", G[:, j], (rest[c] - rest[j]).expand(T, 3))
            target = posed[:, c] - posed[:, j]
            cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).min().item()
            assert cos > 0.999, f"joint {j}->{c} bone-direction cos={cos:.4f}"


def test_inverse_emits_real_rotations_with_a_rest_pose():
    skel = _smplx()
    x = np.random.RandomState(0).randn(12, 263).astype(np.float32)
    mr = MHFMotionRep(J, 20, rest_joints=skel.neutral_joints, joint_parents=skel.joint_parents)
    G = mr.inverse(x, return_numpy=False)["global_rot_mats"]
    assert G.shape == (12, J, 3, 3)
    assert (G - torch.eye(3)).abs().max().item() > 0.1
    assert torch.det(G).min().item() > 0.9
