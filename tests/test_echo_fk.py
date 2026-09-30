# SPDX-License-Identifier: Apache-2.0
"""MuJoCo forward kinematics on the G1 MJCF named by configs/backbones/echo.yaml."""
import numpy as np
import pytest
import yaml

from csf.paths import BACKBONE_CONFIGS_DIR, REPO_ROOT

mujoco = pytest.importorskip("mujoco")

XML = REPO_ROOT / yaml.safe_load((BACKBONE_CONFIGS_DIR / "echo.yaml").read_text())["mujoco_xml"]

REAL_NQ = 36      # free-joint root (7) + 29 hinges
REAL_NBODY = 31   # world + pelvis + 29 links


def test_xml_exists():
    assert XML.exists(), f"G1 MJCF not found: {XML}"


def test_fk_shapes_and_root_placement():
    from csf.backbones.echo.fk import body_names, mujoco_g1_fk

    T = 5
    names = body_names(str(XML))
    assert len(names) == REAL_NBODY

    qpos = np.zeros((T, REAL_NQ), dtype=np.float64)
    qpos[:, 2] = 0.8                       # root height
    qpos[:, 3] = 1.0                       # identity quaternion (wxyz)
    qpos[:, 0] = np.linspace(0, 1, T)      # root moves along +x

    out = mujoco_g1_fk(qpos, str(XML))
    assert out["body_xpos"].shape == (T, REAL_NBODY, 3)
    assert out["body_xquat"].shape == (T, REAL_NBODY, 4)
    assert np.isfinite(out["body_xpos"]).all() and np.isfinite(out["body_xquat"]).all()
    # The pelvis (body 1) follows the root translation, frame by frame.
    assert out["body_xpos"][-1, 1, 0] > out["body_xpos"][0, 1, 0]
    assert (out["body_xpos"][1:, 1, 0] != out["body_xpos"][0, 1, 0]).any()


def test_nq_mismatch_raises():
    from csf.backbones.echo.fk import mujoco_g1_fk

    with pytest.raises(ValueError, match="qpos width"):
        mujoco_g1_fk(np.zeros((2, REAL_NQ + 1)), str(XML))


def test_body_names_content():
    from csf.backbones.echo.fk import body_names

    names = body_names(str(XML))
    assert names[0] == "world"
    assert names[1] == "pelvis"
    assert "left_knee_link" in names
    assert "right_wrist_yaw_link" in names


def test_model_caching_is_deterministic():
    from csf.backbones.echo.fk import mujoco_g1_fk

    qpos = np.zeros((3, REAL_NQ))
    qpos[:, 3] = 1.0
    np.testing.assert_array_equal(mujoco_g1_fk(qpos, str(XML))["body_xpos"],
                                  mujoco_g1_fk(qpos, str(XML))["body_xpos"])


def test_world_body_at_origin():
    from csf.backbones.echo.fk import mujoco_g1_fk

    qpos = np.zeros((1, REAL_NQ))
    qpos[:, 3] = 1.0
    np.testing.assert_allclose(mujoco_g1_fk(qpos, str(XML))["body_xpos"][0, 0], 0.0, atol=1e-9)


def test_n_qpos_matches_model():
    from csf.backbones.echo.fk import N_QPOS

    assert N_QPOS == mujoco.MjModel.from_xml_path(str(XML)).nq


def test_motion_rep_joint_order_matches_the_mjcf():
    """EchoMotionRep reorders ECHO's joints into the MJCF's hinge order; the two must agree."""
    from csf.backbones.echo.motion_rep import MUJOCO_JOINT_ORDER

    model = mujoco.MjModel.from_xml_path(str(XML))
    hinges = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    assert hinges == MUJOCO_JOINT_ORDER
