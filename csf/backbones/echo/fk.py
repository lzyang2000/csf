# SPDX-License-Identifier: Apache-2.0
"""MuJoCo forward kinematics for the Unitree G1 (29 DoF).

The MJCF is the one named by `mujoco_xml` in `configs/backbones/echo.yaml`
(kimodo's `g1skel34/xml/g1.xml`): a free-joint root plus 29 hinges, so
nq = 36 and nbody = 31 (world, pelvis and 29 links).  `EchoMotionRep` builds
qpos rows in this layout; `mujoco_g1_fk` evaluates them with MuJoCo directly,
which gives an FK independent of kimodo's skeleton.
"""
from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

#: qpos width: root translation (3) + root quaternion wxyz (4) + 29 hinge angles.
N_QPOS: int = 36


@lru_cache(maxsize=None)
def _load_model(xml_path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(xml_path)


def body_names(xml_path: str) -> list[str]:
    """Body names in MuJoCo order; index 0 is always ``"world"``."""
    model = _load_model(xml_path)
    return [model.body(i).name for i in range(model.nbody)]


def mujoco_g1_fk(qpos_seq: np.ndarray, xml_path: str) -> dict[str, np.ndarray]:
    """World-frame body poses for a qpos sequence.

    Args:
        qpos_seq: ``[T, nq]`` rows of ``[root_xyz(3), root_quat_wxyz(4),
            hinge angles (rad)]``.
        xml_path: Path to the MJCF (the model is cached per path).

    Returns:
        ``{"body_xpos": [T, nbody, 3], "body_xquat": [T, nbody, 4] (wxyz)}``.

    Raises:
        ValueError: The qpos width does not match the model's nq.
    """
    model = _load_model(xml_path)
    data = mujoco.MjData(model)
    T, width = qpos_seq.shape
    if width != model.nq:
        raise ValueError(f"qpos width mismatch: got {width}, model.nq={model.nq} (xml: {xml_path})")

    xpos = np.empty((T, model.nbody, 3), dtype=np.float64)
    xquat = np.empty((T, model.nbody, 4), dtype=np.float64)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)      # positions only: no dynamics, no contacts
        xpos[t] = data.xpos
        xquat[t] = data.xquat
    return {"body_xpos": xpos, "body_xquat": xquat}
