# SPDX-License-Identifier: Apache-2.0
"""ECHO's 38-d motion feature to a kimodo motion dict on the 34-joint G1 skeleton.

Denormalised 38-d layout (Z-up, +X forward):

    [0:29]   joint_pos     29 hinge angles (rad), in ECHO's Isaac joint order
    [29:31]  root_vel_xy   per-frame planar root displacement (m)
    [31:32]  root_z        root height (m)
    [32:38]  root_rot_6d   continuous 6-d root orientation

The root trajectory is recovered with ECHO's own
`utils.robot_npz_utils.reshape_generated_motion_38d` (with the smoothing and
static-start options ECHO recommends), so clips match ECHO's native decoding.
The joint angles are then reordered from Isaac order to the MuJoCo order of
the G1 MJCF, packed into ``[T, 36]`` qpos rows (root xyz, root quat wxyz, 29
angles) and converted by kimodo's `MujocoQposConverter`, which maps Z-up to
kimodo's Y-up frame, adds kimodo's end-effector joints and runs FK.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from kimodo.exports.mujoco import MujocoQposConverter

from .fk import N_QPOS

# ECHO's feature stores joints in Isaac order (left/right/waist interleaved by
# depth); the G1 MJCF uses the sequential order (left leg, right leg, waist,
# left arm, right arm).  Same remap as ECHO_CODE/deploy/src/text_to_motion.py.
_ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
#: joint_pos_mujoco[:, i] = joint_pos_isaac[:, _ISAAC_TO_MUJOCO[i]]
_ISAAC_TO_MUJOCO = np.array([_ISAAC_JOINT_ORDER.index(j) for j in MUJOCO_JOINT_ORDER])

# ECHO's static-start option needs more frames than static_frames + blend_frames.
_STATIC_START_MIN_FRAMES = 10


class EchoMotionRep:
    """Decode ECHO's denormalised 38-d feature into kimodo's motion dict.

    Args:
        skeleton: kimodo's `G1Skeleton34` (``build_skeleton(34)``).
        fps: Frame rate of ECHO's output.
        mujoco_xml: The G1 MJCF the qpos rows refer to; None uses kimodo's
            default (`g1skel34/xml/g1.xml`).

    Attributes:
        fps: Frame rate.
        size_dict: ``{"pose": 38}``.
    """

    # Foot-contact thresholds: height above the clip's lowest foot point (m)
    # and horizontal foot speed (m/s).
    _FC_HEIGHT_MARGIN = 0.07
    _FC_VEL_THRESH = 0.30

    def __init__(self, skeleton, fps: int = 50, mujoco_xml: Optional[str] = None):
        self._skeleton = skeleton
        self.fps = fps
        self.size_dict: Dict[str, int] = {"pose": 38}
        self._mujoco_xml = None if mujoco_xml is None else str(mujoco_xml)
        self._converter: Optional[MujocoQposConverter] = None
        self._reshape_fn = None

    def _get_converter(self) -> MujocoQposConverter:
        if self._converter is None:
            if self._mujoco_xml is None:
                self._converter = MujocoQposConverter(self._skeleton)
            else:
                self._converter = MujocoQposConverter(self._skeleton, xml_path=self._mujoco_xml)
        return self._converter

    def _reshape_38d(self, x38: np.ndarray) -> Dict[str, np.ndarray]:
        """``joint_pos`` [T, 29], ``root_pos`` [T, 3] and ``root_rot`` [T, 4] via ECHO's decoder."""
        if self._reshape_fn is None:
            from .loader import _echo_import_isolation, _ensure_echo_on_path  # noqa: PLC0415

            with _echo_import_isolation():
                _ensure_echo_on_path()
                from utils.robot_npz_utils import (  # type: ignore[import]  # noqa: PLC0415
                    reshape_generated_motion_38d,
                )
            self._reshape_fn = reshape_generated_motion_38d
        x38 = np.asarray(x38, dtype=np.float64)
        return self._reshape_fn(
            x38,
            fps=int(self.fps),
            smooth=True,
            adaptive_smooth=True,
            static_start=x38.shape[0] > _STATIC_START_MIN_FRAMES,
        )

    def inverse(self, x38: Union[np.ndarray, torch.Tensor], is_normalized: bool = False,
                return_numpy: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Decode one clip.

        Args:
            x38: ``[T, 38]`` denormalised ECHO features (the model denormalises
                before calling this, so ``is_normalized`` must be False).
            is_normalized: Kept for kimodo's motion-rep signature; must be False.
            return_numpy: Return numpy arrays instead of torch tensors.

        Returns:
            kimodo's motion dict on the 34-joint G1, Y-up: ``posed_joints``
            [T, 34, 3], ``global_rot_mats`` / ``local_rot_mats`` [T, 34, 3, 3],
            ``root_positions`` / ``smooth_root_pos`` [T, 3], ``foot_contacts``
            [T, 4] and ``global_root_heading`` [T, 2] (cos, sin).
        """
        assert not is_normalized, "EchoMotionRep.inverse expects denormalised input"
        if isinstance(x38, torch.Tensor):
            x38 = x38.detach().cpu().numpy()
        x38 = np.asarray(x38, dtype=np.float64)
        assert x38.ndim == 2, f"expected one [T, 38] clip, got shape {x38.shape}"
        assert x38.shape[1] == 38, f"expected 38-d input, got {x38.shape[1]}-d"

        decoded = self._reshape_38d(x38)
        joint_pos = np.asarray(decoded["joint_pos"], dtype=np.float64)[:, _ISAAC_TO_MUJOCO]
        qpos = np.zeros((joint_pos.shape[0], N_QPOS), dtype=np.float64)
        qpos[:, :3] = np.asarray(decoded["root_pos"], dtype=np.float64)    # Z-up, MuJoCo frame
        qpos[:, 3:7] = np.asarray(decoded["root_rot"], dtype=np.float64)   # wxyz
        qpos[:, 7:] = joint_pos

        # float32 in, so the motion dict is kimodo's native dtype.
        motion = self._get_converter().qpos_to_motion_dict(
            qpos.astype(np.float32), float(self.fps),
            root_quat_w_first=True, mujoco_rest_zero=False,
        )
        motion["foot_contacts"] = self._foot_contacts(motion["posed_joints"])

        if return_numpy:
            return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                    for k, v in motion.items()}
        return motion

    def _foot_contacts(self, posed_joints) -> torch.Tensor:
        """Contacts ``[T, 4]`` for (left heel, left toe, right heel, right toe).

        A foot point is in contact when it is within `_FC_HEIGHT_MARGIN` of the
        clip's lowest foot point and moves slower than `_FC_VEL_THRESH`
        horizontally.  This replaces the converter's contacts, which
        under-detect on ECHO motion.
        """
        is_tensor = isinstance(posed_joints, torch.Tensor)
        joints = posed_joints.detach().cpu().numpy() if is_tensor else np.asarray(posed_joints)
        foot_idx = [
            int(self._skeleton.left_foot_joint_idx[0]),
            int(self._skeleton.left_foot_joint_idx[1]),
            int(self._skeleton.right_foot_joint_idx[0]),
            int(self._skeleton.right_foot_joint_idx[1]),
        ]
        feet = joints[:, foot_idx, :]                           # [T, 4, 3], Y-up
        T = feet.shape[0]
        height = feet[:, :, 1]
        speed = np.zeros((T, 4), dtype=np.float64)
        if T >= 2:
            step = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1) * float(self.fps)
            speed[1:] = step
            speed[0] = step[0]
        contact = (height < float(height.min()) + self._FC_HEIGHT_MARGIN) & (speed < self._FC_VEL_THRESH)
        contacts = torch.from_numpy(contact.astype(np.float32))
        return contacts.to(posed_joints.device) if is_tensor else contacts
