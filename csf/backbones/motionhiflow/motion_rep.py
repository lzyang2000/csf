# SPDX-License-Identifier: Apache-2.0
"""MotionHiFlow's 263-d HumanML3D feature to a kimodo motion dict (22 joints, Y-up).

Denormalised 263-d layout (J = 22 joints, Y-up):

    [0]        root_rot_vel     yaw velocity (rad/frame)
    [1:3]      root_lin_vel_xz  planar root velocity in the root frame (m/frame)
    [3]        root_y           root height (m)
    [4:67]     ric_data         (J-1) x 3 root-relative joint positions
    [67:193]   rot_data         (J-1) x 6 joint rotations (6-d)
    [193:259]  local_vel        J x 3 joint velocities
    [259:263]  foot_contacts    4 contact flags

Joint positions are recovered with HumanML3D's `recover_from_ric`
(`third_party/MotionHiFlow/src/utils/motion_process.py`), re-implemented here
in plain torch to avoid that module's scipy and package-import side effects.
HumanML3D is already Y-up, as kimodo is, so no axis change is needed.

Given a rest pose (kimodo's `SMPLXSkeleton22`), per-joint global rotations for
mesh skinning are recovered geometrically from the joint positions, starting
from the SMPL-X rest pose.  HumanML3D's own `rot_data` is relative to a
different rest pose (arms down, where SMPL-X has arms out), so it would skin
the SMPL-X mesh incorrectly.  Without a rest pose the rotations are identity.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch


def _qinv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of unit quaternions (w, x, y, z) [*, 4]."""
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v [*, 3] by unit quaternions q [*, 4] (same leading dims)."""
    assert q.shape[:-1] == v.shape[:-1], f"leading dims mismatch: q={q.shape}, v={v.shape}"
    shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    uv = torch.cross(q[:, 1:], v, dim=1)
    uuv = torch.cross(q[:, 1:], uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(shape)


def _recover_root_rot_pos(data: torch.Tensor):
    """Root yaw quaternion [..., T, 4] and root position [..., T, 3] (HumanML3D)."""
    rot_vel = data[..., 0]
    angle = torch.zeros_like(rot_vel)
    angle[..., 1:] = rot_vel[..., :-1]
    angle = torch.cumsum(angle, dim=-1)

    quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    quat[..., 0] = torch.cos(angle)
    quat[..., 2] = torch.sin(angle)

    pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    pos = torch.cumsum(_qrot(_qinv(quat), pos), dim=-2)
    pos[..., 1] = data[..., 3]
    return quat, pos


def _recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """World joint positions ``[T, J, 3]`` from 263-d features (HumanML3D)."""
    quat, root = _recover_root_rot_pos(data)
    joints = data[..., 4:(joints_num - 1) * 3 + 4]
    joints = joints.view(joints.shape[:-1] + (-1, 3))
    joints = _qrot(_qinv(quat)[..., None, :].expand(joints.shape[:-1] + (4,)), joints)
    joints[..., 0] += root[..., 0:1]
    joints[..., 2] += root[..., 2:3]
    return torch.cat([root.unsqueeze(-2), joints], dim=-2)


class MHFMotionRep:
    """Decode MotionHiFlow's denormalised 263-d feature into kimodo's motion dict.

    Args:
        joints_num: Skeleton joints (22 for HumanML3D).
        fps: Frame rate of MotionHiFlow's output.
        rest_joints: Rest-pose joint positions ``[J, 3]`` (SMPL-X), enabling
            rotation recovery; None gives identity rotations.
        joint_parents: Parent index per joint ``[J]`` (-1 for the root).

    Attributes:
        fps: Frame rate.
        joints_num: Joint count.
        size_dict: ``{"pose": 263}``.
    """

    def __init__(self, joints_num: int = 22, fps: int = 20,
                 rest_joints: Union[np.ndarray, torch.Tensor, None] = None,
                 joint_parents: Union[np.ndarray, torch.Tensor, None] = None):
        self.joints_num = joints_num
        self.fps = fps
        self.size_dict: Dict[str, int] = {"pose": 263}
        self._rest_joints: Optional[torch.Tensor] = None
        self._joint_parents: Optional[torch.Tensor] = None
        self._children: list[list[int]] = []
        if rest_joints is not None and joint_parents is not None:
            self._rest_joints = torch.as_tensor(rest_joints).float().cpu()       # [J, 3]
            self._joint_parents = torch.as_tensor(joint_parents).long().cpu()    # [J]
            J = self._joint_parents.shape[0]
            self._children = [[] for _ in range(J)]
            for j in range(J):
                parent = int(self._joint_parents[j])
                if parent >= 0:
                    self._children[parent].append(j)

    # ------------------------------------------------------- rotation recovery
    @staticmethod
    def _rotation_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Minimal (swing) rotations ``[T, 3, 3]`` taking directions a [T, 3] onto b [T, 3]."""
        eps = 1e-8
        a = a / (a.norm(dim=-1, keepdim=True) + eps)
        b = b / (b.norm(dim=-1, keepdim=True) + eps)
        T = a.shape[0]
        device, dtype = a.device, a.dtype
        eye = torch.eye(3, device=device, dtype=dtype)
        axis_sin = torch.cross(a, b, dim=-1)
        cos = (a * b).sum(-1)
        sin = axis_sin.norm(dim=-1)
        out = eye.expand(T, 3, 3).clone()

        general = sin > eps                                       # Rodrigues
        if general.any():
            v = axis_sin[general]
            K = torch.zeros(v.shape[0], 3, 3, device=device, dtype=dtype)
            K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
            K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
            K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
            coef = ((1.0 - cos[general]) / (sin[general] ** 2)).view(-1, 1, 1)
            out[general] = eye + K + coef * torch.matmul(K, K)

        flip = (~general) & (cos < 0)                             # antiparallel: pi about a perpendicular
        if flip.any():
            af = a[flip]
            ref = torch.zeros_like(af)
            ref[:, 0] = 1.0
            alt = torch.zeros_like(af)
            alt[:, 1] = 1.0
            ref = torch.where((af[:, 0].abs() > 0.9).view(-1, 1), alt, ref)
            axis = torch.cross(af, ref, dim=-1)
            axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
            out[flip] = 2.0 * torch.einsum("ni,nj->nij", axis, axis) - eye
        return out

    @staticmethod
    def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """Unit quaternions (w, x, y, z) [*, 4] to rotation matrices [*, 3, 3]."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = torch.empty(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)
        R[..., 0, 0] = 1 - 2 * (y * y + z * z)
        R[..., 0, 1] = 2 * (x * y - z * w)
        R[..., 0, 2] = 2 * (x * z + y * w)
        R[..., 1, 0] = 2 * (x * y + z * w)
        R[..., 1, 1] = 1 - 2 * (x * x + z * z)
        R[..., 1, 2] = 2 * (y * z - x * w)
        R[..., 2, 0] = 2 * (x * z - y * w)
        R[..., 2, 1] = 2 * (y * z + x * w)
        R[..., 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    @staticmethod
    def _kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Least-squares rotations ``[T, 3, 3]`` with R[t] @ P[k] ~ Q[t, k] (det +1).

        P: ``[K, 3]`` rest child offsets; Q: ``[T, K, 3]`` posed child offsets.
        """
        H = torch.einsum("kc, tkd -> tcd", P, Q)
        U, _S, Vh = torch.linalg.svd(H)
        D = torch.eye(3, device=P.device, dtype=P.dtype).expand(H.shape[0], 3, 3).clone()
        D[:, 2, 2] = torch.det(torch.matmul(U, Vh))
        return torch.matmul(torch.matmul(Vh.transpose(-1, -2), D), U.transpose(-1, -2))

    def _recover_global_rotations(self, posed_joints: torch.Tensor,
                                  root_rot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-joint global rotations ``[T, J, 3, 3]`` from world joint positions.

        One pass over the kinematic tree (parents precede children):

        * root: `root_rot` when given (HumanML3D's analytic yaw, which is clean
          and upright), else Kabsch on its children;
        * joints with two or more children: Kabsch over the child offsets;
        * joints with one child: the swing aiming the parent-carried rest bone
          at the posed bone (twist inherited from the parent);
        * leaves: the parent's rotation.
        """
        device, dtype = posed_joints.device, posed_joints.dtype
        rest = self._rest_joints.to(device)
        T, J, _ = posed_joints.shape
        eye = torch.eye(3, device=device, dtype=dtype)
        G = eye.expand(T, J, 3, 3).clone()
        root_idx = int((self._joint_parents < 0).nonzero()[0].item())

        for j in range(J):
            if j == root_idx and root_rot is not None:
                G[:, j] = root_rot.to(device=device, dtype=dtype)
                continue
            kids = self._children[j]
            parent = int(self._joint_parents[j])
            if len(kids) >= 2:
                G[:, j] = self._kabsch(rest[kids] - rest[j], posed_joints[:, kids] - posed_joints[:, j:j + 1])
            elif len(kids) == 1:
                c = kids[0]
                parent_g = G[:, parent] if parent >= 0 else eye.expand(T, 3, 3)
                carried = torch.einsum("tij,tj->ti", parent_g, (rest[c] - rest[j]).expand(T, 3))
                swing = self._rotation_between(carried, posed_joints[:, c] - posed_joints[:, j])
                G[:, j] = torch.matmul(swing, parent_g)
            elif parent >= 0:
                G[:, j] = G[:, parent]
        return G

    # ----------------------------------------------------------------- decode
    def inverse(self, x263: Union[np.ndarray, torch.Tensor], is_normalized: bool = False,
                return_numpy: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Decode one clip.

        Args:
            x263: ``[T, 263]`` denormalised features (the model denormalises
                before calling this, so ``is_normalized`` must be False).
            is_normalized: Kept for kimodo's motion-rep signature; must be False.
            return_numpy: Return numpy float32 arrays instead of torch tensors.

        Returns:
            ``posed_joints`` [T, J, 3], ``global_rot_mats`` / ``local_rot_mats``
            [T, J, 3, 3], ``root_positions`` / ``smooth_root_pos`` [T, 3],
            ``foot_contacts`` [T, 4] (from the feature) and
            ``global_root_heading`` [T] (accumulated yaw, rad).
        """
        assert not is_normalized, "MHFMotionRep.inverse expects denormalised input"
        if isinstance(x263, torch.Tensor):
            x = x263.float()
        else:
            x = torch.from_numpy(np.asarray(x263, dtype=np.float32))
        assert x.ndim == 2, f"expected one [T, 263] clip, got shape {tuple(x.shape)}"
        T, dim = x.shape
        assert dim == 263, f"expected 263-d input, got {dim}-d"
        J = self.joints_num

        posed_joints = _recover_from_ric(x, J)
        root_positions = posed_joints[:, 0, :].clone()
        heading = torch.zeros_like(x[:, 0])
        heading[1:] = x[:-1, 0]
        heading = torch.cumsum(heading, dim=0)

        eye = torch.eye(3, dtype=torch.float32, device=x.device)
        if self._rest_joints is not None:
            quat, _ = _recover_root_rot_pos(x)
            root_rot = self._quat_to_matrix(_qinv(quat))       # the yaw recover_from_ric applies
            global_rot = self._recover_global_rotations(posed_joints, root_rot=root_rot).contiguous()
            parents = self._joint_parents.to(x.device)
            parent_g = global_rot.clone()
            for j in range(J):
                p = int(parents[j])
                parent_g[:, j] = global_rot[:, p] if p >= 0 else eye
            local_rot = torch.matmul(parent_g.transpose(-1, -2), global_rot).contiguous()
        else:
            local_rot = eye[None, None].expand(T, J, 3, 3).contiguous()
            global_rot = eye[None, None].expand(T, J, 3, 3).contiguous()

        out = {
            "local_rot_mats": local_rot,
            "global_rot_mats": global_rot,
            "posed_joints": posed_joints,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions.clone(),
            "foot_contacts": x[:, 259:263].clone(),
            "global_root_heading": heading,
        }
        if return_numpy:
            out = {k: v.detach().cpu().numpy() for k, v in out.items()}
        return out
