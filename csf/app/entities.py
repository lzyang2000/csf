# SPDX-License-Identifier: Apache-2.0
"""The protected person in the 3D scene.

When a person is in the scene, a human body stands where the generated strike
lands, facing the character.  The body is Kimodo's SOMA skin
(`kimodo/assets/skeletons/somaskel77/skin_standard.npz`), posed from its bind
A-pose into a relaxed stand by rotating the arms down, with linear blend
skinning in NumPy.  Scene frame: Y up, characters start facing +Z.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A neutral clay colour, distinct from the orange/blue characters.
HUMAN_COLOR = (196, 186, 172)
#: How far the arms rotate down from the bind A-pose, radians.
_ARM_DROP = np.deg2rad(32.0)
#: Distance from the character's root to where the person stands when the
#: clip has no clear strike, metres.
DEFAULT_STANDOFF = 1.0


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


@functools.lru_cache(maxsize=1)
def human_mesh() -> tuple[np.ndarray, np.ndarray]:
    """``(vertices [V, 3] float32, faces [F, 3] int32)`` of a standing adult, feet at y=0."""
    import kimodo  # noqa: PLC0415

    path = Path(kimodo.__file__).parent / "assets" / "skeletons" / "somaskel77" / "skin_standard.npz"
    skin = np.load(path)
    bind = skin["bind_vertices"].astype(np.float64)
    names = [str(n) for n in skin["rig_joint_names"]]
    rig = skin["bind_rig_transform"].astype(np.float64)

    # Per-joint rigid motion G_j: identity, except the arm chains rotate about
    # the shoulder (the "Arm" joint) around the forward axis.
    motion = np.tile(np.eye(4), (len(names), 1, 1))
    for side, sign in (("Left", -1.0), ("Right", 1.0)):
        pivot = rig[names.index(f"{side}Arm")][:3, 3]
        R = _rotation_about(np.array([0.0, 0.0, 1.0]), sign * _ARM_DROP)
        G = np.eye(4)
        G[:3, :3] = R
        G[:3, 3] = pivot - R @ pivot
        for j, name in enumerate(names):
            if name.startswith((f"{side}Arm", f"{side}ForeArm", f"{side}Hand")):
                motion[j] = G

    homogeneous = np.concatenate([bind, np.ones((bind.shape[0], 1))], axis=1)
    idx, w = skin["lbs_indices"], skin["lbs_weights"].astype(np.float64)
    posed = np.einsum("vkab,vb->vka", motion[idx][..., :3, :], homogeneous)
    vertices = (posed * w[..., None]).sum(axis=1)
    vertices[:, 1] -= vertices[:, 1].min()
    return vertices.astype(np.float32), skin["faces"].astype(np.int32)


@dataclass(frozen=True)
class Placement:
    """Where the person stands: ground position (x, z) and the direction they face."""

    x: float
    z: float
    facing: tuple[float, float]


def placement_from_motion(joints: np.ndarray, standoff: float = DEFAULT_STANDOFF) -> Placement:
    """Stand the person where the clip's strike is aimed, facing the character.

    The strike frame is the one where some joint reaches furthest from the
    root on the ground plane; the person stands along that reach.  A clip
    without a clear strike gets the person ahead of its last frame, along its
    direction of travel (or +Z).

    Args:
        joints: [T, J, 3] global joint positions (root joint first).
        standoff: Distance from the character's root, metres.
    """
    joints = np.asarray(joints, dtype=np.float64)
    root = joints[:, 0]
    offsets = joints[:, :, [0, 2]] - root[:, None, [0, 2]]
    reach = np.linalg.norm(offsets, axis=-1)
    frame, joint = np.unravel_index(int(np.argmax(reach)), reach.shape)
    if reach[frame, joint] > 0.45:
        direction = offsets[frame, joint] / reach[frame, joint]
        anchor = root[frame]
    else:
        travel = root[-1, [0, 2]] - root[0, [0, 2]]
        norm = float(np.linalg.norm(travel))
        direction = travel / norm if norm > 0.3 else np.array([0.0, 1.0])
        anchor = root[-1]
    x, z = anchor[[0, 2]] + standoff * direction
    return Placement(x=float(x), z=float(z), facing=(float(-direction[0]), float(-direction[1])))


def default_placement(x_offset: float, standoff: float = 1.2) -> Placement:
    """In front of a character that has not moved yet."""
    return Placement(x=x_offset, z=standoff, facing=(0.0, -1.0))


class HumanFigure:
    """One protected person in a client's scene."""

    def __init__(self, client: object, name: str, placement: Placement, visible: bool = True) -> None:
        vertices, faces = human_mesh()
        self._handle = client.scene.add_mesh_simple(
            f"/protected/{name}", vertices, faces,
            color=HUMAN_COLOR, material="standard", side="double",
            visible=visible,
        )
        self._label = client.scene.add_label(
            f"/protected/{name}/label", "person (protected)",
            position=(0.0, 1.95, 0.0), anchor="bottom-center", visible=visible,
        )
        self.place(placement)

    def place(self, placement: Placement) -> None:
        # The mesh faces +Z; turn it about +Y to face `placement.facing`.
        yaw = float(np.arctan2(placement.facing[0], placement.facing[1]))
        self._handle.wxyz = (float(np.cos(yaw / 2)), 0.0, float(np.sin(yaw / 2)), 0.0)
        self._handle.position = (placement.x, 0.0, placement.z)

    @property
    def visible(self) -> bool:
        return bool(self._handle.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self._handle.visible = bool(value)
        self._label.visible = bool(value)

    def remove(self) -> None:
        self._label.remove()
        self._handle.remove()
