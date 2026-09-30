# SPDX-License-Identifier: Apache-2.0
"""The 3D scene: camera, floor, characters and the protected person."""
from __future__ import annotations

import threading
from dataclasses import dataclass

_CAMERA_POSITION = (-1.6, 2.2, -3.6)
_CAMERA_LOOK_AT = (0.0, 0.9, 1.6)
_CAMERA_UP = (0.0, 1.0, 0.0)
_CAMERA_FOV_DEG = 45.0


@dataclass
class ViewOptions:
    """The view panel's toggles (defaults match its initial values)."""

    mesh_visible: bool = True
    mesh_opacity: float = 1.0
    skeleton_visible: bool = False
    foot_contacts: bool = False
    dark_mode: bool = False


def view_options(session: object) -> ViewOptions:
    gui = session.gui

    def value(name: str, default):
        return getattr(getattr(gui, name, None), "value", default)

    return ViewOptions(
        mesh_visible=bool(value("gui_viz_skinned_mesh_checkbox", True)),
        mesh_opacity=float(value("gui_viz_skinned_mesh_opacity_slider", 1.0)),
        skeleton_visible=bool(value("gui_viz_skeleton_checkbox", False)),
        foot_contacts=bool(value("gui_viz_foot_contacts_checkbox", False)),
        dark_mode=bool(value("gui_dark_mode_checkbox", False)),
    )


def smplx_available() -> bool:
    """True when the SMPL-X body model (license-gated, fetched by hand) is installed."""
    from pathlib import Path  # noqa: PLC0415

    import kimodo  # noqa: PLC0415

    path = Path(kimodo.__file__).parent / "assets" / "skeletons" / "smplx22" / "SMPLX_NEUTRAL.npz"
    return path.exists()


def rest_pose_rotations(skeleton: object, rest_pos: object) -> object:
    """Global rest rotations [J, 3, 3] for a skeleton-only character."""
    import torch  # noqa: PLC0415

    local_rot = getattr(skeleton, "rest_pose_local_rot", None)
    if local_rot is not None:
        root = torch.zeros(3, device=local_rot.device, dtype=local_rot.dtype)
        global_rot, _local, _pos = skeleton.fk(local_rot.clone(), root)
        return global_rot
    eye = torch.eye(3, dtype=rest_pos.dtype, device=rest_pos.device)
    return eye.expand(rest_pos.shape[-2], 3, 3).clone()


def setup_scene(app: object, session: object) -> None:
    """Camera, floor grid and origin marker."""
    import numpy as np  # noqa: PLC0415
    import viser  # noqa: PLC0415

    from kimodo.demo.config import LIGHT_THEME  # noqa: PLC0415
    from kimodo.viz import viser_utils  # noqa: PLC0415

    client = session.client
    client.camera.position = np.array(_CAMERA_POSITION, dtype=np.float64)
    client.camera.look_at = np.array(_CAMERA_LOOK_AT, dtype=np.float64)
    client.camera.up_direction = np.array(_CAMERA_UP, dtype=np.float64)
    client.camera.fov = np.deg2rad(_CAMERA_FOV_DEG)

    session.grid = client.scene.add_grid(
        "/grid", width=app.floor_len, height=app.floor_len,
        wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
        position=(0.0, 0.0001, 0.0), fade_distance=3 * app.floor_len,
        section_color=LIGHT_THEME["grid"], infinite_grid=True,
    )
    session.origin_marker = viser_utils.WaypointMesh(
        "/origin_waypoint", client, position=np.array([0.0, 0.0, 0.0]),
        heading=np.array([0.0, 1.0]), color=(0, 0, 255),
    )
    app.apply_theme(session)


def add_character_motion(app: object, session: object, *, name: str,
                         joints_pos: object | None = None, joints_rot: object | None = None,
                         foot_contacts: object | None = None,
                         color: tuple[int, int, int] | None = None, x_offset: float = 0.0):
    """Add one character and register its motion; None when nothing is loaded."""
    from kimodo.viz.playback import CharacterMotion  # noqa: PLC0415
    from kimodo.viz.scene import Character  # noqa: PLC0415

    if not app.client_active(session.client_id) or session.skeleton is None:
        return None
    view = view_options(session)
    if joints_pos is not None and x_offset != 0.0:
        joints_pos = joints_pos.clone()
        joints_pos[..., 0] += x_offset

    mesh_mode = getattr(session.spec, "mesh_mode", "g1_stl")
    # Without the SMPL-X model, an SMPL body is drawn as its skeleton.
    skeleton_only = mesh_mode == "smplx_skin" and not smplx_available()
    character = Character(
        name, session.client, session.skeleton,
        create_skeleton_mesh=True, create_skinned_mesh=not skeleton_only,
        visible_skeleton=skeleton_only, visible_skinned_mesh=False,
        skinned_mesh_opacity=view.mesh_opacity, show_foot_contacts=view.foot_contacts,
        dark_mode=view.dark_mode, mesh_mode=None if skeleton_only else mesh_mode,
        gui_use_soma_layer_checkbox=None,
    )
    if color is not None:
        if character.skinned_mesh is not None:
            character.skinned_mesh.color = color
        if character.g1_mesh_rig is not None:
            character.g1_mesh_rig.set_color(color)

    rest_pos, rest_rot = character.get_pose()
    if rest_rot is None:
        rest_rot = rest_pose_rotations(session.skeleton, rest_pos)
    if joints_pos is None:
        joints_pos = rest_pos[None].repeat(session.max_frame_idx + 1, 1, 1)
    if joints_rot is None:
        joints_rot = rest_rot[None].repeat(session.max_frame_idx + 1, 1, 1, 1)

    motion = CharacterMotion(character, joints_pos, joints_rot, foot_contacts)
    session.motions[name] = motion
    motion.set_frame(session.frame_idx)

    def _reveal() -> None:
        # Deferred so the mesh is posed before it is shown; the client or the
        # character may be gone by then.
        if not app.client_active(session.client_id) or session.motions.get(name) is not motion:
            return
        if skeleton_only:
            character.set_skeleton_visibility(True)
            return
        now = view_options(session)
        character.set_skinned_mesh_visibility(now.mesh_visible)
        character.set_skeleton_visibility(now.skeleton_visible)

    threading.Timer(0.2, _reveal).start()
    return motion
