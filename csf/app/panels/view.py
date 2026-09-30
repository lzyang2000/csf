# SPDX-License-Identifier: Apache-2.0
"""Body, camera and interface options.

The mesh and skeleton toggles are read back by the character factory through
`csf.app.scene.view_options`.
"""
from __future__ import annotations

import math


def _for_each_character(session: object):
    """The `Character` of every motion in the scene."""
    for motion in list(session.motions.values()):
        character = getattr(motion, "character", None)
        if character is not None:
            yield motion, character


def build(app: object, session: object) -> None:
    """Create the Body / Camera / Interface option folders for one client."""
    client = session.client
    gui = session.gui
    client_id = session.client_id

    def live() -> bool:
        return app.client_active(client_id)

    with client.gui.add_folder("Body options", expand_by_default=False):
        gui.gui_viz_skinned_mesh_checkbox = client.gui.add_checkbox(
            "Show Mesh", initial_value=True
        )
        gui.gui_viz_skinned_mesh_opacity_slider = client.gui.add_slider(
            "Mesh Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )
        gui.gui_viz_skeleton_checkbox = client.gui.add_checkbox(
            "Show Skeleton", initial_value=False
        )
        gui.gui_viz_foot_contacts_checkbox = client.gui.add_checkbox(
            "Show Foot Contacts", initial_value=False
        )
        gui.gui_viz_foot_contacts_checkbox.visible = (
            gui.gui_viz_skeleton_checkbox.value
        )

        @gui.gui_viz_skinned_mesh_checkbox.on_update
        def _on_mesh(_event) -> None:
            if not live():
                return
            for _motion, character in _for_each_character(session):
                character.set_skinned_mesh_visibility(
                    gui.gui_viz_skinned_mesh_checkbox.value
                )

        @gui.gui_viz_skinned_mesh_opacity_slider.on_update
        def _on_opacity(_event) -> None:
            if not live():
                return
            for _motion, character in _for_each_character(session):
                character.set_skinned_mesh_opacity(
                    gui.gui_viz_skinned_mesh_opacity_slider.value
                )

        @gui.gui_viz_skeleton_checkbox.on_update
        def _on_skeleton(_event) -> None:
            if not live():
                return
            # Foot contacts are drawn on the skeleton, so they follow it.
            gui.gui_viz_foot_contacts_checkbox.visible = (
                gui.gui_viz_skeleton_checkbox.value
            )
            if not gui.gui_viz_skeleton_checkbox.value:
                gui.gui_viz_foot_contacts_checkbox.value = False
            for _motion, character in _for_each_character(session):
                character.set_skeleton_visibility(gui.gui_viz_skeleton_checkbox.value)

        @gui.gui_viz_foot_contacts_checkbox.on_update
        def _on_contacts(_event) -> None:
            if not live():
                return
            for motion, character in _for_each_character(session):
                character.set_show_foot_contacts(
                    gui.gui_viz_foot_contacts_checkbox.value,
                    frame_idx=motion.cur_frame_idx,
                )

    with client.gui.add_folder("Camera options", expand_by_default=False):
        gui.gui_camera_fov_slider = client.gui.add_slider(
            "Camera FOV (deg)", min=30.0, max=90.0, step=1.0, initial_value=45.0
        )
        client.camera.fov = math.radians(gui.gui_camera_fov_slider.value)

        @gui.gui_camera_fov_slider.on_update
        def _on_fov(_event) -> None:
            if live():
                client.camera.fov = math.radians(gui.gui_camera_fov_slider.value)

    with client.gui.add_folder("Interface options", expand_by_default=False):
        gui.gui_show_timeline_checkbox = client.gui.add_checkbox(
            "Show Timeline", initial_value=True
        )
        gui.gui_show_starting_direction_checkbox = client.gui.add_checkbox(
            "Show Starting Direction", initial_value=True
        )
        gui.gui_dark_mode_checkbox = client.gui.add_checkbox(
            "Dark Mode", initial_value=False
        )
        app.set_start_direction_visible(
            session, gui.gui_show_starting_direction_checkbox.value
        )

        @gui.gui_show_timeline_checkbox.on_update
        def _on_timeline(_event) -> None:
            if live():
                client.timeline.set_visible(gui.gui_show_timeline_checkbox.value)

        @gui.gui_show_starting_direction_checkbox.on_update
        def _on_start_dir(_event) -> None:
            if live():
                app.set_start_direction_visible(
                    session, gui.gui_show_starting_direction_checkbox.value
                )

    @gui.gui_dark_mode_checkbox.on_update
    def _on_dark_mode(_event) -> None:
        if not live():
            return
        app.apply_theme(session, dark_mode=gui.gui_dark_mode_checkbox.value)
        for _motion, character in _for_each_character(session):
            character.change_theme(gui.gui_dark_mode_checkbox.value)

    # Re-theme now that the checkbox exists: viser then renders the toggle in the
    # titlebar, and the sidebar copy is hidden (its value is still read).
    app.apply_theme(session, dark_mode=gui.gui_dark_mode_checkbox.value)
    gui.gui_dark_mode_checkbox.visible = False
