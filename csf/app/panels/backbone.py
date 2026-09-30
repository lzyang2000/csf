# SPDX-License-Identifier: Apache-2.0
"""Backbone selector: one dropdown over the registry, one load path.

A load holds the GPU lock throughout: it evicts the resident backbone, moves
the shared text encoder to where the new one needs it, builds the bundle,
then resets the scene and re-lays the timeline at the new frame rate.
"""
from __future__ import annotations

import traceback

from csf.app import generate, timeline, ui

REST_CHARACTER = "rest"

_NO_BACKBONES_MD = (
    "**No backbone is loadable on this machine.**\n\n"
    "Kimodo and ARDY download on first use; fetch ECHO / MotionHiFlow with "
    "`scripts/fetch_*.sh`, then restart."
)


def dropdown_options() -> list[str]:
    from csf.backbones import registry  # noqa: PLC0415

    return registry.available_names()


def status_markdown(session: object, selected: str | None = None) -> str:
    if session.bundle is None:
        return f"**Not loaded**: `{selected or session.model_name}`. Press *Load*."
    spec = session.spec
    gamma = session.filter_defaults.get("gamma", 0.0)
    return f"**{session.model_name}** · {session.model_fps:g} fps · γ {gamma:g}\n\n{spec.description}"


def push_filter_defaults(session: object, spec: object) -> None:
    """Publish the backbone's Table I settings to the session and the γ slider."""
    session.filter_defaults = spec.filter.as_overrides() if spec is not None else {}
    slider = getattr(session.gui, "gui_gamma_slider", None)
    if slider is not None and "gamma" in session.filter_defaults:
        slider.value = session.filter_defaults["gamma"]


def _refresh_status(session: object) -> None:
    handle = getattr(session.gui, "gui_backbone_status", None)
    if handle is not None:
        dropdown = getattr(session.gui, "gui_backbone_dropdown", None)
        handle.content = status_markdown(session, getattr(dropdown, "value", None))


def load_backbone(app: object, session: object, name: str) -> None:
    """Load `name` for this client.  Never raises (errors become a toast)."""
    from csf.backbones import registry  # noqa: PLC0415

    client = session.client
    if name == session.model_name and session.bundle is not None:
        return
    try:
        spec = registry.get_spec(name)
    except KeyError as exc:
        ui.toast_error(client, "Unknown backbone", str(exc))
        return
    note = ui.busy(client, f"Loading {name}...", "The first load downloads weights.")
    try:
        with generate.hold_gpu(app, notification=note):
            bundle = app.load_model(session, name)
            session.playing = False
            app.clear_motions(session)
            previous_spec, previous_fps = session.spec, session.model_fps
            session.model_name, session.spec, session.bundle = name, spec, bundle
            session.model_fps, session.skeleton = bundle.model_fps, bundle.skeleton
            session.frame_idx, session.last, session.handle = 0, None, None
            push_filter_defaults(session, spec)
            dropdown = getattr(session.gui, "gui_backbone_dropdown", None)
            if dropdown is not None and dropdown.value != name:
                dropdown.value = name
            timeline.reseed_prompts(app, session, previous_spec=previous_spec,
                                    previous_fps=previous_fps)
            app.add_character_motion(session, name=REST_CHARACTER)
            from csf.app.panels import controls  # noqa: PLC0415

            app.show_humans(session, controls.person_in_scene(session))
    except Exception as exc:  # noqa: BLE001 - reported, never raised at viser
        traceback.print_exc()
        ui.fail(note)
        ui.toast_error(client, f"{name} failed to load", str(exc))
        _refresh_status(session)
        return
    ui.finish(note, f"{name} ready")
    _refresh_status(session)


def build(app: object, session: object) -> None:
    client, gui, client_id = session.client, session.gui, session.client_id
    push_filter_defaults(session, session.spec)
    options = dropdown_options()
    with client.gui.add_folder("Backbone", expand_by_default=True):
        if not options:
            gui.gui_backbone_status = client.gui.add_markdown(_NO_BACKBONES_MD)
            return
        initial = session.model_name if session.model_name in options else options[0]
        gui.gui_backbone_dropdown = client.gui.add_dropdown(
            "Model", options=options, initial_value=initial,
            hint="The pretrained generator. One is resident at a time.")
        gui.gui_backbone_load_button = client.gui.add_button("Load")
        gui.gui_backbone_status = client.gui.add_markdown("")
        _refresh_status(session)

        @gui.gui_backbone_load_button.on_click
        def _on_load(_event) -> None:
            if app.client_active(client_id):
                load_backbone(app, session, gui.gui_backbone_dropdown.value)

        @gui.gui_backbone_dropdown.on_update
        def _on_select(_event) -> None:
            if app.client_active(client_id):
                _refresh_status(session)
