# SPDX-License-Identifier: Apache-2.0
"""Examples: ready-made scenes, one click each."""
from __future__ import annotations

from csf.app import timeline, ui
from csf.app.examples import load_examples


def _describe(example) -> str:
    scene = ", ".join(example.scene_entities) or "nothing"
    prompts = " → ".join(f"“{text}”" for text, _ in example.prompts)
    lines = [f"{prompts}", f"Scene: {scene} · seed {example.seed}"]
    if example.backbone:
        lines.append(f"Backbone: {example.backbone}")
    if example.note:
        lines.append(example.note)
    return "\n\n".join(lines)


def run_example(app: object, session: object, example) -> None:
    """Set the backbone, timeline, seed and scene, then generate."""
    from csf.app.panels import backbone, controls  # noqa: PLC0415
    from csf.backbones import registry  # noqa: PLC0415

    target = example.backbone
    if target and target in registry.available_names():
        if target != session.model_name or session.bundle is None:
            backbone.load_backbone(app, session, target)
    elif target:
        ui.notify(app, session, f"{target} is not available here",
                  f"Running the example on {session.model_name} instead.")
    controls.set_scene(session, example.person, example.entities)
    session.gui.gui_seed.value = example.seed
    timeline.set_schedule(app, session, example.prompts)
    controls.on_generate(app, session)
    if example.runtime:
        ui.notify(app, session, "Runtime shield",
                  "Nobody is in view, so CSF lets the punch through. Tick 'Person in scene' "
                  "while it plays: the shield redirects the rest of the motion.", seconds=12.0)


def sync_scene(session: object) -> None:
    """Set the Scene panel to the selected example's scene."""
    from csf.app.panels import controls  # noqa: PLC0415

    gui = session.gui
    by_name = getattr(gui, "example_by_name", None)
    dropdown = getattr(gui, "gui_example_dropdown", None)
    if by_name and dropdown is not None and hasattr(gui, "gui_person_checkbox"):
        example = by_name[dropdown.value]
        controls.set_scene(session, example.person, example.entities)


def build(app: object, session: object) -> None:
    client, gui, client_id = session.client, session.gui, session.client_id
    examples = load_examples()
    if not examples:
        return
    by_name = gui.example_by_name = {e.name: e for e in examples}
    with client.gui.add_folder("Examples", expand_by_default=True):
        gui.gui_example_dropdown = client.gui.add_dropdown(
            "Example", options=list(by_name), initial_value=examples[0].name)
        gui.gui_example_text = client.gui.add_markdown(_describe(examples[0]))
        gui.gui_example_button = client.gui.add_button("Run example", color="blue")

        @gui.gui_example_dropdown.on_update
        def _on_select(_event) -> None:
            if not app.client_active(client_id):
                return
            gui.gui_example_text.content = _describe(by_name[gui.gui_example_dropdown.value])
            sync_scene(session)   # the Scene panel follows the selection right away

        @gui.gui_example_button.on_click
        def _on_run(_event) -> None:
            if app.client_active(client_id):
                run_example(app, session, by_name[gui.gui_example_dropdown.value])
