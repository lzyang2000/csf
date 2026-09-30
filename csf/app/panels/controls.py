# SPDX-License-Identifier: Apache-2.0
"""Scene, Generate and Filter settings: what the Generate button reads."""
from __future__ import annotations

import threading
import traceback

from csf.app import generate, timeline, ui

_LEGEND = "🟠 unfiltered generator\n🔵 CSF\n🧍 protected person"

_HINTS = {
    "person": (
        "A person is in the scene. Perception reports 'a person', which activates the "
        "person-protecting rules. After a generation without a person, ticking this during "
        "playback is a person entering view: the runtime shield redirects the rest of the "
        "CSF motion."
    ),
    "entities": (
        "Other labels perception reports, comma-separated (e.g. 'a ball', 'a wall', "
        "'a child', 'a dog'). The gate matches them against the rules' protected entities "
        "with the generator's text encoder, so any wording works."
    ),
    "filter": "Generate a CSF pass beside the unfiltered one.",
    "gamma": (
        "Safe-reference tracking strength γ (Eq. 3). 0 is the minimum-norm correction that "
        "just satisfies the active rules; larger values pull toward the safe reference. "
        "Loading a backbone restores its Table I value."
    ),
    "rho": "Barrier contraction rate ρ (Eq. 3). 1 imposes the safe set at every step.",
    "seed": "Both passes use this seed, so the filter is the only difference between them.",
    "steps": "Denoising steps (backbones trained for a fixed count use their own).",
}


def scene_entities(session: object) -> list[str]:
    """What perception reports: the person (if present) and the typed labels."""
    gui = session.gui
    labels = []
    if getattr(getattr(gui, "gui_person_checkbox", None), "value", False):
        labels.append(generate.PERSON)
    text = getattr(getattr(gui, "gui_entities_text", None), "value", "") or ""
    labels += [t.strip() for t in text.split(",") if t.strip()]
    return labels


def person_in_scene(session: object) -> bool:
    return bool(getattr(getattr(session.gui, "gui_person_checkbox", None), "value", False))


def set_scene(session: object, person: bool, entities) -> None:
    """Set the Scene widgets and show or hide the person, without triggering the shield."""
    gui = session.gui
    session.gui.suppress_person_event = True
    try:
        gui.gui_person_checkbox.value = bool(person)
        gui.gui_entities_text.value = ", ".join(entities)
    finally:
        session.gui.suppress_person_event = False
    if session.app is not None:
        session.app.show_humans(session, bool(person))


def collect_options(session: object) -> dict:
    gui = session.gui
    dropdown = getattr(gui, "gui_backbone_dropdown", None)
    return {
        "model_name": getattr(dropdown, "value", None) or session.model_name,
        "steps": int(gui.gui_steps_slider.value),
        "cfg_weight": [float(gui.gui_cfg_text_weight_slider.value), 2.0],
        "cfg_type": "separated" if gui.gui_cfg_checkbox.value else "nocfg",
        "num_transition_frames": int(gui.gui_transition_slider.value),
        "gamma": float(gui.gui_gamma_slider.value),
        "rho": float(gui.gui_rho_slider.value),
    }


def on_generate(app: object, session: object) -> None:
    """Load if needed, read the timeline and the scene, generate both passes."""
    client, gui = session.client, session.gui
    note = ui.busy(client, "Generating...", "Unfiltered pass, then CSF.")
    gui.gui_generate_button.disabled = True
    try:
        dropdown = getattr(gui, "gui_backbone_dropdown", None)
        generate.ensure_backbone(app, session, getattr(dropdown, "value", None))
        prompts = timeline.timeline_prompts(client)
        timeline.update_duration_auto(app, session)
        filter_on = bool(gui.gui_filter_checkbox.value)
        generate.run_generation(
            app, session,
            prompts=[p.text for p in prompts],
            durations=timeline.compute_prompt_num_frames(prompts),
            seed=int(gui.gui_seed.value),
            entities=scene_entities(session),
            filter_on=filter_on,
            opts=collect_options(session),
        )
        ui.finish(note, "Done", "Orange: unfiltered. Blue: CSF." if filter_on
                  else "CSF is off: this is the raw generator output.")
        session.playing = True
    except Exception as exc:  # noqa: BLE001 - reported, never raised at viser
        traceback.print_exc()
        ui.fail(note)
        ui.toast_error(client, "Generation failed", str(exc))
        app.check_cuda_health()
    finally:
        gui.gui_generate_button.disabled = False


def _refresh_legend(session: object) -> None:
    gui = session.gui
    try:
        old = getattr(gui, "gui_legend", None)
        if old is not None:
            old.remove()
        gui.gui_legend = session.client.add_notification(
            title="Legend", body=_LEGEND, auto_close=False, with_close_button=False)
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def build_scene(app: object, session: object) -> None:
    client, gui, client_id = session.client, session.gui, session.client_id
    gui.suppress_person_event = False
    with client.gui.add_folder("Scene", expand_by_default=True):
        gui.gui_person_checkbox = client.gui.add_checkbox(
            "Person in scene", initial_value=False, hint=_HINTS["person"])
        gui.gui_entities_text = client.gui.add_text(
            "Other entities", initial_value="a ball", hint=_HINTS["entities"])

        @gui.gui_person_checkbox.on_update
        def _on_person(_event) -> None:
            if not app.client_active(client_id) or gui.suppress_person_event:
                return
            if gui.gui_person_checkbox.value:
                # A person entering view mid-motion: the runtime shield (off-thread,
                # so the viser callback returns immediately).
                threading.Thread(target=generate.person_enters_view, args=(app, session),
                                 daemon=True).start()
            else:
                app.show_humans(session, False)
    _refresh_legend(session)


def build_generate(app: object, session: object) -> None:
    client, gui, client_id = session.client, session.gui, session.client_id
    with client.gui.add_folder("Generate", expand_by_default=True):
        gui.gui_total_duration = client.gui.add_markdown(
            f"Total duration: {session.cur_duration:.1f} s")
        gui.gui_seed = client.gui.add_number("Seed", initial_value=0, hint=_HINTS["seed"])
        gui.gui_filter_checkbox = client.gui.add_checkbox(
            "CSF on", initial_value=True, hint=_HINTS["filter"])
        gui.gui_generate_button = client.gui.add_button("Generate", color="green")

        @gui.gui_generate_button.on_click
        def _on_click(_event) -> None:
            if app.client_active(client_id):
                on_generate(app, session)


def build_filter(app: object, session: object) -> None:
    client, gui = session.client, session.gui
    gamma = float(session.filter_defaults.get("gamma", 0.0))
    rho = float(app.policy.cfg.rho)
    with client.gui.add_folder("Filter settings", expand_by_default=False):
        gui.gui_gamma_slider = client.gui.add_slider(
            "Tracking strength γ", min=0.0, max=1.0, step=0.05, initial_value=gamma,
            hint=_HINTS["gamma"])
        gui.gui_rho_slider = client.gui.add_slider(
            "Contraction rate ρ", min=0.05, max=1.0, step=0.05, initial_value=rho,
            hint=_HINTS["rho"])
        with client.gui.add_folder("Sampling", expand_by_default=False):
            gui.gui_steps_slider = client.gui.add_slider(
                "Denoising steps", min=10, max=500, step=10, initial_value=100,
                hint=_HINTS["steps"])
            gui.gui_cfg_checkbox = client.gui.add_checkbox("Classifier-free guidance",
                                                           initial_value=True)
            gui.gui_cfg_text_weight_slider = client.gui.add_slider(
                "Text guidance weight", min=0.0, max=5.0, step=0.1, initial_value=2.0)
            gui.gui_transition_slider = client.gui.add_slider(
                "Transition frames", min=1, max=10, step=1, initial_value=5,
                hint="Frames blended between consecutive timeline prompts.")
