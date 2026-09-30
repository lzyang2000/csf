# SPDX-License-Identifier: Apache-2.0
"""Playback controls, and the frame arithmetic the app's playback loop uses."""
from __future__ import annotations

#: Button-group label -> playback speed multiplier.
PLAYBACK_SPEEDS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}

#: Label the speed group starts on.
DEFAULT_SPEED_LABEL = "1x"


def frame_update_interval(
    playback_fps: float, playback_speed: float, model_fps: float
) -> int:
    """How many loop ticks to wait between frames.

    The loop runs at `playback_fps` (twice the fastest model's rate, so 2x
    playback is reachable) and advances one frame every
    `playback_fps / (speed * model_fps)` ticks.

    Args:
        playback_fps: Loop tick rate.
        playback_speed: 0.5 / 1.0 / 2.0.
        model_fps: Frame rate of the loaded backbone.

    Returns:
        Tick interval, at least 1.
    """
    frames_per_second = playback_speed * model_fps
    if frames_per_second <= 0:
        return 1
    return max(1, int(playback_fps / frames_per_second))


def next_frame_index(frame_idx: int, max_frame_idx: int) -> int:
    """The frame after `frame_idx`, wrapping to 0 past `max_frame_idx`."""
    return 0 if frame_idx >= max_frame_idx else frame_idx + 1


def step_frame(app: object, session: object, delta: int) -> None:
    """Pause and move the playhead by `delta`, clamped to the clip."""
    session.playing = False
    session.frame_idx = min(max(0, session.frame_idx + delta), session.max_frame_idx)
    app.set_frame(session, session.frame_idx)


def build(app: object, session: object) -> None:
    """Create the Playback folder and bind the keyboard shortcuts."""
    client = session.client
    gui = session.gui
    client_id = session.client_id

    def live() -> object | None:
        """The session, or None once this client has disconnected."""
        return session if app.client_active(client_id) else None

    with client.gui.add_folder("Playback", expand_by_default=True):
        gui.gui_model_fps = client.gui.add_number(
            "Model FPS", initial_value=session.model_fps, disabled=True
        )
        gui.gui_playback_speed_buttons = client.gui.add_button_group(
            "Playback Speed", options=list(PLAYBACK_SPEEDS)
        )
        gui.gui_playback_speed_buttons.value = DEFAULT_SPEED_LABEL

        @gui.gui_playback_speed_buttons.on_click
        def _on_speed(_event) -> None:
            if live() is None:
                return
            session.playback_speed = PLAYBACK_SPEEDS[
                gui.gui_playback_speed_buttons.value
            ]

        # On-screen controls for touch devices (the shortcuts need a keyboard).
        gui.gui_play_pause_button = client.gui.add_button("▶  Play  /  ⏸  Pause")
        gui.gui_prev_frame_button = client.gui.add_button("⏮  Prev frame")
        gui.gui_next_frame_button = client.gui.add_button("⏭  Next frame")
        gui.gui_restart_button = client.gui.add_button("⏪  Restart")

        @gui.gui_play_pause_button.on_click
        def _on_play(_event) -> None:
            if live() is not None:
                session.playing = not session.playing

        @gui.gui_prev_frame_button.on_click
        def _on_prev(_event) -> None:
            if live() is not None:
                step_frame(app, session, -1)

        @gui.gui_next_frame_button.on_click
        def _on_next(_event) -> None:
            if live() is not None:
                step_frame(app, session, +1)

        @gui.gui_restart_button.on_click
        def _on_restart(_event) -> None:
            if live() is None:
                return
            session.frame_idx = 0
            app.set_frame(session, 0)

    # Space toggles playback, arrows step frames.  The browser repeats keydown
    # while a key is held; only the first press toggles.
    space_held = [False]

    @client.scene.on_keyboard_event("keydown", debounce_ms=100)
    def _on_key(event) -> None:
        if live() is None:
            return
        if event.event_type == "keyup":
            if event.key == " ":
                space_held[0] = False
            return
        if event.key == " ":
            if not space_held[0]:
                space_held[0] = True
                session.playing = not session.playing
        elif event.key == "ArrowLeft":
            step_frame(app, session, -1)
        elif event.key == "ArrowRight":
            step_frame(app, session, +1)
