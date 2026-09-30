# SPDX-License-Identifier: Apache-2.0
"""The prompt timeline: the demo's text input and duration control.

Prompts are segments on the kimodo-viser `client.timeline` (each with `.text`,
`.start_frame`, `.end_frame`).  Several segments form a multi-prompt schedule;
the total duration follows the schedule.
"""
from __future__ import annotations

from typing import Sequence

#: Default schedule per backbone family, as (text, seconds) segments: the
#: paper's commands for that backbone (Fig. 4, Fig. 6).
FAMILY_PROMPT_SCHEDULES: dict[str, tuple[tuple[str, float], ...]] = {
    "kimodo": (("a person walks forward and high kicks a ball", 4.0),),
    "ardy": (("Walk slowly and punch at a pillar", 4.0),),
    "echo": (("a person uses left hand to punch a wall", 4.0),),
    "motionhiflow": (("a person uses left hand to punch a wall", 4.0),),
}
DEFAULT_SCHEDULE = FAMILY_PROMPT_SCHEDULES["kimodo"]

MIN_DURATION = 2.0
MAX_DURATION = 10.0
_ZOOM_SLACK = 1.10
_MAX_FRAMES_ZOOM = 1000


def timeline_prompts(client: object) -> list:
    """The client's prompt segments, sorted by start frame (possibly empty)."""
    try:
        return sorted(client.timeline._prompts.values(), key=lambda p: p.start_frame)
    except Exception:  # noqa: BLE001 - no timeline or no prompts yet
        return []


def compute_prompt_num_frames(prompts: Sequence) -> list[int]:
    """Frames per segment: [start, end) for all but the last, [start, end] for the last."""
    if not prompts:
        return []
    last = len(prompts) - 1
    return [p.end_frame - p.start_frame + (1 if i == last else 0) for i, p in enumerate(prompts)]


def prompt_schedule_for(spec: object | None) -> tuple[tuple[str, float], ...]:
    """The default (text, seconds) schedule for a backbone."""
    return FAMILY_PROMPT_SCHEDULES.get(getattr(spec, "family", None), DEFAULT_SCHEDULE)


def schedule_spans(segments: Sequence[tuple[str, float]], fps: float) -> list[tuple[str, int]]:
    """(text, seconds) -> (text, end_frame - start_frame)."""
    return [(text, int(seconds * fps - 1)) for text, seconds in segments]


def prompt_spans(client: object) -> list[tuple[str, int]]:
    return [(p.text, int(p.end_frame) - int(p.start_frame)) for p in timeline_prompts(client)]


def set_timeline_spans(timeline: object, fps: float, spans: Sequence[tuple[str, int]]) -> None:
    """Append one prompt segment per (text, frames) entry."""
    for text, frames in spans:
        timeline.set_defaults(
            default_text=text,
            default_duration=int(frames),
            min_duration=int(MIN_DURATION * fps - 1),
            max_duration=int(MAX_DURATION * fps - 1),
            default_num_frames_zoom=int(_ZOOM_SLACK * MAX_DURATION * fps),
            max_frames_zoom=_MAX_FRAMES_ZOOM,
            fps=fps,
        )


def rescale_spans(spans, from_fps: float, to_fps: float) -> list[tuple[str, int]]:
    """The same segments laid out at a new frame rate (same length in seconds)."""
    if not from_fps or from_fps <= 0 or to_fps == from_fps:
        return list(spans)
    scale = to_fps / from_fps
    return [(text, max(1, round(frames * scale))) for text, frames in spans]


def bring_up_timeline(client: object, fps: float, segments=DEFAULT_SCHEDULE) -> None:
    timeline = client.timeline
    set_timeline_spans(timeline, fps, schedule_spans(segments, fps))
    timeline.set_visible(True)
    timeline.set_current_frame(0)


def set_schedule(app: object, session: object, segments: Sequence[tuple[str, float]]) -> None:
    """Replace the timeline with `segments` at the session's frame rate."""
    timeline = session.client.timeline
    timeline.clear_prompts()
    set_timeline_spans(timeline, session.model_fps, schedule_spans(segments, session.model_fps))
    timeline.set_current_frame(0)
    update_duration_auto(app, session)


def reseed_prompts(app: object, session: object, *, previous_spec=None, previous_fps=None) -> None:
    """Re-lay the timeline after a backbone switch.

    The new backbone's default schedule replaces the timeline only when it
    still holds the previous backbone's untouched defaults; prompts the user
    typed are kept and rescaled to the new frame rate.
    """
    old_fps = previous_fps or session.model_fps
    current = prompt_spans(session.client)
    untouched = previous_spec is not None and current == schedule_spans(
        prompt_schedule_for(previous_spec), old_fps)
    if current and not untouched:
        spans = rescale_spans(current, old_fps, session.model_fps)
    else:
        spans = schedule_spans(prompt_schedule_for(session.spec), session.model_fps)
    timeline = session.client.timeline
    timeline.clear_prompts()
    set_timeline_spans(timeline, session.model_fps, spans)
    timeline.set_current_frame(0)
    update_duration_auto(app, session)


def set_new_duration(app: object, session: object, seconds: float) -> None:
    session.cur_duration = seconds
    session.max_frame_idx = max(0, int(seconds * session.model_fps) - 1)
    label = getattr(session.gui, "gui_total_duration", None)
    if label is not None:
        label.content = f"Total duration: {seconds:.1f} s"
    if session.frame_idx > session.max_frame_idx:
        app.set_frame(session, session.max_frame_idx)


def update_duration_auto(app: object, session: object) -> None:
    """The total duration follows the prompt schedule."""
    frames = compute_prompt_num_frames(timeline_prompts(session.client))
    if frames and session.model_fps > 0:
        set_new_duration(app, session, sum(frames) / session.model_fps)


def register_callbacks(app: object, session: object) -> None:
    """Playhead and prompt-edit callbacks for one client."""
    timeline = session.client.timeline
    client_id = session.client_id

    @timeline.on_frame_change
    def _on_frame_change(frame: int) -> None:
        app.set_frame(session, frame, update_timeline=False)

    def _reflow(*_args, **_kwargs) -> None:
        if app.client_active(client_id):
            update_duration_auto(app, session)

    timeline.on_prompt_add(_reflow)
    timeline.on_prompt_update(_reflow)
    timeline.on_prompt_resize(_reflow)
    timeline.on_prompt_move(_reflow)
    timeline.on_prompt_delete(_reflow)
