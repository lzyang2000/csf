# SPDX-License-Identifier: Apache-2.0
"""Choosing the safe reference x_safe (Sec. III-B).

The safe reference is the benign version of the requested command, produced
by the frozen generator from a text chosen out of a small library of neutral
actions, with a neutral stand as the fallback.  The text encoder picks the
nearest library action; the unfiltered clip's own root motion decides whether
the benign version should travel at all:

* the clip travels (net root speed >= the stand threshold): the nearest
  locomotion action, if it is close enough;
* the clip stays in place: the stand.

Everything takes an ``encode(text) -> 1-D array`` callable.
"""
from __future__ import annotations

import numpy as np

from .config import SafeReferencePolicy

_EPS = 1e-8


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return v / (np.linalg.norm(v) + _EPS)


def nearest(encode, prompt: str, candidates) -> tuple[str, float]:
    """The candidate text nearest `prompt` by cosine, and its score."""
    q = _unit(encode(prompt))
    scores = [float(_unit(encode(t)) @ q) for t in candidates]
    i = int(np.argmax(scores))
    return candidates[i], scores[i]


def select_from_library(encode, prompt: str, policy: SafeReferencePolicy) -> str:
    """The nearest library action, or the stand when none is close enough."""
    if not policy.library or not (prompt and prompt.strip()):
        return policy.stand
    text, score = nearest(encode, prompt, list(policy.library))
    return text if score >= policy.min_similarity else policy.stand


def net_root_speed(root_pos, fps: float) -> float:
    """Net root displacement rate (m/s) of a [T, 3] trajectory.

    Net displacement rather than mean frame speed: an in-place motion with a
    bouncing root accumulates frame speed without travelling anywhere.
    """
    p = np.asarray(
        root_pos.detach().cpu().numpy() if hasattr(root_pos, "detach") else root_pos,
        dtype=float,
    ).reshape(-1, 3)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]) / ((p.shape[0] - 1) / float(fps)))


def select_for_motion(encode, prompt: str, speed: float, speed_threshold: float,
                      policy: SafeReferencePolicy) -> tuple[str, str]:
    """The safe reference for a segment whose unfiltered clip moves at `speed`.

    Returns:
        ``(text, reason)``.
    """
    if speed < speed_threshold or not policy.locomotion:
        return policy.stand, f"in place ({speed:.2f} m/s)"
    text, score = nearest(encode, prompt, list(policy.locomotion))
    if score < policy.locomotion_min_similarity:
        return policy.stand, f"moving ({speed:.2f} m/s), no close locomotion ({score:.2f})"
    return text, f"moving ({speed:.2f} m/s), locomotion cos {score:.2f}"


def segment_windows(durations, total_frames: int) -> list[tuple[int, int]]:
    """Split a stitched multi-prompt clip back into one frame range per segment.

    Proportional to the requested durations: exact when the generator returns
    sum(durations) frames and off by at most a transition blend otherwise.
    """
    counts = [int(d) for d in durations]
    span, total = sum(counts), int(total_frames)
    if not counts:
        return []
    if span <= 0 or total <= 0:
        return [(0, max(total, 0))] * len(counts)
    edges, cum = [0], 0
    for d in counts:
        cum += d
        edges.append(int(round(total * cum / span)))
    return list(zip(edges[:-1], edges[1:]))
