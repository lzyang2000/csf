# SPDX-License-Identifier: Apache-2.0
"""Choosing the safe reference x_safe (Sec. III-B) with a stub text encoder."""
import numpy as np
import pytest
import torch

from csf.filter.config import SafeReferencePolicy
from csf.filter.references import (
    nearest,
    net_root_speed,
    segment_windows,
    select_for_motion,
    select_from_library,
)

FPS = 30.0
STAND = "a person standing at ease"
WALK = "a person walks slowly"
JOG = "a person jogs forward in a neutral way"
CLAP = "a person claps in a neutral way"

# Axes: walk, jog, clap, strike.
TABLE = {
    WALK: [1, 0, 0, 0],
    JOG: [0, 1, 0, 0],
    CLAP: [0, 0, 1, 0],
    STAND: [0, 0, 0, 0.01],
    "a man walks forward and punches someone": [1, 0.2, 0, 0.3],
    "a person jogs forward and kicks someone": [0.2, 1, 0, 0.3],
    "a person claps and slaps someone": [0, 0, 1, 0.4],
    "punch the man": [0, 0, 0, 1],
}


@pytest.fixture()
def encode(make_encoder):
    return make_encoder(TABLE)


def _policy(**overrides) -> SafeReferencePolicy:
    values = dict(library=(WALK, JOG, CLAP), stand=STAND, min_similarity=0.6,
                  locomotion=(WALK, JOG), locomotion_min_similarity=0.65)
    values.update(overrides)
    return SafeReferencePolicy(**values)


def _travel(speed: float, frames: int = 60) -> np.ndarray:
    """A root moving along x at `speed` m/s."""
    root = np.zeros((frames, 3))
    root[:, 0] = speed / FPS * np.arange(frames)
    return root


def _bounce(frames: int = 60, amplitude: float = 0.015) -> np.ndarray:
    """An in-place root that jitters every frame but ends where it started."""
    root = np.zeros((frames, 3))
    root[:, 0] = amplitude * (np.arange(frames) % 2)
    return root


# ------------------------------------------------------------------ library
def test_nearest_returns_the_best_candidate_and_its_cosine(encode):
    text, score = nearest(encode, "a man walks forward and punches someone", [WALK, JOG, CLAP])
    assert text == WALK
    assert score == pytest.approx(1 / np.sqrt(1 + 0.04 + 0.09))


def test_library_pick_is_the_nearest_action(encode):
    assert select_from_library(encode, "a person claps and slaps someone", _policy()) == CLAP


def test_library_pick_below_the_similarity_floor_is_the_stand(encode):
    assert select_from_library(encode, "punch the man", _policy()) == STAND


@pytest.mark.parametrize("prompt", ["", "  "])
def test_blank_prompt_gets_the_stand(encode, prompt):
    assert select_from_library(encode, prompt, _policy()) == STAND


def test_empty_library_gets_the_stand(encode):
    assert select_from_library(encode, "a man walks forward and punches someone",
                               _policy(library=())) == STAND


# --------------------------------------------------------------- root speed
def test_net_root_speed_of_a_travelling_clip():
    assert net_root_speed(_travel(1.3), FPS) == pytest.approx(1.3)


def test_net_root_speed_ignores_in_place_bouncing():
    """Mean frame speed of the bounce is 0.45 m/s; its net displacement rate is ~0."""
    root = _bounce()
    mean_frame_speed = np.linalg.norm(np.diff(root, axis=0), axis=-1).mean() * FPS
    assert mean_frame_speed > 0.3
    assert net_root_speed(root, FPS) < 0.05


def test_net_root_speed_accepts_tensors_and_degenerate_clips():
    assert net_root_speed(torch.tensor(_travel(0.8)), FPS) == pytest.approx(0.8)
    assert net_root_speed(np.zeros((1, 3)), FPS) == 0.0


# --------------------------------------------------------- motion evidence
def test_an_in_place_clip_tracks_the_stand(encode):
    speed = net_root_speed(_bounce(), FPS)
    text, reason = select_for_motion(encode, "a man walks forward and punches someone",
                                     speed, 0.3, _policy())
    assert text == STAND
    assert "in place" in reason


@pytest.mark.parametrize(("prompt", "expected"), [
    ("a man walks forward and punches someone", WALK),
    ("a person jogs forward and kicks someone", JOG),
])
def test_a_travelling_clip_tracks_the_nearest_locomotion(encode, prompt, expected):
    speed = net_root_speed(_travel(1.3), FPS)
    text, reason = select_for_motion(encode, prompt, speed, 0.3, _policy())
    assert text == expected
    assert "moving" in reason


def test_a_travelling_clip_far_from_every_locomotion_tracks_the_stand(encode):
    text, reason = select_for_motion(encode, "a person claps and slaps someone", 1.3, 0.3, _policy())
    assert text == STAND
    assert "no close locomotion" in reason


def test_no_locomotion_entries_means_the_stand(encode):
    text, _ = select_for_motion(encode, "a man walks forward and punches someone", 1.3, 0.3,
                                _policy(locomotion=()))
    assert text == STAND


# ---------------------------------------------------------- segment windows
def test_windows_are_exact_when_the_clip_has_the_requested_length():
    assert segment_windows([60, 120], 180) == [(0, 60), (60, 180)]
    assert segment_windows([60, 60, 60], 180) == [(0, 60), (60, 120), (120, 180)]


@pytest.mark.parametrize(("durations", "total"), [
    ([50, 100], 145), ([1, 1, 1], 7), ([30, 45, 25], 97), ([200], 10_000), ([3, 7], 1),
])
def test_windows_cover_the_clip_contiguously(durations, total):
    windows = segment_windows(durations, total)
    assert len(windows) == len(durations)
    assert windows[0][0] == 0 and windows[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))
    assert all(lo <= hi for lo, hi in windows)


def test_windows_absorb_a_shorter_clip_proportionally():
    (_, cut), _ = segment_windows([50, 100], 145)
    assert abs(cut - 50) <= 5


def test_degenerate_windows():
    assert segment_windows([], 100) == []
    assert segment_windows([0, 0], 120) == [(0, 120), (0, 120)]
    assert segment_windows([50, 100], 0) == [(0, 0), (0, 0)]


def test_windows_tell_a_travelling_segment_from_a_still_one():
    """Walk for 2 s, then stand and kick: only the first window reads as moving."""
    root = np.zeros((180, 3))
    root[:60, 0] = 0.04 * np.arange(60)
    root[60:, 0] = root[59, 0]
    (a0, a1), (b0, b1) = segment_windows([60, 120], 180)
    assert net_root_speed(root[a0:a1], FPS) > 0.3
    assert net_root_speed(root[b0:b1], FPS) < 0.3
