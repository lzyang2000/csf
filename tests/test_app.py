# SPDX-License-Identifier: Apache-2.0
"""GPU-free checks of the demo's data and pure helpers."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from csf.app import entities, timeline
from csf.app.examples import load_examples
from csf.backbones import registry


def test_app_main_is_import_light():
    """Importing the entry point must not import torch, viser or kimodo."""
    code = ("import sys, csf.app.main; "
            "print(any(m in sys.modules for m in ('torch', 'viser', 'kimodo')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_examples_parse_and_name_registered_backbones():
    examples = load_examples()
    assert examples
    assert len({e.name for e in examples}) == len(examples)
    for example in examples:
        assert example.backbone in registry.all_names()
        assert example.prompts and all(seconds > 0 for _, seconds in example.prompts)
        if example.runtime:
            assert not example.person, "a runtime-shield scene starts with nobody in view"


def test_scene_entities_include_the_person_first():
    example = next(e for e in load_examples() if e.person and e.entities)
    assert example.scene_entities[0] == "a person"
    assert example.scene_entities[1:] == list(example.entities)


def test_every_family_has_a_default_schedule():
    for spec in registry.SPECS.values():
        assert timeline.prompt_schedule_for(spec)


def test_prompt_frame_convention():
    prompts = [SimpleNamespace(start_frame=0, end_frame=59), SimpleNamespace(start_frame=59, end_frame=119)]
    assert timeline.compute_prompt_num_frames(prompts) == [59, 61]


def test_rescale_spans_keeps_seconds():
    assert timeline.rescale_spans([("a", 120)], 30.0, 25.0) == [("a", 100)]


def test_human_mesh_stands_on_the_ground_with_arms_down():
    vertices, faces = entities.human_mesh()
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    assert vertices[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert 1.6 < vertices[:, 1].max() < 1.9
    assert np.abs(vertices[:, 0]).max() < 0.4        # arms by the sides, not an A-pose


def _walk_with_strike(strike_frame=None):
    T, J = 60, 5
    joints = np.zeros((T, J, 3))
    joints[:, :, 2] = np.linspace(0.0, 2.0, T)[:, None]
    joints[:, :, 1] = 1.0
    if strike_frame is not None:
        joints[strike_frame, 2] = joints[strike_frame, 0] + np.array([0.0, 0.0, 0.8])
    return joints


def test_person_stands_where_the_strike_lands():
    placement = entities.placement_from_motion(_walk_with_strike(30))
    assert placement.z == pytest.approx(2.0 * 30 / 59 + entities.DEFAULT_STANDOFF)
    assert placement.facing == pytest.approx((0.0, -1.0))


def test_person_stands_ahead_of_a_clip_without_a_strike():
    placement = entities.placement_from_motion(_walk_with_strike(None))
    assert placement.z == pytest.approx(2.0 + entities.DEFAULT_STANDOFF)
