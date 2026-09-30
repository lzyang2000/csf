# SPDX-License-Identifier: Apache-2.0
"""Kimodo's step-matched filtered sampling loop and its handle, on a tiny fake model."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from csf.backbones.kimodo import filtered_model
from csf.backbones.kimodo.filtered_generate import generate_filtered_segment, root_path_dims
from csf.filter.config import FilterConfig, Rule, SafeReferencePolicy

needs_kimodo = pytest.mark.skipif(importlib.util.find_spec("kimodo") is None,
                                  reason="the sampling loop uses kimodo helpers")

NBJOINTS, TEXT_DIM, FRAMES, STEPS = 2, 6, 24, 5
ROOT = list(range(5))


def _kimodo_layout(nbjoints: int = NBJOINTS) -> dict:
    return {
        "smooth_root_pos": torch.Size([3]),
        "global_root_heading": torch.Size([2]),
        "local_joints_positions": torch.Size([nbjoints, 3]),
        "global_rot_data": torch.Size([nbjoints, 6]),
        "velocities": torch.Size([nbjoints, 3]),
        "foot_contacts": torch.Size([4]),
    }


# ----------------------------------------------------------- root_path_dims
def test_root_path_is_the_smooth_root_position_and_heading():
    assert root_path_dims(SimpleNamespace(size_dict=_kimodo_layout())) == ROOT


def test_root_path_follows_the_layout():
    layout = {"smooth_root_pos": [3], "global_root_heading": [2], "root_height": [1],
              "local_joints_positions": [4, 3]}
    assert root_path_dims(SimpleNamespace(size_dict=layout)) == list(range(6))


@pytest.mark.parametrize("motion_rep", [None, object(), SimpleNamespace(size_dict={})])
def test_no_layout_means_no_root_path(motion_rep):
    assert root_path_dims(motion_rep) == []


# --------------------------------------------------------------- fake model
class FakeMotionRep:
    """Kimodo's feature layout; `inverse` returns a root moving at `root_speed` m/s."""

    def __init__(self, root_speed: float, fps: float):
        self.size_dict = _kimodo_layout()
        self.motion_rep_dim = sum(int(torch.tensor(s).prod()) for s in self.size_dict.values())
        self.root_speed, self.fps = root_speed, fps

    def inverse(self, x, is_normalized=True, return_numpy=False):
        joints = torch.zeros(1, x.shape[1], NBJOINTS, 3)
        joints[0, :, 0, 0] = self.root_speed / self.fps * torch.arange(x.shape[1])
        joints[0, :, 0, 1] = 0.9
        return {"posed_joints": joints}


class FakeKimodo:
    """The attributes `generate_filtered_segment` touches, recording each denoiser and sampler call.

    The denoiser is 0.5 x + (text feature) W per batch row, so a text's prediction depends
    on the noisy sample and that text only; the sampler steps halfway to the estimate.
    """

    device = torch.device("cpu")
    fps = 30.0

    def __init__(self, root_speed: float = 0.0):
        self.motion_rep = FakeMotionRep(root_speed, self.fps)
        self.skeleton = SimpleNamespace(root_idx=0)
        self.diffusion = SimpleNamespace(space_timesteps=lambda steps: (list(range(steps)),
                                                                       10 * torch.arange(steps)),
                                         calc_diffusion_vars=lambda ut: None)
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(TEXT_DIM, self.motion_rep.motion_rep_dim, generator=g)
        self.guided, self.sampled = [], []

    def denoiser(self, cfg_weight, x, pad_mask, text_feat, text_pad_mask, t, heading,
                 motion_mask, observed_motion, cfg_type=None):
        out = 0.5 * x + (text_feat[:, 0, :] @ self.W)[:, None, :]
        if x.shape[0] == 1:
            self.guided.append(out.clone())
        return out

    def sampler(self, ut, x, x_hat, t):
        self.sampled.append(x_hat.clone())
        return 0.5 * x + 0.5 * x_hat


def _text(seed: int) -> torch.Tensor:
    return torch.randn(1, 1, TEXT_DIM, generator=torch.Generator().manual_seed(seed))


UNSAFE = torch.cat([_text(1), _text(2)])       # [K, 1, d]
SAFE, STAND = _text(3), _text(4)


def _cfg(**overrides) -> FilterConfig:
    values = dict(gamma=0.0, rho=1.0, ridge=1e-2, max_iters=20, root_dims=len(ROOT),
                  pin_dims=list(ROOT), stand_speed_threshold=0.3)
    values.update(overrides)
    return FilterConfig(**values)


def _generate(model, active_mask, *, safe=SAFE, stand=STAND, lock_head=0, seed=0):
    """One filtered segment whose command is the first rule's unsafe description."""
    torch.manual_seed(seed)
    return generate_filtered_segment(
        model, _cfg(), UNSAFE, safe, stand,
        UNSAFE[:1], torch.ones(1, 1, dtype=torch.bool), None,
        None, None, None, 2.0, "regular", FRAMES, STEPS,
        lock_head=lock_head, active_mask=active_mask,
    )


def _unfiltered(model, seed=0):
    """The plain sampling loop from the same noise."""
    torch.manual_seed(seed)
    x = torch.randn(1, FRAMES, model.motion_rep.motion_rep_dim)
    for _ in range(STEPS):
        x = 0.5 * x + 0.5 * (0.5 * x + (UNSAFE[0] @ model.W)[:, None, :])
    return x


# ---------------------------------------------------------------- sampling
@needs_kimodo
def test_filtered_segment_shapes_and_summary():
    model = FakeKimodo()
    x, info = _generate(model, [True, True])
    assert x.shape == (1, FRAMES, model.motion_rep.motion_rep_dim)
    assert info == {"corrected_steps": STEPS, "steps": STEPS}


@needs_kimodo
def test_all_false_mask_leaves_every_estimate_untouched():
    model = FakeKimodo()
    x, info = _generate(model, [False, False])
    assert info["corrected_steps"] == 0
    for guided, sampled in zip(model.guided[STEPS:], model.sampled[STEPS:]):
        assert torch.equal(guided, sampled)
    assert torch.allclose(x, _unfiltered(model), atol=1e-6)


@needs_kimodo
def test_root_path_follows_the_unfiltered_trajectory():
    x_on, _ = _generate(FakeKimodo(), [True, True])
    x_off, _ = _generate(FakeKimodo(), [False, False])
    assert torch.equal(x_on[..., ROOT], x_off[..., ROOT])
    assert not torch.allclose(x_on[..., len(ROOT):], x_off[..., len(ROOT):]), "the pose is filtered"


@needs_kimodo
def test_each_step_correction_leaves_the_root():
    model = FakeKimodo()
    _generate(model, [True, True])
    for guided, sampled in zip(model.guided[STEPS:], model.sampled[STEPS:]):
        assert torch.equal(sampled[..., ROOT], guided[..., ROOT])


@needs_kimodo
def test_locked_transition_frames_are_never_filtered():
    lock = 4
    model = FakeKimodo()
    x_on, _ = _generate(model, [True, True], lock_head=lock)
    x_off, _ = _generate(FakeKimodo(), [False, False])
    assert torch.equal(x_on[:, :lock], x_off[:, :lock])
    for guided, sampled in zip(model.guided[STEPS:], model.sampled[STEPS:]):
        assert torch.equal(sampled[:, :lock], guided[:, :lock])
    assert not torch.allclose(x_on[:, lock:], x_off[:, lock:])


@needs_kimodo
def test_a_still_clip_tracks_the_stand():
    x, _ = _generate(FakeKimodo(root_speed=0.0), [True, True])
    other_safe, _ = _generate(FakeKimodo(root_speed=0.0), [True, True], safe=_text(5))
    other_stand, _ = _generate(FakeKimodo(root_speed=0.0), [True, True], stand=_text(5))
    assert torch.equal(x, other_safe)
    assert not torch.allclose(x, other_stand)


@needs_kimodo
def test_a_moving_clip_tracks_the_segments_safe_reference():
    x, _ = _generate(FakeKimodo(root_speed=2.0), [True, True])
    other_stand, _ = _generate(FakeKimodo(root_speed=2.0), [True, True], stand=_text(5))
    other_safe, _ = _generate(FakeKimodo(root_speed=2.0), [True, True], safe=_text(5))
    assert torch.equal(x, other_stand)
    assert not torch.allclose(x, other_safe)


# ------------------------------------------------------------------ handle
TABLE = {
    "strike another person": [1, 0, 0, 0],
    "another person": [0, 1, 0, 0],
    "an inanimate object": [0, 0, 1, 0],
    "a person walks slowly": [0, 0, 0, 1],
    "punch the man": [0.95, 0.2, 0, 0],
    "a person walks forward": [0, 0, 0.05, 1],
}
RULES = [Rule(name="strike a person", unsafe="strike another person", protects="another person")]
POLICY = SafeReferencePolicy(library=("a person walks slowly",), stand="a person standing at ease",
                             min_similarity=0.6)


class FakeKimodoModel:
    """What the handle wraps: `_generate`, `_multiprompt`, `text_encoder`, `motion_rep`."""

    device = torch.device("cpu")

    def __init__(self, encoder):
        self.motion_rep = FakeMotionRep(0.0, 30.0)
        self.text_encoder = encoder.batch
        self.plain = []

    def _generate(self, texts, max_frames, num_denoising_steps, pad_mask=None,
                  first_heading_angle=None, motion_mask=None, observed_motion=None,
                  cfg_weight=2.0, text_feat=None, text_pad_mask=None, guide_masks=None,
                  cfg_type=None):
        self.plain.append(texts[0])
        return "plain"

    def _multiprompt(self, prompts, durations, num_transition_frames=5):
        return [self._generate([p], d, STEPS) for p, d in zip(prompts, durations)]


@pytest.fixture()
def kimodo(make_encoder, monkeypatch):
    calls = []

    def fake_segment(model, cfg, unsafe, safe, stand, *args, lock_head=0, active_mask=None, **kw):
        calls.append({"cfg": cfg, "unsafe": unsafe, "lock_head": lock_head, "mask": active_mask})
        return "filtered", {}

    monkeypatch.setattr(filtered_model, "generate_filtered_segment", fake_segment)
    model = FakeKimodoModel(make_encoder(TABLE))
    cfg = FilterConfig(rules=RULES, neutral_entities=["an inanimate object"])
    handle = filtered_model.attach_kimodo_filter(model, cfg, policy=POLICY)
    return model, handle, cfg, calls


def test_handle_excludes_and_pins_the_root_path(kimodo):
    _model, handle, cfg, _calls = kimodo
    assert handle.cfg.pin_dims == ROOT and handle.cfg.root_dims == len(ROOT)
    assert cfg.pin_dims == [] and cfg.root_dims == 0, "the caller's config is not modified"


def test_outside_filtering_generation_is_untouched(kimodo):
    model, handle, _cfg, calls = kimodo
    handle.set_entities(["another person"])
    assert model._generate(["punch the man"], FRAMES, STEPS) == "plain"
    assert calls == [] and handle.decisions == []


def test_inactive_segment_runs_the_original_sampler(kimodo):
    model, handle, _cfg, calls = kimodo
    with handle.filtering(["a person walks forward"], [FRAMES], STEPS):
        assert model._generate(["a person walks forward"], FRAMES, STEPS) == "plain"
    assert calls == [] and handle.decisions[0].inactive


def test_active_segments_are_filtered_with_locked_transitions(kimodo):
    model, handle, _cfg, calls = kimodo
    handle.set_entities(["another person"])
    prompts = ["punch the man", "a person walks forward"]
    with handle.filtering(prompts, [FRAMES, FRAMES], STEPS):
        assert model._multiprompt(prompts, [FRAMES, FRAMES], num_transition_frames=7) == [
            "filtered", "filtered"]
    assert [c["lock_head"] for c in calls] == [0, 7]
    assert [c["mask"] for c in calls] == [[True], [True]]
    assert calls[0]["unsafe"].shape == (len(RULES), 1, len(TABLE["another person"]))
    assert calls[0]["cfg"] is handle.cfg


def test_detach_restores_the_model(kimodo):
    model, handle, cfg, _calls = kimodo
    replaced = model._generate
    handle.detach()
    assert model._generate != replaced and not hasattr(model, "_csf_handle")
    assert model._generate(["x"], FRAMES, STEPS) == "plain"


def test_reattaching_replaces_the_previous_handle(kimodo):
    model, first, cfg, _calls = kimodo
    second = filtered_model.attach_kimodo_filter(model, cfg, policy=POLICY)
    assert model._csf_handle is second
    assert second._orig_generate == first._orig_generate, "the original sampler is not double-wrapped"
