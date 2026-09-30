# SPDX-License-Identifier: Apache-2.0
"""EchoModel: kimodo's generation interface and the CSF adapter protocol (mock pipeline)."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")                     # kimodo's G1Skeleton34 needs it

D = 38
CONTRACT_KEYS = ["posed_joints", "root_positions", "global_rot_mats", "local_rot_mats",
                 "smooth_root_pos", "foot_contacts", "global_root_heading"]


class MockPipeline:
    """ECHO's `DiffusePipeline` surface: `generate(caption list, m_lens)` plus the loop fields."""

    input_feats = D

    def __init__(self, value: float = 0.0):
        self.value = value
        self.model = self
        self.cond_mask_prob = 0
        self.scheduler = SimpleNamespace(
            timesteps=torch.arange(2), set_timesteps=lambda n, device=None: None,
            step=lambda predict, t, sample: SimpleNamespace(prev_sample=predict))
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.num_inference_steps = 10
        self.native_calls = 0
        self.filtered_forwards = 0

    # native path (no kwargs, list captions: the real signature)
    def generate(self, caption, m_lens, batch_size=32, timing=False):
        assert isinstance(caption, (list, tuple)), f"caption must be list[str], got {type(caption)}"
        self.native_calls += 1
        return [torch.full((int(n), D), self.value) for n in m_lens]

    # filtered path: the pipeline doubles as ECHO's model
    def encode_text(self, caption, device):
        return torch.zeros(len(caption), 1, 8)

    def __call__(self, sample, t, enc_text=None, lengths=None):
        self.filtered_forwards += 1
        return torch.full_like(sample, self.value)

    def eval(self):
        return self


def _make_model(pipeline=None, mean=None, std=None):
    from kimodo.skeleton.registry import build_skeleton

    from csf.backbones.echo.model import EchoModel
    from csf.backbones.echo.motion_rep import EchoMotionRep

    skeleton = build_skeleton(34)
    return EchoModel(pipeline or MockPipeline(),
                     np.zeros(D, np.float32) if mean is None else mean,
                     np.ones(D, np.float32) if std is None else std,
                     EchoMotionRep(skeleton, fps=50), skeleton, text_encoder=None, device="cpu")


# ------------------------------------------------------------ generation contract
def test_call_returns_contract_dict():
    out = _make_model()("a person walks", 10, num_denoising_steps=4, return_numpy=True)
    for key in CONTRACT_KEYS:
        assert key in out, f"missing key: {key}"
    assert out["posed_joints"].shape == (10, 34, 3)


def test_generate_returns_normalized_38d():
    z = _make_model()._generate(["a person walks"], 10, num_denoising_steps=4)
    assert tuple(z.shape) == (1, 10, D)


def test_generate_batched_and_ignores_kimodo_kwargs():
    z = _make_model()._generate(["walk", "run"], 6, num_denoising_steps=2, pad_mask=None,
                                first_heading_angle=None, motion_mask=None, observed_motion=None,
                                cfg_weight=[2.0, 2.0], cfg_type=None, text_feat=None,
                                text_pad_mask=None, progress_bar=None, guide_masks=None)
    assert z.shape == (2, 6, D)


def test_model_attributes():
    from kimodo.skeleton import G1Skeleton34

    m = _make_model()
    assert m.fps == 50 and m.text_encoder is None and m.device == "cpu"
    assert isinstance(m.skeleton, G1Skeleton34)
    assert m.motion_rep is not None
    assert m.default_denoising_steps == 10
    assert m.root_dims == 0


def test_call_list_prompts():
    out = _make_model()(["walk", "run"], 8, num_denoising_steps=2, return_numpy=True)
    assert out["posed_joints"].shape[:3] == (2, 8, 34)


def test_call_the_way_the_app_does():
    """model(prompts, durations, steps, multi_prompt=True, constraint_lst=[], cfg_weight=, ...)."""
    out = _make_model()(["walk", "stop"], [8, 8], 2, multi_prompt=True, constraint_lst=[],
                        cfg_weight=[2.0, 2.0], num_samples=1, cfg_type="separated",
                        post_processing=False)
    assert out["posed_joints"].shape[0] == 1 and out["posed_joints"].shape[2] == 34
    assert out["posed_joints"].shape[1] >= 8
    assert out["global_rot_mats"].shape[:3] == (1, out["posed_joints"].shape[1], 34)
    assert out["foot_contacts"].shape[:2] == out["posed_joints"].shape[:2]
    assert out["smooth_root_pos"].shape == (1, out["posed_joints"].shape[1], 3)


def test_multi_prompt_cross_fade_length():
    """Two 8-frame segments cross-faded over 5 frames: 3 + 5 + 3 = 11 frames."""
    out = _make_model()(["walk", "stop"], 8, 2, multi_prompt=True, return_numpy=True)
    assert out["posed_joints"].shape[1] == 11


def test_denormalisation_reaches_motion_rep_once():
    """The pipeline returns ones; with mean 0.5 and std 2 the decoder sees 2.5 everywhere."""
    model = _make_model(MockPipeline(value=1.0), mean=np.full(D, 0.5, np.float32),
                        std=np.full(D, 2.0, np.float32))
    captured = {}
    original = model.motion_rep.inverse

    def capture(x, **kw):
        captured["x"] = x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)
        return original(x, **kw)

    model.motion_rep.inverse = capture
    model("a person walks", 6, num_denoising_steps=2, return_numpy=True)
    np.testing.assert_allclose(captured["x"], 2.5, rtol=1e-5, atol=1e-5)


def test_batched_call_with_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal num_frames"):
        _make_model()(["walk", "run"], [8, 12], num_denoising_steps=2)


# ------------------------------------------------------------------ CSF protocol
def test_reference_feat_is_unfiltered_and_restores_the_context():
    m = _make_model()

    def must_not_run(*a, **k):
        raise AssertionError("the filter ran while building a reference")

    m._step_callback = must_not_run
    feat = m.reference_feat("a person punching", 10, 4)
    assert tuple(feat.shape) == (10, D)
    assert m._step_callback is must_not_run


def test_protocol_signatures():
    from csf.backbones.base import PerSegmentFilterMixin
    from csf.backbones.echo.model import EchoModel

    assert issubclass(EchoModel, PerSegmentFilterMixin)
    params = list(inspect.signature(EchoModel.set_filter_context).parameters)
    assert params == ["self", "unsafe", "safe", "cfg", "active_mask"]
    assert list(inspect.signature(EchoModel.reference_feat).parameters) == [
        "self", "text", "num_frames", "num_steps"]
    for name in ("clear_filter_context", "set_filter_segments", "_activate_filter_for"):
        assert callable(getattr(EchoModel, name))


def test_each_segment_is_filtered_with_its_own_context():
    """A gated-off segment runs natively; a gated-on one runs the filtered loop."""
    from csf.filter.config import FilterConfig

    pipe = MockPipeline()
    m = _make_model(pipe)
    unsafe, safe = torch.ones(1, 1, 1, D), torch.zeros(1, 1, D)
    m.set_filter_segments({
        "walk": (None, None, None, None),
        "kick": (unsafe, safe, FilterConfig(), [True]),
    })
    m(["walk", "kick"], [8, 8], 2, multi_prompt=True)
    assert pipe.native_calls == 1                       # walk
    assert pipe.filtered_forwards == 2                  # kick: one forward per step
    m.set_filter_segments(None)


def test_adapter_filter_handle_drives_the_model():
    """The core handle installs a context for the schedule and clears it afterwards."""
    from csf.filter.config import FilterConfig, Rule, SafeReferencePolicy
    from csf.filter.handle import AdapterFilterHandle

    pipe = MockPipeline()
    m = _make_model(pipe)
    cfg = FilterConfig(rules=[Rule("strike", "a person punching another person", "another person"),
                              Rule("kick", "a person kicking another person", "another person")],
                       gamma=0.2)
    policy = SafeReferencePolicy(library=("a person stands still",), stand="a person stands still")
    handle = AdapterFilterHandle(m, cfg, policy, encoder=None)
    assert handle.cfg.root_dims == 0

    with handle.filtering(["a person punches", "a person walks"], [8, 8], 2):
        assert m._step_callback is not None
        assert set(m._filter_per_prompt) == {"a person punches", "a person walks"}
        m(["a person punches", "a person walks"], [8, 8], 2, multi_prompt=True)
    assert m._step_callback is None and not m._filter_per_prompt
    assert pipe.filtered_forwards == 4                  # both segments filtered, 2 steps each
    assert [d.prompt for d in handle.decisions] == ["a person punches", "a person walks"]
