# SPDX-License-Identifier: Apache-2.0
"""MHFModel: kimodo's generation interface and the CSF adapter protocol (mock VAE and DiT)."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

CONTRACT_KEYS = ["posed_joints", "root_positions", "global_rot_mats", "local_rot_mats",
                 "smooth_root_pos", "foot_contacts", "global_root_heading"]


class MockFlow:
    """`FlowModel.generate(text, m_lengths, ...) -> (latent [B, T // 4, 6, 16], lengths)`."""

    def __init__(self, tuple_return: bool = True):
        self.tuple_return = tuple_return
        self.calls = []

    def generate(self, text, m_lengths, time_steps=12, cond_scale=4.5, **kw):
        self.calls.append(dict(text=list(text), m_lengths=m_lengths.tolist(),
                               time_steps=time_steps, cond_scale=cond_scale))
        latent = torch.zeros(len(text), max(int(m_lengths.max()), 1), 6, 16)
        return (latent, m_lengths) if self.tuple_return else latent


class MockVAE:
    """`decode(latent [B, T, 6, 16]) -> [B, 4T, 263]`, constant `value`."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def decode(self, z):
        return torch.full((z.shape[0], z.shape[1] * 4, 263), self.value)


def _skeleton():
    from kimodo.skeleton.definitions import SMPLXSkeleton22

    return SMPLXSkeleton22(load=True)


def _make_model(flow=None, vae=None, inv_transform=None, cond_scale=4.5, rest=False):
    from csf.backbones.motionhiflow.model import MHFModel
    from csf.backbones.motionhiflow.motion_rep import MHFMotionRep

    skeleton = _skeleton()
    rep = (MHFMotionRep(22, 20, rest_joints=skeleton.neutral_joints,
                        joint_parents=skeleton.joint_parents)
           if rest else MHFMotionRep(joints_num=22, fps=20))
    return MHFModel(vae=vae or MockVAE(), flow=flow or MockFlow(),
                    inv_transform=inv_transform or (lambda x: x), motion_rep=rep,
                    skeleton=skeleton, text_encoder=None, device="cpu",
                    cond_scale=cond_scale, time_steps=12)


# ------------------------------------------------------------ generation contract
def test_call_returns_contract_dict():
    out = _make_model()("a person walks", 40, num_denoising_steps=12, return_numpy=True)
    for key in CONTRACT_KEYS:
        assert key in out, f"missing key: {key}"
    assert out["posed_joints"].shape == (40, 22, 3)


def test_generate_returns_normalized_263d_in_token_lengths():
    flow = MockFlow()
    z = _make_model(flow=flow)._generate(["walk", "run"], 40, num_denoising_steps=4)
    assert z.shape == (2, 40, 263)
    assert flow.calls[-1]["m_lengths"] == [10, 10]                 # frames // 4 tokens


def test_generate_ignores_kimodo_kwargs():
    z = _make_model()._generate(["test"], 40, num_denoising_steps=4, pad_mask=None,
                                first_heading_angle=None, motion_mask=None, observed_motion=None,
                                cfg_weight=[2.0, 2.0], cfg_type=None, text_feat=None,
                                text_pad_mask=None, progress_bar=None, guide_masks=None)
    assert z.shape[-1] == 263


def test_denormalisation_reaches_motion_rep_once():
    from csf.backbones.motionhiflow.model import MHFModel
    from csf.backbones.motionhiflow.motion_rep import MHFMotionRep

    captured = {}

    class Capture(MHFMotionRep):
        def inverse(self, x263, **kw):
            captured["x"] = np.asarray(x263).copy()
            return super().inverse(x263, **kw)

    model = MHFModel(vae=MockVAE(1.0), flow=MockFlow(), inv_transform=lambda d: d * 2.0 + 0.5,
                     motion_rep=Capture(22, 20), skeleton=None, text_encoder=None, device="cpu")
    model("a person walks", 40, num_denoising_steps=4)
    np.testing.assert_allclose(captured["x"], 2.5, atol=1e-5)


def test_model_attributes():
    from kimodo.skeleton.definitions import SMPLXSkeleton22

    m = _make_model()
    assert m.fps == 20 and m.text_encoder is None and m.device == "cpu"
    assert isinstance(m.skeleton, SMPLXSkeleton22)
    assert m.default_denoising_steps == 12
    assert m.root_dims == 0


def test_call_list_prompts():
    out = _make_model()(["walk", "run"], 40, num_denoising_steps=4, return_numpy=True)
    assert out["posed_joints"].shape == (2, 40, 22, 3)


def test_call_multi_prompt_the_way_the_app_does():
    """Two 40-frame segments cross-faded over 5 frames: 35 + 5 + 35 = 75 frames, batched."""
    out = _make_model(rest=True)(["walk", "stop"], [40, 40], 4, multi_prompt=True,
                                 constraint_lst=[], cfg_weight=[2.0, 2.0], num_samples=1,
                                 cfg_type="separated")
    assert out["posed_joints"].shape == (1, 75, 22, 3)
    assert out["global_rot_mats"].shape == (1, 75, 22, 3, 3)
    assert out["foot_contacts"].shape[:2] == (1, 75)
    assert out["smooth_root_pos"].shape == (1, 75, 3)


def test_batched_call_with_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal num_frames"):
        _make_model()(["walk", "run"], [40, 80], num_denoising_steps=4)


def test_flow_may_return_a_bare_latent():
    out = _make_model(flow=MockFlow(tuple_return=False))("a person walks", 40, return_numpy=True)
    assert out["posed_joints"].shape == (40, 22, 3)


def test_step_count_and_guidance_scale_reach_the_flow():
    flow = MockFlow()
    m = _make_model(flow=flow, cond_scale=3.0)
    m("a person walks", 40, num_denoising_steps=7)
    assert flow.calls[-1]["time_steps"] == 7 and flow.calls[-1]["cond_scale"] == 3.0
    m("a person walks", 40)
    assert flow.calls[-1]["time_steps"] == 12


def test_return_numpy_false_returns_tensors():
    for key, value in _make_model()("a person walks", 40).items():
        assert isinstance(value, torch.Tensor), key


# ------------------------------------------------------------------ CSF protocol
def test_reference_feat_is_the_unfiltered_clean_latent():
    m = _make_model()

    def must_not_run(*a, **k):
        raise AssertionError("the filter ran while building a reference")

    m._latent_callback = must_not_run
    feat = m.reference_feat("a person punching", 40, 4)
    assert tuple(feat.shape) == (10, 6, 16)
    assert m._latent_callback is must_not_run


def test_protocol_signatures():
    from csf.backbones.base import PerSegmentFilterMixin
    from csf.backbones.motionhiflow.model import MHFModel

    assert issubclass(MHFModel, PerSegmentFilterMixin)
    assert list(inspect.signature(MHFModel.set_filter_context).parameters) == [
        "self", "unsafe", "safe", "cfg", "active_mask"]
    assert list(inspect.signature(MHFModel.reference_feat).parameters) == [
        "self", "text", "num_frames", "num_steps"]
    for name in ("clear_filter_context", "set_filter_segments", "_activate_filter_for"):
        assert callable(getattr(MHFModel, name))


def test_each_segment_is_filtered_with_its_own_context(monkeypatch):
    """A gated-off segment samples natively; a gated-on one runs the filtered sampler."""
    import csf.backbones.motionhiflow.filtered_sampler as sampler
    from csf.filter.config import FilterConfig

    runs = []
    monkeypatch.setattr(sampler, "filtered_flow_generate",
                        lambda flow, text, m_lengths, **kw: runs.append(list(text)) or
                        (torch.zeros(1, int(m_lengths.max()), 6, 16), m_lengths))
    flow = MockFlow()
    m = _make_model(flow=flow)
    m.set_filter_segments({
        "walk": (None, None, None, None),
        "kick": (torch.ones(1, 1, 10, 6, 16), torch.zeros(1, 10, 6, 16), FilterConfig(), [True]),
    })
    m(["walk", "kick"], [40, 40], 4, multi_prompt=True)
    assert [c["text"] for c in flow.calls] == [["walk"]]
    assert runs == [["kick"]]


def test_adapter_filter_handle_drives_the_model(monkeypatch):
    import csf.backbones.motionhiflow.filtered_sampler as sampler
    from csf.filter.config import FilterConfig, Rule, SafeReferencePolicy
    from csf.filter.handle import AdapterFilterHandle

    seen = []

    def fake_filtered(flow, text, m_lengths, step_callback=None, **kw):
        latent = torch.zeros(1, int(m_lengths.max()), 6, 16)
        seen.append(step_callback(latent, torch.tensor(0.5), 0, 0).shape)
        return latent, m_lengths

    monkeypatch.setattr(sampler, "filtered_flow_generate", fake_filtered)
    m = _make_model()
    cfg = FilterConfig(rules=[Rule("strike", "a person punching another person", "another person")],
                       gamma=0.5)
    policy = SafeReferencePolicy(library=("a person stands still",), stand="a person stands still")
    handle = AdapterFilterHandle(m, cfg, policy, encoder=None)
    with handle.filtering(["a person punches"], [40], 4):
        m(["a person punches"], [40], 4, multi_prompt=True)
    assert seen == [torch.Size([1, 10, 6, 16])]
    assert m._latent_callback is None
