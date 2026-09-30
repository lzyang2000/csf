# SPDX-License-Identifier: Apache-2.0
"""`ArdyModel`: the adapter protocol, the generation interface, and Eq. 9.

GPU-free.  `_WindowedArdy` stands in for a loaded ARDY model: it runs the
windowed autoregressive loop through ``self.denoiser`` exactly where
``Ardy.denoising_step`` calls it, and commits each window's final clean
prediction, so filtering inside the sampling loop can be checked end to end.
"""
from __future__ import annotations

import inspect
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from csf.backbones.ardy.filter_proxy import FilteredDenoiser  # noqa: E402
from csf.backbones.ardy.model import ArdyModel, default_history_frames  # noqa: E402
from csf.backbones.base import FilterAdapter, PerSegmentFilterMixin  # noqa: E402
from csf.filter.config import FilterConfig, Rule, SafeReferencePolicy  # noqa: E402
from csf.filter.gate import GateDecision  # noqa: E402
from csf.filter.handle import AdapterFilterHandle  # noqa: E402

ROOT = 2            # explicit root channels per token (20 on ARDY-G1)
D = 6               # hybrid token dim: 2 root + 4 body
FPT = 4             # frames per token
HORIZON = 8         # frames per autoregressive window => 2 tokens
MOTION_D = 8        # explicit motion feature dim
J = 3               # joints


class _Denoiser(nn.Module):
    """ARDY's CFG denoiser: sets every generation token to `gen_value`."""

    latent_embedding_dim = D - ROOT
    nframe_root_dim = ROOT

    def __init__(self):
        super().__init__()
        self.gen_value = torch.zeros(D)

    def forward(self, *args):
        x, generation_token_mask = args[2], args[10]
        out = x.clone()
        out[generation_token_mask] = self.gen_value
        return out


class _MotionRep:
    """Decodes features into kimodo's output dict; remembers the last input."""

    def __init__(self):
        self.last_features = None

    def inverse(self, features, is_normalized):
        assert is_normalized
        self.last_features = features
        B, T = features.shape[:2]
        return {
            "posed_joints": torch.zeros(B, T, J, 3),
            "global_rot_mats": torch.eye(3).expand(B, T, J, 3, 3),
            "local_rot_mats": torch.eye(3).expand(B, T, J, 3, 3),
            "root_positions": torch.zeros(B, T, 3),
            "smooth_root_pos": torch.zeros(B, T, 3),
            "foot_contacts": torch.zeros(B, T, 4, dtype=torch.bool),
            "global_root_heading": torch.zeros(B, T, 2),
        }


class _Hybrid:
    """The tokenizer: folds FPT frames per token (and insists on whole tokens)."""

    def get_hybrid_motion_from_explicit(self, motion, motion_len, motion_pad_mask):
        B, T, _ = motion.shape
        assert T % FPT == 0, f"cannot fold {T} frames into tokens of {FPT}"
        return motion[:, ::FPT, :D].clone(), torch.ones(B, T // FPT, dtype=torch.bool)


class _WindowedArdy:
    """Stand-in for ``ardy.model.ardy_model.Ardy``."""

    class _Diffusion:
        num_base_steps = 4

    num_frames_per_token = FPT
    gen_horizon_len = HORIZON

    def __init__(self, motion_value=None):
        self.denoiser = _Denoiser()
        self.diffusion = self._Diffusion()
        self.motion_rep = _MotionRep()
        self.hybrid = _Hybrid()
        self.motion_value = motion_value or {}
        self.calls = []          # one record per rollout
        self.committed = None    # hybrid tokens of the last rollout

    def __call__(self, texts, num_frames, *, num_denoising_steps, **kwargs):
        B, n = len(texts), HORIZON // FPT
        proxy = self.denoiser if isinstance(self.denoiser, FilteredDenoiser) else None
        self.calls.append({
            "texts": list(texts), "num_frames": num_frames, "steps": num_denoising_steps,
            "filtered": proxy is not None,
            "mask": None if proxy is None else proxy._active_mask, **kwargs,
        })
        history = torch.zeros(B, 0, D)
        for _ in range(math.ceil(num_frames / HORIZON)):
            x = torch.cat([history, torch.zeros(B, n, D)], dim=1)
            generation_token_mask = torch.zeros(B, x.shape[1], dtype=torch.bool)
            generation_token_mask[:, -n:] = True
            for t in reversed(range(num_denoising_steps)):
                args = [None] * 18
                args[2], args[10], args[14] = x, generation_token_mask, torch.full((B,), t)
                x = self.denoiser(*args)
            history = x                               # commit the window
        self.committed = history
        value = self.motion_value.get(texts[0], 0.0)
        return torch.full((B, num_frames, MOTION_D), float(value))


def _model(**kwargs):
    ardy = _WindowedArdy(**kwargs)
    return ArdyModel(ardy, skeleton=None, text_encoder=None, device="cpu"), ardy


def _references(T_tok=4):
    unsafe = torch.zeros(2, 1, T_tok, D)
    unsafe[0, ..., ROOT] = 1.0
    unsafe[1, ..., ROOT + 1] = 1.0
    return unsafe, torch.zeros(1, T_tok, D)


# ------------------------------------------------------------------ protocol
def test_implements_the_adapter_protocol():
    assert issubclass(ArdyModel, PerSegmentFilterMixin)
    for name in ("reference_feat", "set_filter_context", "clear_filter_context",
                 "set_filter_segments"):
        want = inspect.signature(getattr(FilterAdapter, name)).parameters
        got = inspect.signature(getattr(ArdyModel, name)).parameters
        assert list(got) == list(want), name
        assert [p.default for p in got.values()] == [p.default for p in want.values()], name
    model, _ = _model()
    for attribute in ("motion_rep", "skeleton", "text_encoder", "fps", "device",
                      "default_denoising_steps", "root_dims"):
        assert hasattr(model, attribute), attribute


def test_root_dims_is_the_explicit_root_block():
    model, ardy = _model()
    assert model.root_dims == ROOT
    model.set_filter_context(*_references(), FilterConfig())
    assert model.root_dims == ROOT, "the proxy must expose the wrapped denoiser's width"


def test_set_and_clear_swap_the_denoiser():
    model, ardy = _model()
    inner = ardy.denoiser
    model.set_filter_context(*_references(), FilterConfig())
    assert isinstance(ardy.denoiser, FilteredDenoiser) and ardy.denoiser.inner is inner

    model.set_filter_context(*_references(), FilterConfig())    # re-install: no nesting
    assert ardy.denoiser.inner is inner

    model.clear_filter_context()
    assert ardy.denoiser is inner

    model.set_filter_context(*_references(), FilterConfig())
    model.set_filter_context(None, None, FilterConfig())        # None clears
    assert ardy.denoiser is inner


# ----------------------------------------------------------------- references
def test_reference_feat_crops_to_whole_tokens():
    model, _ = _model()
    tokens = model.reference_feat("a person waves", 179, 4)
    assert tokens.shape == (179 // FPT, D)
    assert tokens.dtype == torch.float32


def test_reference_feat_runs_unfiltered_and_restores_the_filter():
    model, ardy = _model()
    model.set_filter_context(*_references(), FilterConfig())
    proxy = ardy.denoiser
    model.reference_feat("a person waves", 16, 4)
    assert ardy.calls[-1]["filtered"] is False
    assert ardy.denoiser is proxy and model._proxy is proxy


# ----------------------------------------------------------------- generation
def test_cfg_weight_pairs_collapse_to_the_text_weight():
    model, ardy = _model()
    model._generate(["hi"], 8, 4, cfg_weight=[2.5, 2.0])
    assert ardy.calls[-1]["cfg_weight"] == 2.5
    model._generate(["hi"], 8, 4, cfg_weight=3.5)
    assert ardy.calls[-1]["cfg_weight"] == 3.5
    model._generate(["hi"], 8, 4, cfg_weight=None)
    assert ardy.calls[-1]["cfg_weight"] == 2.0            # the configured default


def test_steps_are_capped_at_the_trained_count():
    model, ardy = _model()
    assert model.default_denoising_steps == 4
    model._generate(["hi"], 8, 100)
    assert ardy.calls[-1]["steps"] == 4
    model._generate(["hi"], 8, None)
    assert ardy.calls[-1]["steps"] == 4


def test_multi_prompt_returns_one_stitched_motion():
    model, ardy = _model(motion_value={"a": 1.0, "b": 3.0})
    out = model(["a", "b"], [10, 12], 4, multi_prompt=True, constraint_lst=[],
                cfg_weight=[2.0, 2.0], num_samples=1, cfg_type=None,
                post_processing=False)
    total = 10 + 12 - 5                                     # one 5-frame cross-fade
    assert out["posed_joints"].shape == (1, total, J, 3)
    assert out["global_rot_mats"].shape == (1, total, J, 3, 3)
    assert out["smooth_root_pos"].shape == (1, total, 3)
    assert "foot_contacts" in out

    feats = ardy.motion_rep.last_features[0, :, 0]
    alpha = torch.linspace(1.0, 0.0, 5)
    assert torch.allclose(feats[:5], torch.ones(5))
    assert torch.allclose(feats[5:10], alpha * 1.0 + (1 - alpha) * 3.0)
    assert torch.allclose(feats[10:], torch.full((7,), 3.0))


def test_batched_generation_needs_equal_lengths():
    model, _ = _model()
    out = model(["a", "b"], [8, 8], 4)
    assert out["posed_joints"].shape == (2, 8, J, 3)
    with pytest.raises(ValueError, match="equal num_frames"):
        model(["a", "b"], [8, 12], 4)


def test_return_numpy():
    model, _ = _model()
    out = model("a", 8, 4, multi_prompt=True, return_numpy=True)
    assert out["posed_joints"].shape == (1, 8, J, 3)
    assert not isinstance(out["posed_joints"], torch.Tensor)


def test_default_history_frames_is_the_ten_second_rule():
    assert default_history_frames(25, 4, 52) == 196         # ARDY-G1-RP-25FPS-Horizon52


# ---------------------------------------------------------------------- Eq. 9
def test_each_window_is_filtered_before_it_is_committed():
    """gamma = rho = 1 moves each window's body tokens onto its own safe block,
    while the root channels keep the generator's prediction."""
    model, ardy = _model()
    ardy.denoiser.gen_value = torch.tensor([0.7, -0.3, 2.0, 0.0, 0.0, 0.0])
    unsafe, _ = _references(T_tok=4)
    safe = torch.zeros(1, 4, D)
    safe[0, :2, ROOT + 2] = 1.0                             # window 0 block
    safe[0, 2:, ROOT + 3] = 5.0                             # window 1 block
    cfg = FilterConfig(gamma=1.0, rho=1.0, root_dims=model.root_dims)
    model.set_filter_context(unsafe, safe, cfg)

    for _ in range(2):                                      # a second rollout realigns
        model("a", 2 * HORIZON, 4, multi_prompt=True)
        committed = ardy.committed[0]                       # [4 tokens, D]
        assert committed.shape == (4, D)
        assert torch.allclose(committed[:, :ROOT], torch.tensor([0.7, -0.3]).expand(4, ROOT))
        assert torch.allclose(committed[:, ROOT:], safe[0, :, ROOT:], atol=1e-5)


def test_the_barrier_holds_on_committed_tokens():
    model, ardy = _model()
    ardy.denoiser.gen_value = torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # along rule 0
    unsafe, safe = _references(T_tok=4)
    model.set_filter_context(unsafe, safe, FilterConfig(gamma=0.0, root_dims=ROOT))
    model("a", 2 * HORIZON, 4, multi_prompt=True)
    for window in (slice(0, 2), slice(2, 4)):
        v = ardy.committed[0, window, ROOT:].reshape(-1)
        d = unsafe[0, 0, window, ROOT:].reshape(-1)
        assert float(-(v @ d) / d.norm()) > -0.1


# --------------------------------------------------------------------- handle
def _filter_cfg():
    return FilterConfig(
        rules=[Rule("strike", "a person punching another person", "another person"),
               Rule("kick", "a person kicking another person", "another person")],
        gamma=0.2,
    )


def _policy():
    return SafeReferencePolicy(library=("a person standing still",),
                               stand="a person standing still")


def test_the_handle_excludes_ardys_root_block():
    model, _ = _model()
    cfg = _filter_cfg()
    handle = AdapterFilterHandle(model, cfg, _policy())
    assert handle.cfg.root_dims == ROOT
    assert cfg.root_dims == 0, "the caller's config must not be mutated"


def test_the_handle_installs_references_for_one_generation():
    model, ardy = _model()
    inner = ardy.denoiser
    handle = AdapterFilterHandle(model, _filter_cfg(), _policy())
    with handle.filtering(["a person punches"], [16], 4):
        proxy = ardy.denoiser
        assert isinstance(proxy, FilteredDenoiser)
        assert proxy._unsafe.shape == (2, 16 // FPT, D)
        assert proxy._safe.shape == (16 // FPT, D)
        assert proxy._cfg.root_dims == ROOT and proxy._cfg.gamma == 0.2
        model(["a person punches"], [16], 4, multi_prompt=True)
        assert ardy.calls[-1]["filtered"]
    assert ardy.denoiser is inner
    assert not model._filter_per_prompt


WALK = "a person walks forward"
KICK = "a person kicks a box"


def test_each_segment_is_filtered_with_its_own_decision(monkeypatch):
    model, ardy = _model()
    handle = AdapterFilterHandle(model, _filter_cfg(), _policy())

    def decide(prompt):
        mask = [True, True] if prompt == KICK else [False, False]
        return GateDecision(mask=mask, reasons=[[], []], prompt=prompt, entities=(),
                            safe_reference="a person standing still")

    monkeypatch.setattr(handle, "decide", decide)
    with handle.filtering([WALK, KICK], [8, 16], 4):
        ardy.calls.clear()
        model([WALK, KICK], [8, 16], 4, multi_prompt=True)

    walk, kick = ardy.calls
    assert walk["texts"] == [WALK] and not walk["filtered"], "the benign walk was filtered"
    assert kick["texts"] == [KICK] and kick["filtered"] and kick["mask"] == [True, True]
    assert ardy.denoiser is not None and not isinstance(ardy.denoiser, FilteredDenoiser)
