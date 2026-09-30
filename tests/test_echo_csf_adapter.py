# SPDX-License-Identifier: Apache-2.0
"""The per-step filter on ECHO's 38-d estimate, and EchoModel's routing through it.

GPU-free, with synthetic references.  Sign convention of the margin (Eq. 1):
an estimate aligned with an unsafe reference (safe reference at the origin)
has a negative margin and is corrected; one on the far side of the safe
reference has a positive margin and passes through when gamma = 0.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csf.backbones.echo.csf_adapter import make_step_callback, resample_time
from csf.filter.config import FilterConfig

D = 38


def _cfg(gamma: float = 0.0, rho: float = 1.0) -> FilterConfig:
    """Barrier-only by default; root_dims = 0 as the handle sets it for ECHO."""
    return FilterConfig(gamma=gamma, rho=rho, root_dims=0)


def _margin(x, unsafe, safe):
    """Eq. 1 over the flattened clip, per-frame references broadcast over x's frames."""
    x = x if x.dim() == 3 else x.unsqueeze(0)
    T = x.shape[1]
    d = (unsafe - safe).reshape(1, -1).expand(T, -1).reshape(-1)
    v = (x[0] - safe.reshape(1, -1)).reshape(-1)
    return float(-(v @ d) / (d.norm() + 1e-8))


def _axis(T, k, value):
    """A [1, T, D] clip with feature k set to value."""
    x = torch.zeros(1, T, D)
    x[..., k] = value
    return x


# ------------------------------------------------------------------ filter math
def test_filter_corrects_unsafe_and_keeps_safe():
    T = 4
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = 1.0
    safe = torch.zeros(1, 1, D)
    cb = make_step_callback(unsafe, safe, cfg=_cfg())

    x_bad = _axis(T, 0, 1.0)
    m_before = _margin(x_bad, unsafe, safe)
    assert m_before < 0.0
    y = cb(x_bad.clone(), t=torch.tensor(0), step_idx=0)
    m_after = _margin(y, unsafe, safe)
    # rho = 1 imposes the barrier up to the ridge residual (Prop. 1).
    assert m_after >= -0.1 and m_after > m_before + 1.0
    assert float(y[..., 0].abs().max()) < 1.0

    x_ok = _axis(T, 0, -1.0)
    assert _margin(x_ok, unsafe, safe) > 0.0
    assert torch.allclose(cb(x_ok.clone(), t=torch.tensor(0), step_idx=0), x_ok, atol=1e-5)


def test_ridge_residual_matches_eq_ridge_single():
    """A single violated row at rho = 1, gamma = 0 keeps eps/(1+eps) of the margin."""
    T, eps = 3, 1e-2
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = 1.0
    cb = make_step_callback(unsafe, torch.zeros(1, 1, D), cfg=_cfg())
    x = _axis(T, 0, 1.0)
    before = _margin(x, unsafe, torch.zeros(1, 1, D))
    after = _margin(cb(x, t=torch.tensor(0), step_idx=0), unsafe, torch.zeros(1, 1, D))
    assert after == pytest.approx(eps / (1 + eps) * before, rel=1e-3)


def test_multiple_rules_corrected_jointly():
    T = 3
    unsafe = torch.zeros(2, 1, D)
    unsafe[0, :, 0] = 1.0
    unsafe[1, :, 1] = 1.0
    safe = torch.zeros(1, 1, D)
    cb = make_step_callback(unsafe, safe, cfg=_cfg())
    x = torch.zeros(1, T, D)
    x[..., 0] = 1.0
    x[..., 1] = 1.0
    y = cb(x.clone(), t=torch.tensor(0), step_idx=0)
    for k in range(2):
        assert _margin(y, unsafe[k], safe) >= -0.1


def test_inactive_rules_leave_the_estimate_unchanged():
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = 1.0
    cb = make_step_callback(unsafe, torch.zeros(1, 1, D), cfg=_cfg(gamma=0.5), active_mask=[False])
    x = _axis(3, 0, 1.0)
    assert torch.allclose(cb(x.clone(), t=torch.tensor(0), step_idx=0), x, atol=1e-6)


def test_tracking_blends_toward_the_safe_reference():
    """Feasible estimate, gamma > 0: x_filt = (1 - gamma) x_hat + gamma x_safe (Thm. 1(iii))."""
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = 1.0
    safe = torch.full((1, 1, D), 0.1)
    safe[..., 0] = -2.0
    cb = make_step_callback(unsafe, safe, cfg=_cfg(gamma=0.2))
    x = _axis(4, 0, -3.0)                          # far on the safe side of the barrier
    assert _margin(x, unsafe, safe) > 0.0
    y = cb(x.clone(), t=torch.tensor(0), step_idx=0)
    assert torch.allclose(y, 0.8 * x + 0.2 * safe.expand_as(x), atol=1e-5)


def test_partial_contraction_rate():
    """rho < 1 contracts the violation to (1 - rho) of its value, up to the ridge residual."""
    eps, rho = 1e-2, 0.5
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = 1.0
    safe = torch.zeros(1, 1, D)
    x = _axis(3, 0, 1.0)
    before = _margin(x, unsafe, safe)
    after = _margin(make_step_callback(unsafe, safe, cfg=_cfg(rho=rho))(x, torch.tensor(0), 0),
                    unsafe, safe)
    assert after == pytest.approx((1 - rho + eps) / (1 + eps) * before, rel=1e-3)


# -------------------------------------------------------------- reference shapes
def test_2d_references_match_3d_references():
    T, K = 5, 2
    unsafe_2d = torch.zeros(K, D)
    unsafe_2d[0, 0] = 1.0
    unsafe_2d[1, 1] = 1.0
    safe_2d = torch.zeros(1, D)
    cb_2d = make_step_callback(unsafe_2d, safe_2d, cfg=_cfg())
    cb_3d = make_step_callback(unsafe_2d.unsqueeze(1), safe_2d.unsqueeze(1), cfg=_cfg())

    x = _axis(T, 0, 1.0)
    y_2d = cb_2d(x.clone(), t=torch.tensor(0), step_idx=0)
    assert y_2d.shape == (1, T, D) and torch.isfinite(y_2d).all()
    assert torch.allclose(y_2d, cb_3d(x.clone(), t=torch.tensor(0), step_idx=0), atol=1e-5)
    assert float(y_2d[..., 0].abs().max()) < 1.0


def test_references_of_another_length_are_resampled():
    """Per-frame references built at T0 filter an estimate of length T != T0."""
    T0, T = 179, 180
    unsafe = torch.zeros(2, 1, T0, D)
    unsafe[0, ..., 0] = 1.0
    unsafe[1, ..., 1] = 1.0
    safe = torch.zeros(1, T0, D)
    cb = make_step_callback(unsafe, safe, cfg=_cfg())

    y = cb(_axis(T, 0, 1.0), t=torch.tensor(0), step_idx=0)
    assert y.shape == (1, T, D) and torch.isfinite(y).all()
    assert float(y[..., 0].abs().max()) < 1.0
    x_ok = _axis(T, 0, -1.0)
    assert torch.allclose(cb(x_ok.clone(), t=torch.tensor(0), step_idx=0), x_ok, atol=1e-5)


def test_resample_time_endpoints_and_constant():
    T0, T = 179, 180
    const = torch.randn(1, 1, D).expand(1, T0, D)
    assert torch.allclose(resample_time(const, T), const[:, :1].expand(1, T, D), atol=1e-6)

    ramp = torch.linspace(0.0, 1.0, T0).unsqueeze(-1).expand(T0, D).unsqueeze(0)
    out = resample_time(ramp, T)
    assert torch.allclose(out[0, 0], ramp[0, 0], atol=1e-6)
    assert torch.allclose(out[0, -1], ramp[0, -1], atol=1e-6)
    assert (out[0, 1:, 0] - out[0, :-1, 0]).min() > 0

    assert resample_time(ramp, T0) is ramp
    single = torch.randn(1, 1, D)
    assert resample_time(single, T) is single


# ------------------------------------------------------------------ model routing
class _Scheduler:
    def __init__(self):
        self.timesteps = torch.arange(2)

    def set_timesteps(self, n, device=None):
        pass

    def step(self, predict, t, sample):
        return SimpleNamespace(prev_sample=predict)


class _Net:
    """Always predicts a clip aligned with feature 0."""

    input_feats = D
    cond_mask_prob = 0

    def encode_text(self, caption, device):
        return torch.zeros(len(caption), 4, 8)

    def __call__(self, sample, t, enc_text=None, lengths=None):
        out = torch.zeros_like(sample)
        out[..., 0] = 1.0
        return out

    def eval(self):
        return self


class _Pipeline:
    """Usable by both ECHO's native `generate` and `filtered_generate_batch`."""

    def __init__(self):
        self.model = _Net()
        self.scheduler = _Scheduler()
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.num_inference_steps = 2
        self.native_calls = 0

    def generate(self, caption, m_lens, batch_size=32, timing=False):
        self.native_calls += 1
        out = []
        for length in m_lens:
            clip = torch.zeros(int(length), D)
            clip[..., 0] = 1.0
            out.append(clip)
        return out


def _model(pipeline):
    from csf.backbones.echo.model import EchoModel

    return EchoModel(pipeline=pipeline, mean=np.zeros(D, np.float32), std=np.ones(D, np.float32),
                     motion_rep=SimpleNamespace(fps=50), skeleton=None, text_encoder=None,
                     device="cpu")


def _refs(sign=1.0):
    unsafe = torch.zeros(1, 1, D)
    unsafe[..., 0] = sign
    return unsafe, torch.zeros(1, 1, D)


def test_generate_uses_the_native_path_without_a_context():
    pipe = _Pipeline()
    out = _model(pipe)._generate(["walk"], max_frames=4, num_denoising_steps=2)
    assert out.shape == (1, 4, D)
    assert pipe.native_calls == 1


def test_generate_is_filtered_when_a_context_is_set():
    pipe = _Pipeline()
    model = _model(pipe)
    model.set_filter_context(*_refs(), _cfg())
    out = model._generate(["punch"], max_frames=4, num_denoising_steps=2)
    assert out.shape == (1, 4, D)
    assert pipe.native_calls == 0
    assert float(out[..., 0].abs().max()) < 1.0


def test_filtered_generation_honours_the_step_count():
    pipe = _Pipeline()
    model = _model(pipe)
    model.set_filter_context(*_refs(), _cfg())
    model._generate(["punch"], max_frames=4, num_denoising_steps=7)
    assert pipe.num_inference_steps == 7


def test_slack_context_leaves_the_estimate_unchanged():
    pipe = _Pipeline()
    model = _model(pipe)
    model.set_filter_context(*_refs(sign=-1.0), _cfg())
    out = model._generate(["walk"], max_frames=5, num_denoising_steps=2)
    assert out.shape == (1, 5, D)
    assert pipe.native_calls == 0
    assert torch.allclose(out[..., 0], torch.ones(1, 5))


def test_clear_and_none_context_restore_native_sampling():
    pipe = _Pipeline()
    model = _model(pipe)
    model.set_filter_context(*_refs(), _cfg())
    model.clear_filter_context()
    model._generate(["walk"], max_frames=4, num_denoising_steps=2)
    model.set_filter_context(None, None, _cfg())
    model._generate(["walk"], max_frames=4, num_denoising_steps=2)
    assert pipe.native_calls == 2
