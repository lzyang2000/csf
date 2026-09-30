# SPDX-License-Identifier: Apache-2.0
"""The per-step filter on MotionHiFlow's clean latent, and MHFModel's routing through it.

GPU-free, with synthetic latents (6 joint groups x 16 channels = 96-d per frame).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from csf.backbones.motionhiflow.csf_adapter import (
    broadcast_safe,
    broadcast_unsafe,
    make_latent_step_callback,
)
from csf.filter.config import FilterConfig

J, DLAT = 6, 16
D = J * DLAT


def _cfg(gamma: float = 0.0, **kw) -> FilterConfig:
    return FilterConfig(gamma=gamma, rho=1.0, root_dims=0, **kw)


def _margin(x_flat, unsafe_flat, safe_flat):
    x = x_flat if x_flat.dim() == 3 else x_flat.unsqueeze(0)
    T = x.shape[1]
    d = (unsafe_flat - safe_flat).reshape(1, -1).expand(T, -1).reshape(-1)
    v = (x[0] - safe_flat.reshape(1, -1)).reshape(-1)
    return float(-(v @ d) / (d.norm() + 1e-8))


def _latent(T, j, c, value):
    x = torch.zeros(1, T, J, DLAT)
    x[..., j, c] = value
    return x


# ------------------------------------------------------------------ filter math
def test_latent_filter_corrects_unsafe_and_keeps_safe():
    T = 4
    unsafe = _latent(1, 0, 0, 1.0)
    safe = torch.zeros(1, 1, J, DLAT)
    cb = make_latent_step_callback(unsafe, safe, cfg=_cfg())
    uf, sf = unsafe.reshape(1, 1, D), safe.reshape(1, 1, D)

    x_bad = _latent(T, 0, 0, 1.0)
    before = _margin(x_bad.reshape(1, T, D), uf, sf)
    assert before < 0.0
    y = cb(x_bad.clone(), sigma=0.5, stage_id=0, step_idx=0)
    assert y.shape == (1, T, J, DLAT)
    after = _margin(y.reshape(1, T, D), uf, sf)
    assert after >= -0.1 and after > before + 1.0

    x_ok = _latent(T, 0, 0, -1.0)
    assert torch.allclose(cb(x_ok.clone(), sigma=0.5, stage_id=0, step_idx=0), x_ok, atol=1e-5)


def test_latent_multiple_rules_corrected_jointly():
    T = 3
    unsafe = torch.cat([_latent(1, 0, 0, 1.0), _latent(1, 0, 1, 1.0)], dim=0)   # [2, 1, J, D]
    safe = torch.zeros(1, 1, J, DLAT)
    cb = make_latent_step_callback(unsafe, safe, cfg=_cfg())
    x = _latent(T, 0, 0, 1.0) + _latent(T, 0, 1, 1.0)
    y = cb(x.clone(), sigma=0.5, stage_id=0, step_idx=0)
    for k in range(2):
        assert _margin(y.reshape(1, T, D), unsafe[k].reshape(1, 1, D), safe.reshape(1, 1, D)) >= -0.1


def test_latent_tracking_blends_toward_the_safe_reference():
    """gamma = 0.5 (Table I) on a feasible estimate: halfway to the safe latent."""
    torch.manual_seed(0)
    unsafe = _latent(1, 0, 0, 1.0)
    safe = torch.randn(1, 1, J, DLAT) * 0.1
    safe[..., 0, 0] = -1.0
    cb = make_latent_step_callback(unsafe, safe, cfg=_cfg(gamma=0.5))
    x = _latent(3, 0, 0, -2.0)
    y = cb(x.clone(), sigma=0.5, stage_id=0, step_idx=0)
    assert torch.allclose(y, 0.5 * x + 0.5 * safe.expand_as(x), atol=1e-5)


def test_flattened_references_match_tiled_references():
    T, K = 5, 2
    unsafe_2d = torch.zeros(K, D)
    unsafe_2d[0, 0] = 1.0
    unsafe_2d[1, 1] = 1.0
    safe_2d = torch.zeros(1, D)
    cb_2d = make_latent_step_callback(unsafe_2d, safe_2d, cfg=_cfg())
    cb_4d = make_latent_step_callback(unsafe_2d.reshape(K, 1, J, DLAT),
                                      safe_2d.reshape(1, 1, J, DLAT), cfg=_cfg())
    x = _latent(T, 0, 0, 1.0)
    y_2d = cb_2d(x.clone(), sigma=0.5, stage_id=0, step_idx=0)
    assert torch.allclose(y_2d, cb_4d(x.clone(), sigma=0.5, stage_id=0, step_idx=0), atol=1e-5)
    assert float(y_2d[..., 0, 0].abs().max()) < 1.0


def test_references_broadcast_across_pyramid_stage_lengths():
    """The handle's [K, 1, T, J, D] references are averaged over time at other stage lengths."""
    unsafe = torch.randn(3, 1, 10, J, DLAT)
    safe = torch.randn(1, 10, J, DLAT)
    assert broadcast_unsafe(unsafe, 1, 10).shape == (3, 1, 10, D)
    u5 = broadcast_unsafe(unsafe, 1, 5)
    assert u5.shape == (3, 1, 5, D)
    assert torch.allclose(u5[:, :, 0], unsafe.mean(dim=2).reshape(3, 1, D), atol=1e-6)
    s5 = broadcast_safe(safe, 1, 5)
    assert torch.allclose(s5[:, 0], safe.mean(dim=1).reshape(1, D), atol=1e-6)


# ------------------------------------------------------------------ model routing
class _Scheduler:
    def __init__(self, sigmas):
        self.config = SimpleNamespace(stages=1)
        self.start_sigmas = [sigmas[0]]
        self.end_sigmas = [sigmas[-1]]
        self._sigmas = torch.tensor(sigmas)
        self.sigmas = self._sigmas
        self.timesteps = torch.arange(len(sigmas) - 1)

    def set_timesteps(self, n, stage_id):
        self.sigmas = self._sigmas
        self.timesteps = torch.arange(len(self._sigmas) - 1)


class _Flow:
    """A native `generate` plus the fields `filtered_flow_generate` reads."""

    def __init__(self):
        self.native_calls = 0
        self.scheduler = _Scheduler([1.0, 0.5, 0.0])
        self.scales = [1]
        self.device = "cpu"
        self.joints_num = J
        self.vae_dim = DLAT
        self.short_cut_emb = None
        self.interpolation_type = "half-diff"
        self.interpolation_mode = "linear"

    def generate(self, text, m_lengths, time_steps=12, cond_scale=4.5, **kw):
        self.native_calls += 1
        return torch.zeros(len(text), int(m_lengths.max()), J, DLAT), m_lengths

    def forward(self, input_latents, timestep, input_text, len_mask=None, scale_id=None, d_cond=None):
        out = torch.zeros_like(input_latents)
        out[..., 0, 0] = 2.0                         # a velocity that drives latent (0, 0) up
        return out


class _VAE:
    def decode(self, latent):
        out = torch.zeros(latent.shape[0], 4 * latent.shape[1], 263)
        out[..., 0] = latent[..., 0, 0].abs().sum(dim=1, keepdim=True)
        return out


_HELPERS = dict(
    append_dims=lambda x, n: x[(...,) + (None,) * (n - x.ndim)],
    lengths_to_mask=lambda lengths, max_length=None: torch.ones(
        len(lengths), max_length or int(lengths.max()), dtype=torch.bool),
    interpolate=lambda x, scale_factor, type=None, mode=None: x,
)


def _cpu_helpers(monkeypatch):
    import csf.backbones.motionhiflow.filtered_sampler as sampler

    monkeypatch.setattr(sampler, "_resolve_helpers", lambda helpers=None: _HELPERS)


def _model(flow):
    from csf.backbones.motionhiflow.model import MHFModel

    return MHFModel(vae=_VAE(), flow=flow, inv_transform=lambda a: a,
                    motion_rep=SimpleNamespace(fps=20), skeleton=None, text_encoder=None,
                    device="cpu", cond_scale=1.0, time_steps=2)


def _refs():
    return _latent(1, 0, 0, 1.0).unsqueeze(0), torch.zeros(1, 1, J, DLAT)   # [1,1,1,J,D], [1,1,J,D]


def test_generate_uses_the_native_path_without_a_context():
    flow = _Flow()
    out = _model(flow)._generate(["walk"], max_frames=8, num_denoising_steps=2)
    assert out.shape == (1, 8, 263)
    assert flow.native_calls == 1


def test_generate_is_filtered_when_a_context_is_set(monkeypatch):
    from csf.backbones.motionhiflow.filtered_sampler import filtered_flow_generate

    _cpu_helpers(monkeypatch)
    flow = _Flow()
    model = _model(flow)
    model.set_filter_context(*_refs(), _cfg())
    torch.manual_seed(123)
    out = model._generate(["punch"], max_frames=20, num_denoising_steps=2)
    assert out.shape == (1, 20, 263) and torch.isfinite(out).all()
    assert flow.native_calls == 0

    torch.manual_seed(123)
    native, _ = filtered_flow_generate(_Flow(), ["punch"], torch.tensor([5]), time_steps=2,
                                       cond_scale=1.0, helpers=_HELPERS)
    assert float(out[..., 0].abs().max()) < float(_VAE().decode(native)[..., 0].abs().max())


def test_decoder_aware_flag_selects_the_decoded_path(monkeypatch):
    import csf.backbones.motionhiflow.decoded_margin as decoded

    made = []
    monkeypatch.setattr(decoded, "make_decoded_step_callback",
                        lambda vae, unsafe, safe, cfg, active_mask=None: made.append(cfg) or (lambda x, *a: x))
    model = _model(_Flow())
    model.set_filter_context(*_refs(), _cfg())
    assert not made
    model.set_filter_context(*_refs(), _cfg(decoder_aware=True))
    assert len(made) == 1 and made[0].decoder_aware


def test_clear_and_none_context_restore_native_sampling():
    flow = _Flow()
    model = _model(flow)
    model.set_filter_context(*_refs(), _cfg())
    model.clear_filter_context()
    model._generate(["walk"], max_frames=8, num_denoising_steps=2)
    model.set_filter_context(None, None, _cfg())
    model._generate(["walk"], max_frames=8, num_denoising_steps=2)
    assert flow.native_calls == 2
