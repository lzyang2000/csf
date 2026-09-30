# SPDX-License-Identifier: Apache-2.0
"""MotionHiFlow's Euler sampler with the clean-latent hook (mock sampler, no MotionHiFlow import)."""
from __future__ import annotations

import pytest
import torch

from csf.backbones.motionhiflow.filtered_sampler import (
    clean_latent_from_velocity,
    filtered_flow_generate,
    velocity_from_clean_latent,
)


# ------------------------------------------------------------ velocity <-> x̂₀
def test_clean_latent_recovery_roundtrip():
    """x_t = sigma * noise + (1 - sigma) * data and v = data - noise give x̂₀ = data."""
    torch.manual_seed(0)
    data = torch.randn(2, 5, 6, 16)
    noise = torch.randn_like(data)
    sigma = 0.4
    x_t = sigma * noise + (1 - sigma) * data
    assert torch.allclose(clean_latent_from_velocity(x_t, sigma, data - noise), data, atol=1e-5)


def test_velocity_roundtrip():
    torch.manual_seed(1)
    x_t = torch.randn(3, 4, 8)
    v = torch.randn_like(x_t)
    x0 = clean_latent_from_velocity(x_t, 0.7, v)
    assert torch.allclose(velocity_from_clean_latent(x_t, 0.7, x0), v, atol=1e-5)


def test_recovery_accepts_tensor_sigma():
    torch.manual_seed(2)
    data = torch.randn(2, 3, 4)
    noise = torch.randn_like(data)
    sigma = torch.tensor(0.25)
    x_t = sigma * noise + (1 - sigma) * data
    assert torch.allclose(clean_latent_from_velocity(x_t, sigma, data - noise), data, atol=1e-5)


def test_velocity_at_sigma_zero_is_finite():
    x_t = torch.randn(2, 3)
    assert torch.isfinite(velocity_from_clean_latent(x_t, 0.0, torch.randn_like(x_t))).all()
    assert torch.isfinite(velocity_from_clean_latent(x_t, torch.tensor(0.0), torch.randn_like(x_t))).all()


# ------------------------------------------------------------ single-stage loop
class _MockScheduler:
    """One stage whose sigmas fall from 1 to 0 over len(sigmas) - 1 Euler steps."""

    def __init__(self, sigmas):
        self.config = type("cfg", (), {"stages": 1})()
        self.start_sigmas = [sigmas[0]]
        self.end_sigmas = [sigmas[-1]]
        self._sigmas = torch.tensor(sigmas)
        self.sigmas = self._sigmas
        self.timesteps = torch.arange(len(sigmas) - 1)

    def set_timesteps(self, n, stage_id):
        self.sigmas = self._sigmas
        self.timesteps = torch.arange(len(self._sigmas) - 1)


class _MockSampler:
    """FlowModel fields for a single-stage loop; forward returns a constant velocity."""

    def __init__(self, sigmas, velocity, J=1, D=8):
        self.scheduler = _MockScheduler(sigmas)
        self.scales = [1]
        self.device = "cpu"
        self.joints_num = J
        self.vae_dim = D
        self.short_cut_emb = None
        self.interpolation_type = "half-diff"
        self.interpolation_mode = "linear"
        self._velocity = velocity

    def forward(self, input_latents, timestep, input_text, len_mask=None, scale_id=None, d_cond=None):
        return torch.full_like(input_latents, self._velocity)


def _append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def _ones_mask(lengths, max_length=None):
    return torch.ones(len(lengths), max_length or int(lengths.max()), dtype=torch.bool)


HELPERS = dict(append_dims=_append_dims, lengths_to_mask=_ones_mask,
               interpolate=lambda x, scale_factor, type=None, mode=None: x)


def _run(sampler, T=5, **kw):
    return filtered_flow_generate(sampler, ["walk"], torch.tensor([T]), cond_scale=1.0,
                                  helpers=HELPERS, **kw)


def test_callback_invoked_once_per_step():
    seen = []

    def cb(x0_hat, sigma, stage_id, step_idx):
        seen.append((float(sigma), stage_id, step_idx))
        return x0_hat

    _run(_MockSampler([1.0, 0.5, 0.0], 0.3), step_callback=cb)
    assert [s[2] for s in seen] == [0, 1] and all(s[1] == 0 for s in seen)
    assert seen[0][0] == pytest.approx(1.0) and seen[1][0] == pytest.approx(0.5)


def test_identity_callback_matches_the_native_loop():
    sigmas = [1.0, 0.6, 0.2, 0.0]
    torch.manual_seed(7)
    native, _ = _run(_MockSampler(sigmas, 0.7))
    torch.manual_seed(7)
    hooked, _ = _run(_MockSampler(sigmas, 0.7), step_callback=lambda x, *a: x)
    assert torch.allclose(native, hooked, atol=1e-6)


def test_modified_callback_changes_the_trajectory():
    sigmas = [1.0, 0.5, 0.0]
    torch.manual_seed(8)
    native, _ = _run(_MockSampler(sigmas, 0.4))
    torch.manual_seed(8)
    shifted, _ = _run(_MockSampler(sigmas, 0.4), step_callback=lambda x, *a: x + 1.0)
    assert not torch.allclose(native, shifted, atol=1e-4)


def test_output_shape_and_minimum_length():
    torch.manual_seed(9)
    out, lengths = _run(_MockSampler([1.0, 0.4, 0.0], 0.5), T=3)
    assert torch.isfinite(out).all()
    assert out.shape[0] == 1 and out.shape[1] == 5          # lengths clamp to >= 5 tokens
    assert int(lengths[0]) == 5


def test_sde_path_is_not_reproduced():
    with pytest.raises(NotImplementedError):
        _run(_MockSampler([1.0, 0.0], 0.1), use_sde=True)
