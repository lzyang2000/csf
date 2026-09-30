# SPDX-License-Identifier: Apache-2.0
"""The decoder-aware latent path (Eqs. 7-8), against linear stub decoders (no VAE, no GPU).

With a linear decoder the decoded margin is affine in the latent, so the
iteration of Eq. 8 must reach the decoded safe set.  A stub VAE whose decoded
length depends on the latent length exercises the full-resolution stage gate.
"""
import pytest
import torch

from csf.backbones.motionhiflow.decoded_margin import (
    POSE_SLICE,
    decoded_margins,
    decoder_aware_filter,
    make_decoded_step_callback,
)
from csf.filter.config import FilterConfig


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _linear_decoder(N, P):
    W = torch.randn(P, N)
    return lambda z: W @ z.reshape(-1)


def _directions(dec, refs, d_safe):
    return torch.stack([(dec(r) - d_safe) / (dec(r) - d_safe).norm() for r in refs])


def test_iteration_reaches_the_decoded_safe_set():
    dec = _linear_decoder(96, 40)
    z_safe = torch.randn(1, 1, 6, 16)
    d_safe = dec(z_safe)
    refs = [torch.randn(1, 1, 6, 16) for _ in range(3)]
    dirs = _directions(dec, refs, d_safe)
    z0 = refs[0].clone()                                   # at rule 0's unsafe reference
    assert float(decoded_margins(dec(z0), d_safe, dirs).min()) < 0.0
    z = decoder_aware_filter(dec, z0, dirs, d_safe, iters=30, step=1.0)
    assert float(decoded_margins(dec(z), d_safe, dirs).min()) >= -1e-3


def test_one_full_step_meets_a_single_affine_barrier():
    """eta = 1 on an affine margin is an exact Newton step onto m~ = 0 (Eq. 8)."""
    dec = _linear_decoder(96, 30)
    z_safe = torch.randn(1, 1, 6, 16)
    d_safe = dec(z_safe)
    dirs = _directions(dec, [torch.randn(1, 1, 6, 16)], d_safe)
    z0 = z_safe + 2.0 * (torch.randn(1, 1, 6, 16))
    if float(decoded_margins(dec(z0), d_safe, dirs)[0]) >= 0:
        z0 = 2 * z_safe - z0                               # reflect onto the unsafe side
    z = decoder_aware_filter(dec, z0, dirs, d_safe, iters=1, step=1.0)
    assert float(decoded_margins(dec(z), d_safe, dirs)[0]) == pytest.approx(0.0, abs=1e-3)


def test_no_active_rule_returns_the_latent_unchanged():
    dec = _linear_decoder(96, 20)
    z_safe = torch.randn(1, 1, 6, 16)
    dirs = _directions(dec, [torch.randn(1, 1, 6, 16)], dec(z_safe))
    z0 = torch.randn(1, 1, 6, 16)
    assert torch.equal(decoder_aware_filter(dec, z0, dirs, dec(z_safe), active=[False], iters=10), z0)


def test_a_satisfied_barrier_is_left_alone():
    dec = _linear_decoder(96, 20)
    z_safe = torch.randn(1, 1, 6, 16)
    d_safe = dec(z_safe)
    dirs = _directions(dec, [torch.randn(1, 1, 6, 16)], d_safe)
    assert torch.allclose(decoder_aware_filter(dec, z_safe.clone(), dirs, d_safe, iters=10), z_safe)


def test_filter_runs_inside_no_grad():
    """The sampler calls the hook under no_grad; the decoder gradient must still be available."""
    dec = _linear_decoder(96, 20)
    z_safe = torch.randn(1, 1, 6, 16)
    d_safe = dec(z_safe)
    refs = [torch.randn(1, 1, 6, 16)]
    dirs = _directions(dec, refs, d_safe)
    with torch.no_grad():
        z = decoder_aware_filter(dec, refs[0].clone(), dirs, d_safe, iters=20, step=1.0)
    assert float(decoded_margins(dec(z), d_safe, dirs).min()) >= -1e-3
    assert not z.requires_grad


class _StubVAE:
    """decode: [1, T, 6, 16] -> [1, 4T, 263], linear in z, so decoded length tracks latent length."""

    UPS = 4

    def __init__(self):
        self._W = torch.randn(96, 263)

    def decode(self, z4):
        B, T, J, D = z4.shape
        return (z4.reshape(B, T, J * D) @ self._W).repeat_interleave(self.UPS, dim=1)


def test_callback_filters_the_reference_length_and_passes_other_stages_through():
    vae = _StubVAE()
    T = 3
    safe = torch.randn(1, T, 6, 16)
    unsafe = torch.stack([torch.randn(1, T, 6, 16) for _ in range(2)])      # [2, 1, T, 6, 16]
    cfg = FilterConfig(decoder_aware=True, decoder_iters=30, decoder_step=1.0)
    cb = make_decoded_step_callback(vae, unsafe, safe, cfg)

    def min_margin(z4):
        d_safe = vae.decode(safe)[..., POSE_SLICE].reshape(-1)
        dirs = []
        for k in range(unsafe.shape[0]):
            d = vae.decode(unsafe[k])[..., POSE_SLICE].reshape(-1) - d_safe
            dirs.append(d / d.norm())
        return float(decoded_margins(vae.decode(z4)[..., POSE_SLICE].reshape(-1), d_safe,
                                     torch.stack(dirs)).min())

    x = unsafe[0].clone()
    before = min_margin(x)
    out = cb(x, None, 0, 0)
    assert before < 0.0 and min_margin(out) >= -1e-3
    assert out.shape == x.shape

    x_small = torch.randn(1, 1, 6, 16)                     # an earlier, coarser pyramid stage
    assert torch.equal(cb(x_small, None, 0, 0), x_small)


def test_callback_honours_the_gate_and_the_iteration_budget():
    vae = _StubVAE()
    safe = torch.randn(1, 3, 6, 16)
    unsafe = torch.stack([torch.randn(1, 3, 6, 16)])
    x = unsafe[0].clone()
    off = make_decoded_step_callback(vae, unsafe, safe, FilterConfig(decoder_aware=True),
                                     active_mask=[False])
    assert torch.equal(off(x, None, 0, 0), x)
    frozen = make_decoded_step_callback(vae, unsafe, safe,
                                        FilterConfig(decoder_aware=True, decoder_iters=0))
    assert torch.allclose(frozen(x, None, 0, 0), x)
