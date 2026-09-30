# SPDX-License-Identifier: Apache-2.0
"""`FilteredDenoiser`: the filter on ARDY's per-window clean token prediction.

GPU-free, on synthetic hybrid tokens.  Sign convention of the margin (Eq. 1):
an estimate aligned with an unsafe reference, relative to the safe reference,
has h < 0 and is corrected; one on the safe side has h > 0 and, with gamma = 0,
passes through bit-identical.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from csf.backbones.ardy.filter_proxy import FilteredDenoiser, reference_window  # noqa: E402
from csf.filter.config import FilterConfig  # noqa: E402

D = 6          # hybrid token dim (synthetic)
NGEN = 3       # generation tokens per window
SEQ = 6        # tokens fed to the denoiser: 3 history + 3 generation


def _cfg(**kw) -> FilterConfig:
    return FilterConfig(**{"gamma": 0.0, "rho": 1.0, "root_dims": 0, **kw})


class _Denoiser(nn.Module):
    """Stand-in for ARDY's CFG denoiser: returns a preset clean sequence."""

    latent_embedding_dim = 4
    nframe_root_dim = 2

    def __init__(self):
        super().__init__()
        self.next_clean = None

    def forward(self, *args):
        return self.next_clean.clone()


def _call(proxy, clean, t):
    """Call the proxy with only the arguments it reads populated."""
    proxy.inner.next_clean = clean
    generation_token_mask = torch.zeros(1, SEQ, dtype=torch.bool)
    generation_token_mask[0, SEQ - NGEN:] = True
    args = [None] * 18
    args[2] = clean
    args[10] = generation_token_mask
    args[14] = torch.tensor([t])
    return proxy(*args)


def _references():
    """Unsafe references [K=2, T_tok=6, D] along e0 / e1, safe reference zeros."""
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0
    unsafe = torch.stack([e0, e1]).unsqueeze(1).expand(2, 2 * NGEN, D).contiguous()
    safe = torch.zeros(2 * NGEN, D)
    return unsafe, safe


def _margin(x, unsafe, safe):
    v = (x - safe).reshape(-1)
    d = (unsafe - safe).reshape(-1)
    return float(-(v @ d) / (d.norm() + 1e-9))


def test_filter_lifts_the_margin_of_unsafe_aligned_tokens():
    unsafe, safe = _references()
    proxy = FilteredDenoiser(_Denoiser(), unsafe, safe, _cfg())
    clean = torch.zeros(1, SEQ, D)
    clean[0, 3:, 0] = 2.0                      # generation tokens along rule 0 => h < 0
    out = _call(proxy, clean, t=9)

    before = _margin(clean[0, 3:], unsafe[0, :NGEN], safe[:NGEN])
    after = _margin(out[0, 3:], unsafe[0, :NGEN], safe[:NGEN])
    assert before < -1.0
    assert after > -0.1, f"rho=1 should restore h >= 0 up to the ridge residual, got {after}"
    assert torch.equal(out[0, :3], clean[0, :3]), "committed history must not change"
    assert proxy.audit and proxy.audit[-1]["active"]


def test_benign_tokens_pass_bit_identical():
    unsafe, safe = _references()
    proxy = FilteredDenoiser(_Denoiser(), unsafe, safe, _cfg())
    clean = torch.zeros(1, SEQ, D)
    clean[0, 3:, 0] = -2.0                     # opposite side of every rule => h > 0
    assert torch.equal(_call(proxy, clean, t=9), clean)


def test_an_all_inactive_gate_is_a_no_op():
    unsafe, safe = _references()
    proxy = FilteredDenoiser(
        _Denoiser(), unsafe, safe, _cfg(gamma=1.0), active_mask=[False, False],
    )
    clean = torch.zeros(1, SEQ, D)
    clean[0, 3:, 0] = 2.0                      # unsafe-aligned, but the gate selected nothing
    assert torch.equal(_call(proxy, clean, t=9), clean)


def test_each_window_is_filtered_against_its_own_reference_block():
    """gamma = rho = 1 moves the estimate onto the safe reference (the barrier is
    slack there), so the output reveals which block the proxy sliced."""
    unsafe, _ = _references()
    safe = torch.zeros(2 * NGEN, D)
    safe[:NGEN, 2] = 1.0                       # window 0 block
    safe[NGEN:, 3] = 5.0                       # window 1 block
    proxy = FilteredDenoiser(_Denoiser(), unsafe, safe, _cfg(gamma=1.0))
    clean = torch.randn(1, SEQ, D)

    def close(a, b):
        return torch.allclose(a, b, atol=1e-5)

    assert close(_call(proxy, clean, t=9)[0, 3:], safe[:NGEN])
    assert close(_call(proxy, clean, t=0)[0, 3:], safe[:NGEN])    # same window, later step
    assert close(_call(proxy, clean, t=9)[0, 3:], safe[NGEN:])    # t jumps up: next window
    proxy.reset()
    assert close(_call(proxy, clean, t=9)[0, 3:], safe[:NGEN])    # reset rewinds to window 0


def test_root_channels_are_left_to_the_generator():
    unsafe, _ = _references()
    safe = torch.zeros(2 * NGEN, D)
    safe[:, 2:] = 1.0
    proxy = FilteredDenoiser(_Denoiser(), unsafe, safe, _cfg(gamma=1.0, root_dims=2))
    clean = torch.randn(1, SEQ, D)
    out = _call(proxy, clean, t=9)
    assert torch.equal(out[0, 3:, :2], clean[0, 3:, :2]), "root channels were edited"
    assert torch.allclose(out[0, 3:, 2:], safe[:NGEN, 2:], atol=1e-5)


def test_reference_window_repeats_the_last_token_past_the_end():
    ref = torch.arange(4, dtype=torch.float32).unsqueeze(-1).expand(4, D).contiguous()
    block = reference_window(ref, 2, 3)        # tokens 2, 3, then 3 again
    assert block.shape == (3, D)
    assert block[:, 0].tolist() == [2.0, 3.0, 3.0]
    assert reference_window(ref, 10, 2)[:, 0].tolist() == [3.0, 3.0]


def test_sampler_attributes_fall_through_to_the_wrapped_denoiser():
    unsafe, safe = _references()
    proxy = FilteredDenoiser(_Denoiser(), unsafe, safe, _cfg())
    assert proxy.latent_embedding_dim == 4
    assert proxy.nframe_root_dim == 2


def test_the_handles_reference_layout_is_accepted():
    """`AdapterFilterHandle` passes unsafe [K, 1, T, D] and safe [1, T, D]."""
    unsafe, safe = _references()
    proxy = FilteredDenoiser(_Denoiser(), unsafe.unsqueeze(1), safe.unsqueeze(0), _cfg())
    assert proxy._unsafe.shape == (2, 2 * NGEN, D)
    assert proxy._safe.shape == (2 * NGEN, D)


def test_malformed_references_are_rejected():
    unsafe, safe = _references()
    with pytest.raises(ValueError, match="expected unsafe references"):
        FilteredDenoiser(_Denoiser(), unsafe[0], safe, _cfg())
