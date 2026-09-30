# SPDX-License-Identifier: Apache-2.0
"""The per-step filter for MotionHiFlow's clean latent (Eq. 3 in the VAE latent).

At every Euler step `filtered_flow_generate` hands the hook the clean latent
x̂₀ as ``[B, T, J, D]`` (J = 6 joint groups, D = 16 channels; T is the latent
length of the current pyramid stage).  The unsafe and safe references are
MotionHiFlow's own unfiltered clean latents for the rule and safe-reference
texts (`MHFModel.reference_feat`).  Each frame's ``[J, D]`` tile is
flattened to a 96-d vector, `filter_correction` is applied, and the result is
reshaped back.  The latent has no separable root block, so the whole vector
is filtered (``root_dims = 0``, Table I).

This is the benchmark configuration (Sec. III-E, gamma = 0.5).  The margin
lives in the latent, so its effect on the decoded motion is mediated by the
decoder; `decoded_margin` implements the decoder-aware alternative.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from torch import Tensor

from csf.filter.qp import filter_correction

#: MotionHiFlow's VAE latent tile per frame: J joint groups x D channels.
LATENT_J, LATENT_D = 6, 16
LATENT_FLAT = LATENT_J * LATENT_D


def flatten_latent(x: Tensor) -> Tensor:
    """``[..., J, D]`` latent tiles to ``[..., J*D]`` (already flat input passes through)."""
    if x.shape[-1] == LATENT_FLAT:
        return x
    if x.dim() >= 2 and x.shape[-2:] == (LATENT_J, LATENT_D):
        return x.reshape(*x.shape[:-2], LATENT_FLAT)
    raise ValueError(f"latent trailing dims must be ({LATENT_J}, {LATENT_D}) or {LATENT_FLAT}, "
                     f"got {tuple(x.shape)}")


def broadcast_unsafe(unsafe: Tensor, B: int, T: int) -> Tensor:
    """Unsafe references as flat ``[K, B, T, 96]``.

    Accepts ``[K, J, D]``, ``[K, 1, J, D]`` or ``[K, B, T, J, D]`` (or their
    flattened forms).  Pyramid stages run at different latent lengths, so a
    time axis matching neither 1 nor the current stage is averaged to one
    frame before broadcasting.
    """
    u = flatten_latent(unsafe)
    if u.dim() == 2:
        u = u[:, None, None, :]
    elif u.dim() == 3:
        u = u.unsqueeze(2)
    elif u.dim() != 4:
        raise ValueError(f"unsafe references must flatten to 2-, 3- or 4-D, got {tuple(u.shape)}")
    if u.shape[2] not in (1, T):
        u = u.mean(dim=2, keepdim=True)
    return u.expand(u.shape[0], B, T, LATENT_FLAT).contiguous()


def broadcast_safe(safe: Tensor, B: int, T: int) -> Tensor:
    """The safe reference as flat ``[B, T, 96]`` (same rules as `broadcast_unsafe`)."""
    s = flatten_latent(safe)
    if s.dim() == 2:
        s = s.unsqueeze(1)
    if s.dim() != 3:
        raise ValueError(f"safe reference must flatten to 2- or 3-D, got {tuple(s.shape)}")
    if s.shape[1] not in (1, T):
        s = s.mean(dim=1, keepdim=True)
    return s.expand(B, T, LATENT_FLAT).contiguous()


def make_latent_step_callback(unsafe: Tensor, safe: Tensor, cfg,
                              active_mask: Optional[Sequence[bool]] = None
                              ) -> Callable[[Tensor, object, int, int], Tensor]:
    """The hook that filters each clean latent of MotionHiFlow's Euler loop.

    Args:
        unsafe: Unsafe latent references, one per rule (see `broadcast_unsafe`).
        safe: The safe latent reference (see `broadcast_safe`).
        cfg: The `FilterConfig` in force (gamma, rho, ridge, root_dims, ...).
        active_mask: The gate's rule selection; None enforces every rule.

    Returns:
        ``callback(x0_hat [B, T, 6, 16], sigma, stage_id, step_idx) -> x̂₀_filt``.
    """
    def callback(x0_hat: Tensor, sigma, stage_id: int, step_idx: int) -> Tensor:  # noqa: ARG001
        x = flatten_latent(x0_hat)                                     # [B, T, 96]
        if x.dim() != 3:
            raise ValueError(f"clean latent must be [B, T, {LATENT_J}, {LATENT_D}] or "
                             f"[B, T, {LATENT_FLAT}], got {tuple(x0_hat.shape)}")
        B, T, _ = x.shape
        x_safe = broadcast_safe(safe, B, T).to(x.device, x.dtype)
        x_unsafe = broadcast_unsafe(unsafe, B, T).to(x.device, x.dtype)
        delta, _info = filter_correction(x, x_safe, cfg, x_unsafe=x_unsafe, active_mask=active_mask)
        return (x + delta).reshape(x0_hat.shape)

    return callback
