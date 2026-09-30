# SPDX-License-Identifier: Apache-2.0
"""The per-step filter for ECHO's output estimate (Eq. 3 in the 38-d feature).

ECHO predicts the clean motion x̂₀ at every denoising step in its normalised
38-d feature space, and the unsafe and safe references are ECHO's own
unfiltered predictions for the rule and safe-reference texts in that same
space (`EchoModel.reference_feat`).  `make_step_callback` returns the hook
`filtered_generate_batch` calls on each x̂₀: it broadcasts the references to
the estimate's shape and applies `filter_correction`.

The 38-d layout is joints first (29 hinge angles, then root velocity, height
and 6-d orientation), so no leading block is explicit root motion and the
whole vector is filtered (``root_dims = 0``, Table I).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from csf.filter.qp import filter_correction


def resample_time(x: Tensor, T: int) -> Tensor:
    """Linearly resample a per-frame reference ``[..., T0, D]`` to T frames (endpoints kept).

    A reference built at a different length than the estimate it filters (a
    frame of rounding, or a segment of another duration) is a trajectory, so it
    is time-scaled rather than truncated or averaged.  Singleton and matching
    time axes pass through unchanged.
    """
    T0 = x.shape[-2]
    if T0 == T or T0 == 1:
        return x
    D = x.shape[-1]
    flat = x.reshape(-1, T0, D).permute(0, 2, 1)                       # [N, D, T0]
    out = torch.nn.functional.interpolate(flat, size=T, mode="linear", align_corners=True)
    return out.permute(0, 2, 1).reshape(*x.shape[:-2], T, D)


def broadcast_unsafe(unsafe: Tensor, B: int, T: int) -> Tensor:
    """Unsafe references as ``[K, B, T, D]``.

    Accepts ``[K, D]`` (one vector per rule), ``[K, 1, D]`` or ``[K, B, T, D]``;
    singleton batch and time axes are expanded and a mismatched time axis is
    resampled.
    """
    u = unsafe
    D = u.shape[-1]
    if u.dim() == 2:
        u = u[:, None, None, :]
    elif u.dim() == 3:
        u = u.unsqueeze(2)
    elif u.dim() != 4:
        raise ValueError(f"unsafe references must be 2-, 3- or 4-D, got {tuple(unsafe.shape)}")
    if u.shape[2] not in (1, T):
        u = resample_time(u, T)
    return u.expand(u.shape[0], B, T, D).contiguous()


def broadcast_safe(safe: Tensor, B: int, T: int) -> Tensor:
    """The safe reference as ``[B, T, D]``.

    Accepts ``[1, D]`` or ``[B, T, D]``; singleton axes are expanded and a
    mismatched time axis is resampled.
    """
    s = safe
    D = s.shape[-1]
    if s.dim() == 2:
        s = s.unsqueeze(1)
    if s.dim() != 3:
        raise ValueError(f"safe reference must be 2- or 3-D, got {tuple(safe.shape)}")
    if s.shape[1] not in (1, T):
        s = resample_time(s, T)
    return s.expand(B, T, D).contiguous()


def make_step_callback(unsafe: Tensor, safe: Tensor, cfg,
                       active_mask: Optional[Sequence[bool]] = None
                       ) -> Callable[[Tensor, Tensor, int], Tensor]:
    """The hook that filters each x̂₀ of ECHO's denoising loop.

    Args:
        unsafe: Unsafe references, one per rule (see `broadcast_unsafe`).
        safe: The safe reference (see `broadcast_safe`).
        cfg: The `FilterConfig` in force (gamma, rho, ridge, root_dims, ...).
        active_mask: The gate's rule selection; None enforces every rule.

    Returns:
        ``callback(x0_hat [B, T, 38], t, step_idx) -> x̂₀_filt``, on the
        estimate's device and dtype.
    """
    def callback(x0_hat: Tensor, t: Tensor, step_idx: int) -> Tensor:  # noqa: ARG001
        B, T, _ = x0_hat.shape
        x_safe = broadcast_safe(safe, B, T).to(x0_hat.device, x0_hat.dtype)
        x_unsafe = broadcast_unsafe(unsafe, B, T).to(x0_hat.device, x0_hat.dtype)
        delta, _info = filter_correction(x0_hat, x_safe, cfg, x_unsafe=x_unsafe,
                                         active_mask=active_mask)
        return x0_hat + delta

    return callback
