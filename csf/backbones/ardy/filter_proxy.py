# SPDX-License-Identifier: Apache-2.0
"""Per-window filtering of ARDY's clean hybrid-token prediction (Eq. 9).

ARDY generates autoregressively.  Window j denoises ``gen_horizon_len /
num_frames_per_token`` fresh tokens conditioned on the committed history H_j;
at every denoising step ``Ardy.denoising_step`` asks ``model.denoiser`` for the
clean token prediction and DDIM-steps it.  `FilteredDenoiser` takes the place
of ``model.denoiser`` and filters the generation-token block of that
prediction, the uncommitted window estimate x̂_j, with
`csf.filter.qp.filter_correction`.  ARDY then commits the finished window with
its native token requantization (``Ardy._recenter_history(..., requantize=
True)``), so H_{j+1} = H_j ⊕ q(x̂_j,filt) with no change to ARDY itself.

Filter space: the hybrid token, per token the explicit normalized root channels
(20 on ARDY-G1) followed by the FSQ body-latent embedding (128).  The root
channels are excluded from the margin and the correction
(`FilterConfig.root_dims`), so the root trajectory is left to the generator.
This also makes the comparison independent of ARDY's per-window recentering of
the root.

Window alignment: the unsafe and safe references are whole-rollout token
sequences [T_tok, D] from `ArdyModel.reference_feat` at the same frame count,
so window j is filtered against reference tokens [j*n, (j+1)*n), n the number
of generation tokens per window.  A new window is detected when the diffusion
timestep jumps back up (each window runs its own high-noise-to-clean loop).  If
the rollout outruns the references, the last reference token is repeated.

The proxy is an `nn.Module` because ``Ardy`` registers ``denoiser`` as a child
module and `nn.Module.__setattr__` rejects anything else.  Attribute reads the
sampler performs on the denoiser (``latent_embedding_dim``,
``nframe_root_dim``) fall through to the wrapped denoiser.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from csf.filter.qp import filter_correction


def reference_window(ref: Tensor, offset: int, n: int) -> Tensor:
    """Tokens ``[offset, offset + n)`` of a reference ``[..., T_tok, D]``.

    Past the end of the reference the final token is repeated: the reference
    is a motion trajectory and its terminal pose is the natural constant
    extrapolation.
    """
    T = ref.shape[-2]
    start = min(max(offset, 0), max(T - 1, 0))
    block = ref[..., start:min(start + n, T), :]
    if block.shape[-2] < n:
        pad = block[..., -1:, :].expand(*block.shape[:-2], n - block.shape[-2], block.shape[-1])
        block = torch.cat([block, pad], dim=-2)
    return block


class FilteredDenoiser(nn.Module):
    """ARDY's CFG denoiser with the CSF filter on each window's clean tokens.

    Args:
        inner: The original ``model.denoiser`` (anything with the 18-argument
            call convention of ``Ardy.denoising_step``).
        unsafe: Unsafe references in hybrid-token space, [K, T_tok, D] (the
            handle's [K, 1, T_tok, D] layout is accepted).
        safe: Safe reference, [T_tok, D] (or [1, T_tok, D]).
        cfg: The `FilterConfig` in force (gamma, rho, ridge, max_iters,
            root_dims).
        active_mask: bool[K] from the context gate; None enforces every rule.
            An all-False mask leaves the prediction untouched.

    Attributes:
        audit: One record per filtered step of the current rollout:
            ``{"window", "t", "active", "delta_norm"}``.
    """

    def __init__(
        self,
        inner: nn.Module,
        unsafe: Tensor,
        safe: Tensor,
        cfg,
        active_mask: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        if unsafe.dim() == 4 and unsafe.shape[1] == 1:
            unsafe = unsafe.squeeze(1)
        if safe.dim() == 3 and safe.shape[0] == 1:
            safe = safe.squeeze(0)
        if unsafe.dim() != 3 or safe.dim() != 2:
            raise ValueError(
                f"expected unsafe references [K, T, D] and safe reference [T, D]; "
                f"got {tuple(unsafe.shape)} and {tuple(safe.shape)}"
            )
        self._unsafe = unsafe
        self._safe = safe
        self._cfg = cfg
        self._active_mask = active_mask
        self._window = -1
        self._prev_t: Optional[float] = None
        self.audit: list[dict] = []

    def reset(self) -> None:
        """Start a fresh rollout: rewind the window counter."""
        self._window = -1
        self._prev_t = None
        self.audit = []

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(
        self,
        cfg_weight_text,
        cfg_weight_cstr,
        x,
        history_len,
        generation_len,
        future_len,
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        text_feat,
        text_pad_mask,
        t_map,
        first_heading_angle,
        motion_mask,
        observed_motion,
    ) -> Tensor:
        clean = self.inner(
            cfg_weight_text, cfg_weight_cstr, x,
            history_len, generation_len, future_len,
            history_mask, generation_mask, future_mask,
            history_token_mask, generation_token_mask, future_token_mask,
            text_feat, text_pad_mask, t_map, first_heading_angle,
            motion_mask, observed_motion,
        )

        # Each window denoises from high noise to clean, so a timestep that
        # jumps back up starts the next window.
        t_now = float(t_map.max())
        if self._prev_t is None or t_now > self._prev_t:
            self._window += 1
        self._prev_t = t_now

        B, _, D = clean.shape
        n_gen = int(generation_token_mask[0].sum())
        if n_gen == 0:
            return clean
        x_hat = clean[generation_token_mask].reshape(B, n_gen, D)

        offset = self._window * n_gen
        safe = reference_window(self._safe, offset, n_gen).to(clean.device, clean.dtype)
        unsafe = reference_window(self._unsafe, offset, n_gen).to(clean.device, clean.dtype)
        x_safe = safe.unsqueeze(0).expand(B, n_gen, D)
        x_unsafe = unsafe.unsqueeze(1).expand(-1, B, n_gen, D)

        delta, info = filter_correction(
            x_hat, x_safe, self._cfg, x_unsafe=x_unsafe, active_mask=self._active_mask,
        )
        self.audit.append({
            "window": self._window,
            "t": t_now,
            "active": bool(info.get("active", False)),
            "delta_norm": float(delta.norm()),
        })
        if float(delta.abs().max()) == 0.0:
            return clean

        # Only the generation tokens change; the committed history is untouched.
        out = clean.clone()
        out[generation_token_mask] = (x_hat + delta).reshape(-1, D)
        return out
