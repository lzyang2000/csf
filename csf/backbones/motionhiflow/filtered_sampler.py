# SPDX-License-Identifier: Apache-2.0
"""MotionHiFlow's pyramid Euler sampler with a hook on the clean latent.

MotionHiFlow interpolates ``x_t = sigma * noise + (1 - sigma) * data`` (sigma
falls from 1 at noise to 0 at data) and its DiT predicts the velocity
``v = data - noise`` (`FlowModel.train_forward`).  Substituting gives the
output estimate, the clean latent,

    x0_hat = x_t + sigma * v,          and conversely   v = (x0_hat - x_t) / sigma.

`filtered_flow_generate` reproduces `FlowModel.generate` (tmdit.py) for the
deterministic Euler path: the same noise draw, per-stage renoising and
resolution change, length masks and classifier-free guidance.  At every step
it recovers x̂₀ from the guided velocity, passes it through
``step_callback(x0_hat, sigma, stage_id, step_idx)``, converts the result
back to a velocity and takes the native Euler step.  Without a callback the
guided velocity is used as is, so the loop is MotionHiFlow's own.

The loop takes any FlowModel-like object; the three MotionHiFlow tensor
helpers it needs are imported lazily (or injected, in tests).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

StepCallback = Callable[[Tensor, Any, int, int], Tensor]

# Below this sigma the velocity inverse is singular; the Euler coefficient
# (sigma_i - sigma_{i+1}) is then ~0 too, so the correction is skipped.
_SIGMA_EPS = 1e-6

_HELPERS: Optional[dict] = None


def clean_latent_from_velocity(x_t: Tensor, sigma, v: Tensor) -> Tensor:
    """x̂₀ = x_t + sigma * v, for MotionHiFlow's ``v = data - noise`` convention."""
    return x_t + sigma * v


def velocity_from_clean_latent(x_t: Tensor, sigma, x0_hat: Tensor) -> Tensor:
    """The velocity that moves x_t to x0_hat: ``(x0_hat - x_t) / sigma`` (zero at sigma ~ 0)."""
    if isinstance(sigma, Tensor):
        tiny = sigma.abs() < _SIGMA_EPS
        v = (x0_hat - x_t) / torch.where(tiny, torch.ones_like(sigma), sigma)
        return torch.where(tiny, torch.zeros_like(v), v)
    if abs(sigma) < _SIGMA_EPS:
        return torch.zeros_like(x_t)
    return (x0_hat - x_t) / sigma


def _resolve_helpers(helpers: Optional[dict]) -> dict:
    """``append_dims``, ``lengths_to_mask`` and ``interpolate`` from MotionHiFlow."""
    global _HELPERS
    if helpers is not None:
        return helpers
    if _HELPERS is None:
        from .loader import _ensure_mhf_on_path, _mhf_import_isolation  # noqa: PLC0415

        with _mhf_import_isolation():
            _ensure_mhf_on_path()
            from src.arch.dit.tmdit import append_dims, lengths_to_mask  # type: ignore[import]  # noqa: PLC0415
            from src.utils.interp import interpolate  # type: ignore[import]  # noqa: PLC0415
        _HELPERS = dict(append_dims=append_dims, lengths_to_mask=lengths_to_mask,
                        interpolate=interpolate)
    return _HELPERS


def filtered_flow_generate(sampler, text: list[str], m_lengths: Tensor, time_steps: int = 12,
                           cond_scale: float = 4.5, cfg_interval=(0.0, 1.0), use_sde: bool = False,
                           step_callback: Optional[StepCallback] = None,
                           helpers: Optional[dict] = None):
    """`FlowModel.generate` with the clean latent filtered at every Euler step.

    Args:
        sampler: MotionHiFlow's `FlowModel` (or anything with the same fields).
        text: B prompts.
        m_lengths: ``[B]`` lengths in latent tokens (frames // 4).
        time_steps: Euler step budget, split across the pyramid stages.
        cond_scale: Classifier-free guidance scale (1 disables guidance).
        cfg_interval: Sigma range in which guidance is applied.
        use_sde: Must be False; only the deterministic path is reproduced.
        step_callback: ``(x0_hat, sigma, stage_id, step_idx) -> x0_hat``, or None.
        helpers: Tensor helpers to use instead of MotionHiFlow's (tests).

    Returns:
        ``(latents [B, T, J, D], m_lengths)``, as `FlowModel.generate`.
    """
    if use_sde:
        raise NotImplementedError("filtered_flow_generate reproduces the deterministic Euler path only")

    h = _resolve_helpers(helpers)
    append_dims, lengths_to_mask, interpolate = h["append_dims"], h["lengths_to_mask"], h["interpolate"]
    scheduler = sampler.scheduler
    scales = sampler.scales

    m_lengths = torch.clamp(m_lengths, min=5)
    orig_len = int(m_lengths.max().item())
    m_lens = [(m_lengths * scale).long() for scale in scales]
    lens = [ml.max().item() for ml in m_lens]

    input_text = text
    if cond_scale != 1.0:
        input_text = [""] * len(text) + list(text)

    noise = torch.randn(len(text), orig_len, sampler.joints_num, sampler.vae_dim, device=sampler.device)
    noise_list = [
        interpolate(noise, scale_factor=lens[i] / orig_len + 1e-6,
                    type=sampler.interpolation_type, mode=sampler.interpolation_mode)
        for i in range(len(lens))
    ]
    latents = noise_list[0]
    start_sigmas, end_sigmas = scheduler.start_sigmas, scheduler.end_sigmas

    for stage_id in range(scheduler.config.stages):
        add_noise = 0
        if stage_id > 0:                                         # renoise at the next resolution
            latents = interpolate(
                latents - end_sigmas[stage_id - 1] * noise_list[stage_id - 1],
                scale_factor=lens[stage_id] / lens[stage_id - 1] + 1e-6,
                type=sampler.interpolation_type, mode=sampler.interpolation_mode,
            )
            latents = latents * ((1 - start_sigmas[stage_id]) / (1 - end_sigmas[stage_id - 1]))
            add_noise = start_sigmas[stage_id] * noise_list[stage_id]
        latents = torch.nn.functional.pad(
            latents,
            (0,) * (latents.ndim * 2 - 3) + (noise_list[stage_id].shape[1] - latents.shape[1],),
            mode="constant", value=0,
        )
        len_mask = lengths_to_mask(m_lens[stage_id], max_length=latents.shape[1])
        latents = (latents + add_noise) * append_dims(len_mask, latents.ndim)

        n_steps = int(time_steps * (start_sigmas[stage_id] - end_sigmas[stage_id]) + 0.5)
        scheduler.set_timesteps(max(n_steps, 1), stage_id)
        sigmas = scheduler.sigmas.to(latents.device)
        timesteps = scheduler.timesteps.to(latents.device)

        for i, timestep in enumerate(timesteps):
            if cond_scale != 1.0:
                input_latents = torch.cat([latents] * 2, dim=0)
                input_len_mask = torch.cat([len_mask] * 2, dim=0)
            else:
                input_latents, input_len_mask = latents, len_mask

            with torch.no_grad():
                if sampler.short_cut_emb is not None:
                    pred = sampler.forward(input_latents, timestep, input_text,
                                           d_cond=sigmas[i] - sigmas[i + 1],
                                           len_mask=input_len_mask, scale_id=scales[stage_id])
                else:
                    pred = sampler.forward(input_latents, timestep, input_text,
                                           len_mask=input_len_mask, scale_id=scales[stage_id])
                if cond_scale != 1.0:
                    pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                    pred = pred_uncond + cond_scale * (pred_cond - pred_uncond)
                    if sigmas[i] < cfg_interval[0] or sigmas[i] > cfg_interval[1]:
                        pred = pred_cond

            v = pred
            if step_callback is not None:
                x0_hat = clean_latent_from_velocity(latents, sigmas[i], pred).detach()
                x0_hat = step_callback(x0_hat, sigmas[i], stage_id, i)
                v = velocity_from_clean_latent(latents, sigmas[i], x0_hat)

            latents = latents + (sigmas[i] - sigmas[i + 1]) * v
            latents = latents * append_dims(len_mask, latents.ndim)

    if scales[-1] != 1:
        latents = interpolate(latents, scale_factor=orig_len / lens[-1] + 1e-6,
                              type=sampler.interpolation_type, mode=sampler.interpolation_mode)
    try:
        latents = latents.squeeze(-2)
    except Exception:  # noqa: BLE001 - mirrors FlowModel.generate
        pass
    return latents, m_lengths
