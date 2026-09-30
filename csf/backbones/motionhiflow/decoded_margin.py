# SPDX-License-Identifier: Apache-2.0
"""The decoder-aware latent path for MotionHiFlow (Sec. III-E, Eqs. 7-8).

Distances in MotionHiFlow's VAE latent need not separate the declared
concepts, so this path evaluates the same semantic margin after the pretrained
decoder D.  With phi(z) = P D(z) the decoded non-root pose channels
(HumanML3D's local joint positions and rotations, `POSE_SLICE`), Eq. 7 reads

    m~_i(z) = -< phi(z) - phi(z_s), unit(phi(z_i) - phi(z_s)) >,

with z_s and z_i the safe and unsafe latent references.  D is nonlinear, so the
barrier is imposed iteratively (Eq. 8): with k the worst active margin and
q = grad_z m~_k(z) its gradient through the decoder, while m~_k(z) < 0,

    z <- z - eta * m~_k(z) / (||q||^2 + eps) * q,

re-decoding after every update, for at most N iterations (N =
`cfg.decoder_iters`, eta = `cfg.decoder_step`).  The margin compares whole
decoded clips, so on the pyramid sampler the path acts only at the
full-resolution stage, whose decoded length matches the references; earlier
stages pass through unchanged.

The path is off by default (`FilterConfig.decoder_aware`): the benchmark
filters in the latent with gamma = 0.5 (`csf_adapter`).  It is a pure barrier,
so gamma and rho do not enter it.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor

#: Non-root pose channels of HumanML3D's 263-d feature: ric_data (63) + rot_data (126).
POSE_SLICE = slice(4, 193)
_EPS = 1e-9


def decoded_margins(pose: Tensor, d_safe: Tensor, directions: Tensor) -> Tensor:
    """Eq. 7 for every rule: ``-<pose - d_safe, dir_k>``.

    Args:
        pose: Flat decoded pose ``[P]``.
        d_safe: Flat decoded safe reference ``[P]``.
        directions: Unit decoded rule directions ``[K, P]``.

    Returns:
        ``[K]`` margins (negative on the unsafe side).
    """
    return -(directions @ (pose - d_safe))


def decoder_aware_filter(decode_flat: Callable[[Tensor], Tensor], z: Tensor, directions: Tensor,
                         d_safe: Tensor, *, active: Optional[Sequence[bool]] = None,
                         iters: int = 8, step: float = 0.5, eps: float = _EPS) -> Tensor:
    """Iterate Eq. 8 on one latent until every active decoded margin is met.

    Args:
        decode_flat: Differentiable ``latent -> flat decoded pose [P]``.
        z: The clean latent to filter.
        directions: Unit decoded rule directions ``[K, P]``.
        d_safe: Flat decoded safe reference ``[P]``.
        active: The gate's rule selection; None enforces every rule.  With no
            active rule `z` is returned unchanged.
        iters: Iteration budget N.
        step: Step size eta.
        eps: Guard for a vanishing gradient.

    Returns:
        The filtered latent (detached), shaped like `z`.
    """
    rows: List[int] = (list(range(int(directions.shape[0]))) if active is None
                       else [k for k, on in enumerate(active) if on])
    if not rows:
        return z
    z = z.detach().clone()
    for _ in range(int(iters)):
        # The sampler runs under no_grad; the decoder gradient needs it locally.
        with torch.enable_grad():
            zr = z.requires_grad_(True)
            margins = decoded_margins(decode_flat(zr), d_safe, directions)
            violated = [(float(margins[k]), k) for k in rows if float(margins[k]) < 0.0]
            if not violated:
                break
            m_k, k = min(violated)                                   # worst active rule
            (q,) = torch.autograd.grad(margins[k], zr)
        coef = float(step) * m_k / (float(q.reshape(-1) @ q.reshape(-1)) + eps)
        z = (z.detach() - coef * q).detach()
    return z.detach()


def _as_4d(z: Tensor) -> Tensor:
    if z.dim() == 3:                                                 # [T, J, D]
        return z.unsqueeze(0)
    if z.dim() == 4:                                                 # [B, T, J, D]
        return z
    raise ValueError(f"latent must be [T, J, D] or [B, T, J, D], got {tuple(z.shape)}")


def _split_references(unsafe: Tensor) -> List[Tensor]:
    """One ``[1, T, J, D]`` latent per rule from ``[K, 1, T, J, D]`` or ``[K, T, J, D]``."""
    if unsafe.dim() == 5:
        return [unsafe[k] for k in range(unsafe.shape[0])]
    unsafe = _as_4d(unsafe)
    return [unsafe[k:k + 1] for k in range(unsafe.shape[0])]


def make_decoded_step_callback(vae, unsafe: Tensor, safe: Tensor, cfg,
                               active_mask: Optional[Sequence[bool]] = None
                               ) -> Callable[[Tensor, object, int, int], Tensor]:
    """The decoder-aware alternative to `csf_adapter.make_latent_step_callback`.

    The references are decoded once; each call then runs `decoder_aware_filter`
    through ``vae.decode`` on single-sample clean latents whose decoded length
    matches the references, and passes every other estimate through.

    Args:
        vae: MotionHiFlow's VAE (``decode(latent [B, T, J, D]) -> [B, T', 263]``).
        unsafe: Unsafe latent references ``[K, 1, T, J, D]``.
        safe: Safe latent reference ``[1, T, J, D]``.
        cfg: The `FilterConfig` (reads `decoder_iters` and `decoder_step`).
        active_mask: The gate's rule selection; None enforces every rule.
    """
    iters = int(getattr(cfg, "decoder_iters", 8))
    step = float(getattr(cfg, "decoder_step", 0.5))

    def decode_flat(z4: Tensor) -> Tensor:
        return vae.decode(z4)[..., POSE_SLICE].reshape(-1)

    with torch.no_grad():
        d_safe = decode_flat(_as_4d(safe))
        directions = []
        for ref in _split_references(unsafe):
            d = decode_flat(ref) - d_safe
            directions.append(d / (d.norm() + _EPS))
        directions = torch.stack(directions, dim=0)                  # [K, P]
    pose_size = d_safe.shape[0]

    def callback(x0_hat: Tensor, sigma, stage_id: int, step_idx: int) -> Tensor:  # noqa: ARG001
        z4 = _as_4d(x0_hat)
        if z4.shape[0] != 1:
            return x0_hat
        with torch.no_grad():
            if decode_flat(z4).shape[0] != pose_size:                # not the full-resolution stage
                return x0_hat
        z = decoder_aware_filter(decode_flat, z4, directions, d_safe, active=active_mask,
                                 iters=iters, step=step)
        return z.reshape(x0_hat.shape)

    return callback
