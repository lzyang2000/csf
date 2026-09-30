# SPDX-License-Identifier: Apache-2.0
"""The semantic margin (Definition 2, Eq. 1).

For rule i with unsafe reference x_unsafe^(i) and safe reference x_safe,

    m_i(x) = - < x - x_safe, (x_unsafe^(i) - x_safe) / ||x_unsafe^(i) - x_safe|| >.

It is affine in x with unit gradient a_i = -(x_unsafe - x_safe)/||.|| and is
zero at x_safe (Lemma 1).  A lower margin places the estimate further along
the unsafe direction; the barrier of rule i is h_i = m_i (Definition 3).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def semantic_margin(x: torch.Tensor, x_unsafe: torch.Tensor, x_safe: torch.Tensor) -> torch.Tensor:
    """m(x) per sample for tensors of shape [B, ...].  Returns [B]."""
    v = (x - x_safe).reshape(x.shape[0], -1).float()
    d = (x_unsafe - x_safe).reshape(x.shape[0], -1).float()
    return -(v * d).sum(-1) / (d.norm(dim=-1) + EPS)
