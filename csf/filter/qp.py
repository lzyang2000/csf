# SPDX-License-Identifier: Apache-2.0
"""The safe-reference tracking CBF-QP (Eq. 3) and its ridge dual (Prop. 1).

With c = x_safe - x_hat and Delta = x_filt - x_hat, Eq. 3 is

    min_Delta  1/2 || Delta - gamma c ||^2
    s.t.       <a_i, Delta> >= r_i := -rho h_i(x_hat),   i in A,

because each margin is affine (Lemma 1) with unit gradient a_i, so
h_i(x_hat + Delta) = h_i(x_hat) + <a_i, Delta>.  The problem has one primal
variable per channel but only |A| constraints, so it is solved in the dual:
with A stacking the rows a_i, Delta = gamma c + A^T lambda and lambda >= 0
maximises

    -1/2 lambda^T (A A^T + eps I) lambda + lambda^T (r - gamma A c),

which cyclic coordinate ascent (Hildreth's procedure) solves exactly.  The
ridge eps stabilises the small Gram matrix of correlated rule directions and
leaves each violated row a residual eps * lambda_i (Prop. 1).

Everything here is pure tensor math with no model dependency.
"""
from __future__ import annotations

import numpy as np
import torch

from .margin import EPS


def hildreth(G: torch.Tensor, b: torch.Tensor, ridge: float, max_iters: int = 20,
             tol: float = 1e-6) -> torch.Tensor:
    """max_{lambda >= 0} -1/2 lambda^T (G + ridge I) lambda + lambda^T b.

    Projected cyclic coordinate ascent.  G is |A| x |A| with |A| at most the
    number of rules, so the loop runs on the CPU in float64: per-coefficient
    reads of a GPU tensor would force a device sync per update.

    Returns:
        lambda [K] >= 0 on the device and dtype of `G`.
    """
    device, dtype = G.device, G.dtype
    K = int(b.shape[0])
    Gr = G.detach().to("cpu", torch.float64).numpy() + float(ridge) * np.eye(K)
    bb = b.detach().to("cpu", torch.float64).numpy()
    lam = np.zeros(K, dtype=np.float64)
    for _ in range(max_iters):
        largest_change = 0.0
        for i in range(K):
            off_diagonal = float(Gr[i] @ lam) - Gr[i, i] * lam[i]
            new = max(0.0, (bb[i] - off_diagonal) / Gr[i, i])
            largest_change = max(largest_change, abs(new - lam[i]))
            lam[i] = new
        if largest_change < tol:
            break
    return torch.from_numpy(lam).to(device=device, dtype=dtype)


def exclude_root(x: torch.Tensor, root_dims: int) -> torch.Tensor:
    """A copy of `x` with the leading `root_dims` channels zeroed (projector M)."""
    if root_dims <= 0:
        return x
    y = x.clone()
    y[..., :root_dims] = 0.0
    return y


def filter_correction(x_hat, x_safe, cfg, *, x_unsafe, active_mask=None, region=None):
    """Solve Eq. 3 for one output estimate.  Returns ``(delta, info)``.

    Args:
        x_hat: Output estimate [B, T, D] in the generator's prediction space.
        x_safe: Safe reference [B, T, D] in the same space.
        cfg: A `FilterConfig` (reads gamma, rho, ridge, max_iters, root_dims).
        x_unsafe: Unsafe references [K, B, T, D], one per declared rule.
        active_mask: bool[K] from the context gate; None enforces every rule.
            An empty active set returns a zero correction, so x_filt = x_hat.
        region: Optional frame range (a, b); the correction is computed and
            applied only there.

    Returns:
        delta [B, T, D] with x_filt = x_hat + delta (root channels untouched),
        and info {"margins": per-sample active margins before the solve,
        "lambda": per-sample dual variables, "active": bool}.
    """
    B, T, D = x_hat.shape
    rd = int(getattr(cfg, "root_dims", 0) or 0)
    a, b = region if region is not None else (0, T)
    info = {"margins": [], "lambda": [], "active": False}

    delta = torch.zeros_like(x_hat)
    if active_mask is not None:
        rows_on = [k for k, on in enumerate(active_mask) if on]
        if not rows_on:
            return delta, info
        x_unsafe = x_unsafe[rows_on]

    gamma = float(cfg.gamma)
    c = exclude_root(x_safe - x_hat, rd)
    delta[:, a:b] = gamma * c[:, a:b]                   # unconstrained optimum gamma*c

    K = x_unsafe.shape[0]
    for s in range(B):
        v = exclude_root(x_hat[s] - x_safe[s], rd)[a:b].reshape(-1)
        tracking = delta[s, a:b].reshape(-1)
        rows = []
        margins = torch.zeros(K, dtype=x_hat.dtype, device=x_hat.device)
        for k in range(K):
            direction = exclude_root(x_unsafe[k, s] - x_safe[s], rd)[a:b].reshape(-1)
            norm = direction.norm() + EPS
            margins[k] = -(v @ direction) / norm         # h_k(x_hat), Eq. 1
            rows.append(-direction / norm)               # a_k, the unit margin gradient
        A = torch.stack(rows, 0)
        required = -float(cfg.rho) * margins             # r_k = -rho h_k(x_hat)
        lam = hildreth(A @ A.t(), required - A @ tracking, cfg.ridge, cfg.max_iters)
        if float(lam.abs().max()) > 0:
            delta[s, a:b] += (A.t() @ lam).reshape(b - a, D)
            info["active"] = True
        info["margins"].append(margins.detach().cpu().tolist())
        info["lambda"].append(lam.detach().cpu().tolist())
    return delta, info
