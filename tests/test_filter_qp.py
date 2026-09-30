# SPDX-License-Identifier: Apache-2.0
"""The safe-reference tracking CBF-QP (Eq. 3), its ridge dual (Prop. 1) and Theorem 1."""
import numpy as np
import pytest
import torch

from csf.filter.config import FilterConfig
from csf.filter.margin import semantic_margin
from csf.filter.qp import exclude_root, filter_correction, hildreth


def _cfg(**overrides) -> FilterConfig:
    values = dict(gamma=0.0, rho=1.0, ridge=1e-2, max_iters=20, root_dims=0)
    values.update(overrides)
    return FilterConfig(**values)


def _references(seed: int, K: int = 1, B: int = 1, T: int = 4, D: int = 6):
    """Random x_safe [B, T, D] and x_unsafe [K, B, T, D]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g), torch.randn(K, B, T, D, generator=g)


def _margins(x, x_unsafe, x_safe, root_dims: int = 0) -> torch.Tensor:
    """h_k(x) = m_k(x) on the non-root channels, as the filter measures it: [K, B]."""
    return torch.stack([
        semantic_margin(exclude_root(x, root_dims), exclude_root(u, root_dims),
                        exclude_root(x_safe, root_dims))
        for u in x_unsafe
    ])


def _safe_estimate(x_safe, x_unsafe, root_dims: int = 0, step: float = 0.5):
    """An estimate displaced away from every unsafe reference (h_k > 0 for all k)."""
    away = torch.zeros_like(x_safe)
    for u in x_unsafe:
        d = exclude_root(u - x_safe, root_dims)
        away -= d / d.reshape(d.shape[0], -1).norm(dim=-1).view(-1, 1, 1)
    x_hat = x_safe + step * away
    x_hat[..., :root_dims] = torch.randn_like(x_hat[..., :root_dims])
    return x_hat


# ----------------------------------------------------------------- hildreth
@pytest.mark.parametrize("b", [0.7, -0.3])
@pytest.mark.parametrize("ridge", [0.0, 0.5])
def test_hildreth_single_row_closed_form(b, ridge):
    lam = hildreth(torch.tensor([[2.0]]), torch.tensor([b]), ridge)
    assert float(lam) == pytest.approx(max(0.0, b / (2.0 + ridge)), abs=1e-7)


def test_hildreth_satisfies_the_kkt_conditions():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(4, 10, generator=g, dtype=torch.float64)
    G, b, ridge = A @ A.t(), torch.randn(4, generator=g, dtype=torch.float64), 1e-3
    lam = hildreth(G, b, ridge, max_iters=2000, tol=1e-12)
    residual = (G + ridge * torch.eye(4, dtype=torch.float64)) @ lam - b
    assert (lam >= 0).all()
    for i in range(4):
        if lam[i] > 1e-9:
            assert abs(float(residual[i])) < 1e-8
        else:
            assert float(residual[i]) > -1e-8


def test_hildreth_is_the_linear_solve_when_every_row_is_active():
    G = torch.tensor([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]], dtype=torch.float64)
    expected = torch.tensor([0.4, 1.1, 0.7], dtype=torch.float64)
    b = (G + 0.1 * torch.eye(3, dtype=torch.float64)) @ expected
    lam = hildreth(G, b, 0.1, max_iters=500, tol=1e-12)
    assert torch.allclose(lam, expected, atol=1e-9)


def test_hildreth_with_nonpositive_targets_returns_exact_zeros():
    A = torch.randn(3, 5)
    lam = hildreth(A @ A.t(), torch.tensor([-1.0, 0.0, -0.2]), 1e-2)
    assert torch.equal(lam, torch.zeros(3))


def test_hildreth_returns_the_dtype_of_the_gram_matrix():
    lam = hildreth(torch.eye(2, dtype=torch.float32), torch.tensor([1.0, 2.0]), 1e-2)
    assert lam.dtype == torch.float32 and lam.device.type == "cpu"


# ------------------------------------------------------------- exclude_root
def test_exclude_root_zeroes_only_the_leading_channels():
    x = torch.randn(2, 3, 6)
    y = exclude_root(x, 2)
    assert torch.equal(y[..., :2], torch.zeros(2, 3, 2))
    assert torch.equal(y[..., 2:], x[..., 2:])
    assert not torch.equal(x[..., :2], torch.zeros(2, 3, 2)), "the input must not be modified"


def test_exclude_root_with_no_root_channels_is_the_identity():
    x = torch.randn(2, 3, 6)
    assert exclude_root(x, 0) is x


# -------------------------------------------------------- filter_correction
def test_single_violated_rule_is_projected_to_the_boundary():
    """Theorem 1 (gamma = 0, rho = 1): h(x_filt) = 0 up to the ridge residual."""
    x_safe, x_unsafe = _references(0)
    x_hat = x_safe + 0.9 * (x_unsafe[0] - x_safe)
    h_before = float(_margins(x_hat, x_unsafe, x_safe)[0, 0])
    assert h_before < -1.0

    delta, info = filter_correction(x_hat, x_safe, _cfg(ridge=1e-8), x_unsafe=x_unsafe)
    h_after = float(_margins(x_hat + delta, x_unsafe, x_safe)[0, 0])
    assert abs(h_after) < 1e-4
    assert info["active"]


@pytest.mark.parametrize("ridge", [1e-2, 0.1, 1.0])
def test_ridge_residual_matches_prop1(ridge):
    """Prop. 1 / Eq. 4 at gamma = 0: one violated unit row keeps h(x_filt) = eps / (1 + eps) h(x_hat)."""
    x_safe, x_unsafe = _references(1)
    x_hat = x_safe + 0.9 * (x_unsafe[0] - x_safe)
    h_before = float(_margins(x_hat, x_unsafe, x_safe)[0, 0])

    delta, _ = filter_correction(x_hat, x_safe, _cfg(ridge=ridge, max_iters=200), x_unsafe=x_unsafe)
    h_after = float(_margins(x_hat + delta, x_unsafe, x_safe)[0, 0])
    assert h_after == pytest.approx(ridge / (1 + ridge) * h_before, rel=1e-4)


def test_barrier_only_correction_moves_straight_away_from_the_unsafe_reference():
    """At gamma = 0 the minimum-norm correction is parallel to the margin gradient."""
    x_safe, x_unsafe = _references(2)
    x_hat = x_safe + 0.7 * (x_unsafe[0] - x_safe)
    delta, _ = filter_correction(x_hat, x_safe, _cfg(), x_unsafe=x_unsafe)
    away = -(x_unsafe[0] - x_safe)
    cosine = torch.nn.functional.cosine_similarity(delta.reshape(1, -1), away.reshape(1, -1))
    assert float(cosine) == pytest.approx(1.0, abs=1e-5)


def test_estimate_satisfying_every_rule_passes_through_exactly():
    """Theorem 1 at gamma = 0: h(x_hat) >= 0 for all rules means x_filt = x_hat bit for bit."""
    x_safe, x_unsafe = _references(3, K=3, B=2, T=8, D=16)
    x_hat = _safe_estimate(x_safe, x_unsafe)
    assert (_margins(x_hat, x_unsafe, x_safe) > 0).all()
    original = x_hat.clone()

    delta, info = filter_correction(x_hat, x_safe, _cfg(), x_unsafe=x_unsafe)
    assert torch.equal(delta, torch.zeros_like(x_hat))
    assert not info["active"]
    assert all(lam == [0.0, 0.0, 0.0] for lam in info["lambda"])
    assert torch.equal(x_hat, original)


@pytest.mark.parametrize(("rho", "gamma", "root_dims"),
                         [(1.0, 0.0, 0), (0.5, 0.0, 0), (1.0, 0.3, 2), (0.5, 0.8, 2)])
def test_every_active_rule_is_contracted(rho, gamma, root_dims):
    """Theorem 1: h_i(x_filt) >= (1 - rho) h_i(x_hat) - eps lambda_i for every active row."""
    x_safe, x_unsafe = _references(4, K=4, B=2, T=5, D=8)
    x_hat = x_safe.clone()
    for k, weight in enumerate([0.8, 0.4, 0.0, -0.3]):
        x_hat += weight * (x_unsafe[k] - x_safe)
    before = _margins(x_hat, x_unsafe, x_safe, root_dims)
    assert (before < 0).any(), "at least one rule must start violated"

    cfg = _cfg(rho=rho, gamma=gamma, ridge=1e-2, max_iters=2000, root_dims=root_dims)
    delta, info = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe)
    after = _margins(x_hat + delta, x_unsafe, x_safe, root_dims)
    lam = torch.tensor(info["lambda"]).t()                     # [K, B]
    assert (lam >= 0).all()
    assert torch.allclose(torch.tensor(info["margins"]).t(), before, atol=1e-5)
    assert (after >= (1 - rho) * before - cfg.ridge * lam - 1e-4).all()


def test_correlated_violated_rules_are_lifted_jointly():
    """Two strongly correlated rules, both violated: the joint solve lifts both, not just the worst."""
    T, D = 4, 16
    x_safe = torch.zeros(1, T, D)
    x_unsafe = torch.zeros(2, 1, T, D)
    x_unsafe[0, ..., 0] = 1.0
    x_unsafe[1, ..., 0] = 0.7
    x_unsafe[1, ..., 1] = 0.7
    x_hat = 0.9 * (x_unsafe[0] + x_unsafe[1])
    assert (_margins(x_hat, x_unsafe, x_safe) < 0).all()

    delta, _ = filter_correction(x_hat, x_safe, _cfg(ridge=1e-6, max_iters=500), x_unsafe=x_unsafe)
    assert (_margins(x_hat + delta, x_unsafe, x_safe) > -1e-3).all()


@pytest.mark.parametrize("gamma", [0.0, 0.2, 0.5, 1.0])
def test_slack_constraints_give_the_gamma_interpolation(gamma):
    """No violated rule: x_filt = x_hat + gamma (x_safe - x_hat) off the root; the root is kept."""
    root_dims = 2
    x_safe, x_unsafe = _references(5, K=3, B=2, T=8, D=16)
    x_hat = _safe_estimate(x_safe, x_unsafe, root_dims)
    assert (_margins(x_hat, x_unsafe, x_safe, root_dims) > 0).all()

    cfg = _cfg(gamma=gamma, root_dims=root_dims)
    delta, info = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe)
    x_filt = x_hat + delta
    blend = x_hat + gamma * (x_safe - x_hat)
    assert torch.allclose(x_filt[..., root_dims:], blend[..., root_dims:], atol=1e-6)
    assert torch.equal(x_filt[..., :root_dims], x_hat[..., :root_dims])
    if gamma < cfg.rho:                       # at gamma = rho every row is exactly tight
        assert not info["active"], "no dual variable is needed"


def test_root_channels_are_never_corrected():
    root_dims = 3
    x_safe, x_unsafe = _references(6, K=2, B=2, T=5, D=9)
    x_hat = x_safe + 0.9 * (x_unsafe[0] - x_safe)
    delta, info = filter_correction(x_hat, x_safe, _cfg(gamma=0.5, root_dims=root_dims),
                                    x_unsafe=x_unsafe)
    assert info["active"]
    assert torch.equal(delta[..., :root_dims], torch.zeros_like(delta[..., :root_dims]))
    assert delta[..., root_dims:].abs().max() > 0


def test_empty_active_set_returns_zero_correction_even_with_tracking():
    x_safe, x_unsafe = _references(7, K=2)
    x_hat = x_safe + 0.9 * (x_unsafe[0] - x_safe)
    cfg = _cfg(gamma=0.5)
    delta, info = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe, active_mask=[False, False])
    assert torch.equal(delta, torch.zeros_like(x_hat))
    assert not info["active"]
    on, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe, active_mask=[True, False])
    assert on.abs().max() > 0


def test_no_mask_enforces_every_rule():
    x_safe, x_unsafe = _references(8, K=3)
    x_hat = x_safe + 0.6 * (x_unsafe[0] - x_safe) + 0.6 * (x_unsafe[2] - x_safe)
    cfg = _cfg(gamma=0.2)
    everything, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe)
    all_true, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe, active_mask=[True] * 3)
    assert torch.equal(everything, all_true)


def test_mask_selects_the_enforced_rules():
    x_safe, x_unsafe = _references(9, K=3)
    x_hat = x_safe + 0.6 * (x_unsafe[0] - x_safe) + 0.6 * (x_unsafe[1] - x_safe)
    cfg = _cfg(gamma=0.2)
    masked, info = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe,
                                     active_mask=[True, False, True])
    subset, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe[[0, 2]])
    assert torch.allclose(masked, subset, atol=1e-7)
    assert len(info["margins"][0]) == 2


def test_region_limits_the_correction_to_its_frames():
    x_safe, x_unsafe = _references(10, K=2, B=2, T=7, D=5)
    x_hat = x_safe + 0.8 * (x_unsafe[1] - x_safe)
    cfg = _cfg(gamma=0.3, root_dims=1)
    delta, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe, region=(2, 5))
    assert torch.equal(delta[:, :2], torch.zeros_like(delta[:, :2]))
    assert torch.equal(delta[:, 5:], torch.zeros_like(delta[:, 5:]))
    sliced, _ = filter_correction(x_hat[:, 2:5], x_safe[:, 2:5], cfg, x_unsafe=x_unsafe[:, :, 2:5])
    assert torch.allclose(delta[:, 2:5], sliced, atol=1e-6)


def test_samples_are_filtered_independently():
    x_safe, x_unsafe = _references(11, K=2, B=3)
    x_hat = x_safe + 0.8 * (x_unsafe[0] - x_safe)
    x_hat[1] = x_safe[1] - 0.8 * (x_unsafe[0, 1] - x_safe[1])
    cfg = _cfg(gamma=0.2)
    batched, _ = filter_correction(x_hat, x_safe, cfg, x_unsafe=x_unsafe)
    for s in range(3):
        single, _ = filter_correction(x_hat[s:s + 1], x_safe[s:s + 1], cfg,
                                      x_unsafe=x_unsafe[:, s:s + 1])
        assert torch.allclose(batched[s:s + 1], single, atol=1e-6)


def test_more_violation_needs_a_larger_correction():
    x_safe, x_unsafe = _references(12)
    norms = []
    for level in (0.2, 0.5, 0.9):
        x_hat = x_safe + level * (x_unsafe[0] - x_safe)
        delta, _ = filter_correction(x_hat, x_safe, _cfg(), x_unsafe=x_unsafe)
        norms.append(float(delta.norm()))
    assert norms == sorted(norms) and np.all(np.diff(norms) > 0)
