# SPDX-License-Identifier: Apache-2.0
"""The semantic margin (Definition 2, Eq. 1) and Lemma 1."""
import pytest
import torch

from csf.filter.margin import semantic_margin


def _tensors(seed: int, B: int = 3, T: int = 5, D: int = 7):
    """Random (x, x_unsafe, x_safe), each [B, T, D]."""
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, T, D, generator=g) for _ in range(3))


def test_one_margin_per_sample():
    x, x_unsafe, x_safe = _tensors(0)
    assert semantic_margin(x, x_unsafe, x_safe).shape == (3,)


def test_zero_at_the_safe_reference():
    """Lemma 1: m(x_safe) = 0."""
    _, x_unsafe, x_safe = _tensors(1)
    assert torch.allclose(semantic_margin(x_safe, x_unsafe, x_safe), torch.zeros(3), atol=1e-6)


def test_unsafe_reference_scores_minus_its_distance():
    _, x_unsafe, x_safe = _tensors(2)
    distance = (x_unsafe - x_safe).reshape(3, -1).norm(dim=-1)
    assert torch.allclose(semantic_margin(x_unsafe, x_unsafe, x_safe), -distance, atol=1e-5)


@pytest.mark.parametrize("alpha", [-1.5, 0.3, 2.0])
def test_affine_in_x(alpha):
    """Lemma 1: m(alpha x + (1 - alpha) y) = alpha m(x) + (1 - alpha) m(y)."""
    x, x_unsafe, x_safe = _tensors(3)
    y = torch.randn_like(x)
    lhs = semantic_margin(alpha * x + (1 - alpha) * y, x_unsafe, x_safe)
    rhs = alpha * semantic_margin(x, x_unsafe, x_safe) + (1 - alpha) * semantic_margin(y, x_unsafe, x_safe)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_gradient_is_the_unit_vector_away_from_the_unsafe_reference():
    """m(x + Delta) - m(x) = <a, Delta> with a = -(x_unsafe - x_safe) / ||x_unsafe - x_safe||."""
    x, x_unsafe, x_safe = _tensors(4)
    x = x.clone().requires_grad_(True)
    semantic_margin(x, x_unsafe, x_safe).sum().backward()
    d = x_unsafe - x_safe
    expected = -d / d.reshape(3, -1).norm(dim=-1).view(3, 1, 1)
    assert torch.allclose(x.grad, expected, atol=1e-6)
    assert torch.allclose(x.grad.reshape(3, -1).norm(dim=-1), torch.ones(3), atol=1e-6)


@pytest.mark.parametrize("scale", [0.1, 1.0, 25.0])
def test_invariant_to_positive_rescaling_of_the_unsafe_direction(scale):
    """Lemma 1: only the direction of x_unsafe - x_safe matters, not its length."""
    x, x_unsafe, x_safe = _tensors(5)
    rescaled = x_safe + scale * (x_unsafe - x_safe)
    assert torch.allclose(semantic_margin(x, rescaled, x_safe),
                          semantic_margin(x, x_unsafe, x_safe), atol=1e-5)


def test_reversing_the_unsafe_direction_flips_the_sign():
    x, x_unsafe, x_safe = _tensors(6)
    mirrored = x_safe - (x_unsafe - x_safe)
    assert torch.allclose(semantic_margin(x, mirrored, x_safe),
                          -semantic_margin(x, x_unsafe, x_safe), atol=1e-5)


def test_moving_toward_the_unsafe_reference_lowers_the_margin():
    x, x_unsafe, x_safe = _tensors(7)
    step = 0.1 * (x_unsafe - x_safe)
    assert (semantic_margin(x + step, x_unsafe, x_safe) < semantic_margin(x, x_unsafe, x_safe)).all()
    assert (semantic_margin(x - step, x_unsafe, x_safe) > semantic_margin(x, x_unsafe, x_safe)).all()


def test_samples_are_scored_independently():
    x, x_unsafe, x_safe = _tensors(8)
    batched = semantic_margin(x, x_unsafe, x_safe)
    single = torch.cat([semantic_margin(x[i:i + 1], x_unsafe[i:i + 1], x_safe[i:i + 1])
                        for i in range(3)])
    assert torch.allclose(batched, single)
