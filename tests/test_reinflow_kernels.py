"""Tests for exact clipped ReinFlow transition likelihoods."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.distributions import Normal

from mini_pi0.rl.kernels import CensoredDiagonalNormal


def test_censored_normal_uses_tail_mass_at_bounds() -> None:
    mean = torch.zeros(1, 1, 2)
    std = torch.ones_like(mean)
    low = torch.full_like(mean, -1.0)
    high = torch.full_like(mean, 1.0)
    distribution = CensoredDiagonalNormal(mean, std, low, high)
    target = torch.tensor([[[-1.0, 1.0]]])

    actual = distribution.log_prob(target)
    normal = Normal(mean, std)
    expected = torch.stack(
        (
            normal.cdf(low)[..., 0].log(),
            (1.0 - normal.cdf(high)[..., 1]).log(),
        ),
        dim=-1,
    )

    assert torch.allclose(actual, expected)


def test_censored_normal_samples_respect_bounds() -> None:
    distribution = CensoredDiagonalNormal(
        mean=torch.zeros(4096, 1, 1),
        std=torch.full((4096, 1, 1), 10.0),
        low=torch.tensor([-0.5]),
        high=torch.tensor([0.5]),
    )

    samples = distribution.sample()

    assert torch.all(samples >= -0.5)
    assert torch.all(samples <= 0.5)
