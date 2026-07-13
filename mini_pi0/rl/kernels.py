"""Transition kernels and distributions used by stochastic flow policies."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from mini_pi0.models.fm import SinusoidalTimestep

VelocityFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class TransitionDistribution(Protocol):
    """Distribution contract required by ReinFlow path sampling."""

    @property
    def mean(self) -> torch.Tensor:
        """Return the pre-clipping transition mean."""

    @property
    def stddev(self) -> torch.Tensor:
        """Return the pre-clipping transition standard deviation."""

    def sample(self) -> torch.Tensor:
        """Sample one transition state."""

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate elementwise transition log-probability."""

    def entropy(self) -> torch.Tensor:
        """Return elementwise pre-clipping Gaussian entropy."""


@dataclass(frozen=True)
class TransitionParameters:
    """Pre-clipping diagonal-Gaussian transition parameters."""

    mean: torch.Tensor
    std: torch.Tensor


class CensoredDiagonalNormal:
    """Diagonal Normal followed by deterministic clipping at finite bounds.

    Interior values use the Gaussian density. Values clipped to a boundary use
    the corresponding Gaussian tail probability, preserving a normalized mixed
    distribution for PPO likelihood ratios.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
    ) -> None:
        """Create a possibly censored diagonal Normal distribution."""

        self._normal = Normal(mean.float(), std.float())
        self._low = _broadcast_bound(low, mean)
        self._high = _broadcast_bound(high, mean)
        if (self._low is None) != (self._high is None):
            raise ValueError("Both low and high bounds are required for censored transitions.")
        if self._low is not None and not torch.all(self._low < self._high):
            raise ValueError("Every transition lower bound must be smaller than its upper bound.")

    @property
    def mean(self) -> torch.Tensor:
        """Return the pre-clipping Gaussian mean."""

        return self._normal.mean

    @property
    def stddev(self) -> torch.Tensor:
        """Return the pre-clipping Gaussian standard deviation."""

        return self._normal.stddev

    def sample(self) -> torch.Tensor:
        """Sample and clip one transition state."""

        value = self._normal.sample()
        if self._low is None or self._high is None:
            return value
        return torch.maximum(torch.minimum(value, self._high), self._low)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate density or boundary probability mass elementwise."""

        value_f = value.float()
        interior = self._normal.log_prob(value_f)
        if self._low is None or self._high is None:
            return interior
        eps = torch.finfo(interior.dtype).tiny
        lower_mass = self._normal.cdf(self._low).clamp_min(eps).log()
        upper_mass = (1.0 - self._normal.cdf(self._high)).clamp_min(eps).log()
        return torch.where(
            value_f <= self._low,
            lower_mass,
            torch.where(value_f >= self._high, upper_mass, interior),
        )

    def entropy(self) -> torch.Tensor:
        """Return pre-clipping Gaussian entropy as an exploration diagnostic."""

        return self._normal.entropy()


class FlowNoiseNetwork(nn.Module):
    """Predict smooth bounded coordinate-wise ReinFlow transition noise."""

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        cond_dim: int,
        hidden_dim: int,
        std_min: float,
        std_max: float,
        std_init: float,
        std_final_max: float | None,
        schedule_hold_fraction: float,
    ) -> None:
        """Initialize the bounded noise network and constant-noise prior."""

        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.std_min = float(std_min)
        self.initial_std_max = float(std_max)
        self.final_std_max = float(std_max if std_final_max is None else std_final_max)
        self.schedule_hold_fraction = float(schedule_hold_fraction)
        self.tau_embed = SinusoidalTimestep(int(hidden_dim))
        self.action_summary = nn.Linear(self.action_dim, int(hidden_dim))
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim) * 3, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.chunk_size * self.action_dim),
        )
        self.register_buffer("current_std_max", torch.tensor(self.initial_std_max, dtype=torch.float32))
        self._initialize_output(float(std_init))

    def forward(self, actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return bounded standard deviations shaped ``[B, H, A]``."""

        pooled = cond.mean(dim=1) if cond.ndim == 3 else cond
        features = torch.cat(
            (
                self.cond_proj(pooled),
                self.action_summary(actions.mean(dim=1)),
                self.tau_embed(tau),
            ),
            dim=-1,
        )
        raw = self.net(features).reshape(actions.shape[0], self.chunk_size, self.action_dim)
        maximum = self.current_std_max.to(device=raw.device, dtype=raw.dtype)
        return self.std_min + (maximum - self.std_min) * torch.sigmoid(raw)

    def set_training_progress(self, progress: float) -> None:
        """Update the scheduled upper noise bound at an update boundary."""

        clipped = min(1.0, max(0.0, float(progress)))
        if clipped <= self.schedule_hold_fraction or self.schedule_hold_fraction >= 1.0:
            value = self.initial_std_max
        else:
            decay_progress = (clipped - self.schedule_hold_fraction) / (1.0 - self.schedule_hold_fraction)
            value = self.initial_std_max + decay_progress * (self.final_std_max - self.initial_std_max)
        self.current_std_max.fill_(float(value))

    def _initialize_output(self, std_init: float) -> None:
        """Initialize the final projection to emit the requested constant std."""

        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("FlowNoiseNetwork final layer must be linear.")
        fraction = (std_init - self.std_min) / (self.initial_std_max - self.std_min)
        fraction = min(1.0 - 1e-6, max(1e-6, fraction))
        bias = math.log(fraction / (1.0 - fraction))
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, bias)


class FlowTransitionKernel(Protocol):
    """Interchangeable stochastic flow transition-kernel contract."""

    def distribution(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        dt: torch.Tensor,
        cond: torch.Tensor,
        velocity: VelocityFunction,
        bounds: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> TransitionDistribution:
        """Build one conditional transition distribution."""


class ReinFlowTransitionKernel(nn.Module):
    """Paper-aligned discrete Gaussian flow transition kernel."""

    def __init__(self, noise: FlowNoiseNetwork) -> None:
        """Create a kernel backed by a learned bounded noise network."""

        super().__init__()
        self.noise = noise

    def distribution(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        dt: torch.Tensor,
        cond: torch.Tensor,
        velocity: VelocityFunction,
        bounds: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> CensoredDiagonalNormal:
        """Return the conditional censored diagonal Normal transition."""

        mean = x + dt * velocity(x, tau, cond)
        std = self.noise(x, tau, cond)
        low, high = bounds if bounds is not None else (None, None)
        return CensoredDiagonalNormal(mean=mean, std=std, low=low, high=high)


def diagonal_gaussian_kl(actor: TransitionParameters, reference: TransitionParameters) -> torch.Tensor:
    """Return elementwise KL between pre-clipping diagonal Gaussian kernels."""

    return kl_divergence(Normal(actor.mean.float(), actor.std.float()), Normal(reference.mean.float(), reference.std.float()))


def _broadcast_bound(bound: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
    """Move and broadcast one optional action bound to a transition tensor."""

    if bound is None:
        return None
    value = bound.to(device=target.device, dtype=torch.float32)
    while value.ndim < target.ndim:
        value = value.unsqueeze(0)
    return value.expand_as(target.float())
