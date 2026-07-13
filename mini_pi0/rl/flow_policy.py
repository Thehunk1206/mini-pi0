"""ReinFlow actor, detached-feature critic, and stable path evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mini_pi0.config.schema import RootConfig
from mini_pi0.models.registry import make_model
from mini_pi0.rl.kernels import (
    FlowNoiseNetwork,
    ReinFlowTransitionKernel,
    TransitionParameters,
)

ActionBounds = tuple[torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True)
class ReinFlowSample:
    """One sampled denoising path and actor-critic outputs."""

    path: torch.Tensor
    action_chunk: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    initial_noise: torch.Tensor


@dataclass(frozen=True)
class ReinFlowPathEvaluation:
    """Current-policy evaluation of a stored denoising path."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ReinFlowActor(nn.Module):
    """Flow-matching actor with a learned ReinFlow transition kernel."""

    def __init__(self, cfg: RootConfig, *, policy: nn.Module | None = None) -> None:
        """Build the FM velocity actor and bounded noise network."""

        super().__init__()
        self.cfg = cfg
        self.policy = policy if policy is not None else make_model(cfg)
        self.action_dim = int(cfg.model.action_dim)
        self.chunk_size = int(cfg.model.chunk_size)
        self.flow_steps = int(cfg.rl.flow_steps)
        noise = FlowNoiseNetwork(
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            cond_dim=int(cfg.model.cond_dim),
            hidden_dim=int(cfg.model.d_model),
            std_min=float(cfg.rl.noise_std_min),
            std_max=float(cfg.rl.noise_std_max),
            std_init=float(cfg.rl.noise_std_init),
            std_final_max=cfg.rl.noise_std_final_max,
            schedule_hold_fraction=float(cfg.rl.noise_schedule_hold_fraction),
        )
        self.kernel = ReinFlowTransitionKernel(noise)
        if bool(cfg.rl.freeze_vision_during_rl):
            self._freeze_vision_backbone()
        self.likelihood_mode()

    def likelihood_mode(self) -> None:
        """Disable untracked dropout and mutable normalization state."""

        self.eval()
        vision = self._vision_backbone()
        if vision is not None:
            vision.eval()

    def encode_conditioning(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Encode observations with deterministic actor modules."""

        self.likelihood_mode()
        return self.policy._encode_conditioning(image, proprio)  # noqa: SLF001

    def sample_from_conditioning(
        self,
        cond: torch.Tensor,
        *,
        bounds: ActionBounds,
        initial_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a complete path from precomputed observation conditioning."""

        batch = int(cond.shape[0])
        x = initial_noise if initial_noise is not None else torch.randn(
            batch,
            self.chunk_size,
            self.action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        initial = x
        path = [x]
        log_prob = torch.zeros(batch, device=cond.device, dtype=torch.float32)
        entropy = torch.zeros_like(log_prob)
        time_grid = self._time_grid(cond)
        for step in range(self.flow_steps):
            tau = time_grid[step].expand(batch)
            distribution = self.kernel.distribution(
                x,
                tau,
                time_grid[step + 1] - time_grid[step],
                cond,
                self.policy.action_transformer,
                bounds,
            )
            x = distribution.sample().to(dtype=cond.dtype)
            path.append(x)
            log_prob = log_prob + distribution.log_prob(x).sum(dim=(-1, -2))
            entropy = entropy + distribution.entropy().sum(dim=(-1, -2))
        entropy = self._normalize_entropy(entropy)
        return torch.stack(path, dim=1), log_prob, entropy, initial

    def evaluate_from_conditioning(
        self,
        cond: torch.Tensor,
        path: torch.Tensor,
        *,
        bounds: ActionBounds,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate an exact stored path under current actor parameters."""

        self._validate_path(path)
        batch = int(path.shape[0])
        log_prob = torch.zeros(batch, device=path.device, dtype=torch.float32)
        entropy = torch.zeros_like(log_prob)
        time_grid = self._time_grid(path)
        for step in range(self.flow_steps):
            tau = time_grid[step].expand(batch)
            distribution = self.kernel.distribution(
                path[:, step],
                tau,
                time_grid[step + 1] - time_grid[step],
                cond,
                self.policy.action_transformer,
                bounds,
            )
            log_prob = log_prob + distribution.log_prob(path[:, step + 1]).sum(dim=(-1, -2))
            entropy = entropy + distribution.entropy().sum(dim=(-1, -2))
        return log_prob, self._normalize_entropy(entropy)

    def deterministic_from_conditioning(
        self,
        cond: torch.Tensor,
        *,
        bounds: ActionBounds,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the deterministic FM Euler ODE from shared base noise."""

        batch = int(cond.shape[0])
        x = initial_noise if initial_noise is not None else torch.randn(
            batch,
            self.chunk_size,
            self.action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        time_grid = self._time_grid(cond)
        for step in range(self.flow_steps):
            tau = time_grid[step].expand(batch)
            x = x + (time_grid[step + 1] - time_grid[step]) * self.policy.action_transformer(x, tau, cond)
            if bounds is not None and bool(self.cfg.rl.clip_denoised_actions):
                x = torch.maximum(torch.minimum(x, bounds[1]), bounds[0])
        return x

    def transition_parameters(
        self,
        cond: torch.Tensor,
        path: torch.Tensor,
    ) -> TransitionParameters:
        """Return stacked pre-clipping Gaussian parameters along a path."""

        self._validate_path(path)
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        time_grid = self._time_grid(path)
        for step in range(self.flow_steps):
            tau = time_grid[step].expand(path.shape[0])
            x = path[:, step]
            means.append(x + (time_grid[step + 1] - time_grid[step]) * self.policy.action_transformer(x, tau, cond))
            stds.append(self.kernel.noise(x, tau, cond))
        return TransitionParameters(mean=torch.stack(means, dim=1), std=torch.stack(stds, dim=1))

    def velocity_path(self, cond: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Return velocity predictions at every stored transition state."""

        self._validate_path(path)
        time_grid = self._time_grid(path)
        values = [
            self.policy.action_transformer(path[:, step], time_grid[step].expand(path.shape[0]), cond)
            for step in range(self.flow_steps)
        ]
        return torch.stack(values, dim=1)

    def _normalize_entropy(self, entropy: torch.Tensor) -> torch.Tensor:
        """Convert block entropy to per-symbol entropy when configured."""

        if not bool(self.cfg.rl.entropy_per_symbol):
            return entropy
        return entropy / float(self.flow_steps * self.chunk_size * self.action_dim)

    def _time_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the actor's uniform flow integration grid."""

        return torch.linspace(0.0, 1.0, self.flow_steps + 1, device=tensor.device, dtype=tensor.dtype)

    def _validate_path(self, path: torch.Tensor) -> None:
        """Fail fast when a rollout path does not match actor dimensions."""

        expected = (self.flow_steps + 1, self.chunk_size, self.action_dim)
        if path.ndim != 4 or tuple(path.shape[1:]) != expected:
            raise ValueError(f"Expected flow path trailing shape {expected}, got {tuple(path.shape)}.")

    def _vision_backbone(self) -> nn.Module | None:
        """Return the wrapped policy vision backbone when present."""

        encoder = getattr(self.policy, "obs_encoder", None)
        backbone = getattr(encoder, "img_backbone", None)
        return backbone if isinstance(backbone, nn.Module) else None

    def _freeze_vision_backbone(self) -> None:
        """Freeze both parameters and mutable state of the vision backbone."""

        vision = self._vision_backbone()
        if vision is None:
            return
        for parameter in vision.parameters():
            parameter.requires_grad = False
        vision.eval()


class FlowCritic(nn.Module):
    """Value head trained from detached actor observation features."""

    def __init__(self, cond_dim: int, *, output_bias: float = 0.0) -> None:
        """Create a separate critic that cannot update actor features."""

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cond_dim), 1),
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("FlowCritic final layer must be linear.")
        nn.init.constant_(output.bias, float(output_bias))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Estimate state value without propagating into actor conditioning."""

        pooled = cond.mean(dim=1) if cond.ndim == 3 else cond
        return self.net(pooled.detach()).squeeze(-1)


class ReinFlowActorCritic(nn.Module):
    """Composition root for a ReinFlow actor and detached-feature critic."""

    def __init__(self, cfg: RootConfig, *, policy: nn.Module | None = None) -> None:
        """Build actor and critic modules from the root configuration."""

        super().__init__()
        self.cfg = cfg
        self.actor = ReinFlowActor(cfg, policy=policy)
        self.critic = FlowCritic(int(cfg.model.cond_dim), output_bias=float(cfg.rl.critic_output_bias_init))

    @property
    def policy(self) -> nn.Module:
        """Expose the wrapped FM policy for checkpoint compatibility."""

        return self.actor.policy

    def sample_path(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        *,
        bounds: ActionBounds = None,
        initial_noise: torch.Tensor | None = None,
    ) -> ReinFlowSample:
        """Sample a stochastic flow path and value estimate."""

        cond = self.actor.encode_conditioning(image, proprio)
        path, log_prob, entropy, initial = self.actor.sample_from_conditioning(
            cond,
            bounds=bounds,
            initial_noise=initial_noise,
        )
        return ReinFlowSample(
            path=path,
            action_chunk=path[:, -1],
            log_prob=log_prob,
            entropy=entropy,
            value=self.critic(cond),
            initial_noise=initial,
        )

    def evaluate_path(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        path: torch.Tensor,
        *,
        bounds: ActionBounds = None,
    ) -> ReinFlowPathEvaluation:
        """Evaluate a stored path and current value estimate."""

        cond = self.actor.encode_conditioning(image, proprio)
        log_prob, entropy = self.actor.evaluate_from_conditioning(cond, path, bounds=bounds)
        return ReinFlowPathEvaluation(log_prob=log_prob, entropy=entropy, value=self.critic(cond))

    @torch.no_grad()
    def deterministic_sample(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        *,
        bounds: ActionBounds = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a noise-free deterministic ODE action chunk."""

        cond = self.actor.encode_conditioning(image, proprio)
        return self.actor.deterministic_from_conditioning(cond, bounds=bounds, initial_noise=initial_noise)

    def value(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return critic values for a batch of observations."""

        return self.critic(self.actor.encode_conditioning(image, proprio))


StochasticFlowPolicy = ReinFlowActorCritic
FlowPolicyOutput = ReinFlowSample
FlowPathEvaluation = ReinFlowPathEvaluation
