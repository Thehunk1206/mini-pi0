"""PPO objective components for FM warm-start fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from mini_pi0.config.schema import RootConfig
from mini_pi0.models.registry import make_model
from mini_pi0.rl.buffers import PPORolloutBuffer


@dataclass(frozen=True)
class PPOUpdateStats:
    """Scalar metrics from one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    reference_kl: float
    total_loss: float


class FlowMatchingActorCritic(nn.Module):
    """Gaussian actor-critic initialized from a mini-pi0 FM policy.

    The actor reuses the FM observation encoder and action denoiser. It obtains
    a differentiable single-step action mean from the denoiser at the clean end
    of the flow field, then uses a diagonal Gaussian for PPO exploration.
    """

    def __init__(self, cfg: RootConfig, *, log_std_init: float) -> None:
        """Create actor-critic modules from root config."""

        super().__init__()
        self.policy = make_model(cfg)
        self.action_dim = int(cfg.model.action_dim)
        self.chunk_size = int(cfg.model.chunk_size)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))
        self.value_head = nn.Sequential(
            nn.LayerNorm(int(cfg.model.cond_dim)),
            nn.Linear(int(cfg.model.cond_dim), int(cfg.model.cond_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.model.cond_dim), 1),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return action distribution and value estimate."""

        cond = self.policy._encode_conditioning(image, proprio)  # noqa: SLF001 - intentional FM warm-start hook.
        pooled = cond.mean(dim=1) if cond.ndim == 3 else cond
        batch = int(image.shape[0])
        noisy = torch.zeros(
            (batch, self.chunk_size, self.action_dim),
            dtype=image.dtype,
            device=image.device,
        )
        tau = torch.ones((batch,), dtype=image.dtype, device=image.device)
        mean_chunk = self.policy.action_transformer(noisy, tau, cond)
        mean = mean_chunk[:, 0, :]
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), self.value_head(pooled).squeeze(-1)

    def act(self, image: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one normalized action and return log-prob/value."""

        dist, value = self(image, proprio)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), value

    def evaluate_actions(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probability, entropy, and value for actions."""

        dist, value = self(image, proprio)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOUpdater:
    """Optimize a warm-started actor-critic with PPO and reference KL."""

    def __init__(
        self,
        *,
        actor: FlowMatchingActorCritic,
        reference: FlowMatchingActorCritic,
        cfg: RootConfig,
    ) -> None:
        """Create PPO updater."""

        self.actor = actor
        self.reference = reference
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(actor.parameters(), lr=float(cfg.rl.lr))
        for param in self.reference.parameters():
            param.requires_grad = False
        self.reference.eval()

    def update(self, buffer: PPORolloutBuffer, device: torch.device) -> PPOUpdateStats:
        """Run PPO optimization over one collected rollout."""

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "reference_kl": 0.0,
            "total_loss": 0.0,
        }
        count = 0
        stop = False
        for _epoch in range(int(self.cfg.rl.epochs_per_update)):
            if stop:
                break
            for batch in buffer.minibatches(int(self.cfg.rl.minibatch_size), device):
                log_prob, entropy, value = self.actor.evaluate_actions(batch.images, batch.proprio, batch.actions)
                ratio = torch.exp(log_prob - batch.old_log_probs)
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.cfg.rl.clip_ratio),
                    1.0 + float(self.cfg.rl.clip_ratio),
                ) * batch.advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (batch.returns - value).square().mean()
                approx_kl = (batch.old_log_probs - log_prob).mean()
                ref_log_prob, _ref_entropy, _ref_value = self.reference.evaluate_actions(
                    batch.images,
                    batch.proprio,
                    batch.actions,
                )
                reference_kl = (log_prob - ref_log_prob).mean()
                loss = (
                    policy_loss
                    + float(self.cfg.rl.value_coef) * value_loss
                    - float(self.cfg.rl.entropy_coef) * entropy.mean()
                    + float(self.cfg.rl.kl_coef) * reference_kl
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.cfg.rl.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.rl.max_grad_norm))
                self.optimizer.step()

                row = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.mean().detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "reference_kl": float(reference_kl.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                }
                for key, value_f in row.items():
                    totals[key] += value_f
                count += 1
                target_kl = self.cfg.rl.target_kl
                if target_kl is not None and row["approx_kl"] > float(target_kl):
                    stop = True
                    break

        denom = max(1, count)
        return PPOUpdateStats(
            policy_loss=totals["policy_loss"] / denom,
            value_loss=totals["value_loss"] / denom,
            entropy=totals["entropy"] / denom,
            approx_kl=totals["approx_kl"] / denom,
            reference_kl=totals["reference_kl"] / denom,
            total_loss=totals["total_loss"] / denom,
        )
