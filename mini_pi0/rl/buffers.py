from __future__ import annotations

"""Rollout storage for on-policy PPO updates."""

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class RolloutBatch:
    """Flattened PPO minibatch."""

    images: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    ref_log_probs: torch.Tensor


class PPORolloutBuffer:
    """Fixed-size rollout buffer with GAE computation."""

    def __init__(self) -> None:
        """Create an empty rollout buffer."""

        self.images: list[torch.Tensor] = []
        self.proprio: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.ref_log_probs: list[torch.Tensor] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        *,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        ref_log_prob: torch.Tensor,
    ) -> None:
        """Append one vectorized rollout step."""

        self.images.append(image.detach().cpu())
        self.proprio.append(proprio.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.dones.append(done.detach().cpu())
        self.values.append(value.detach().cpu())
        self.ref_log_probs.append(ref_log_prob.detach().cpu())

    def compute_returns_and_advantages(
        self,
        *,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns."""

        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        last_value_cpu = last_value.detach().cpu()
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(last_value_cpu)
        for step in reversed(range(rewards.shape[0])):
            next_value = last_value_cpu if step == rewards.shape[0] - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + float(gamma) * next_value * nonterminal - values[step]
            last_gae = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.advantages = advantages
        self.returns = returns

    def minibatches(self, minibatch_size: int, device: torch.device) -> Iterator[RolloutBatch]:
        """Yield shuffled PPO minibatches."""

        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before sampling minibatches.")
        images = _flatten_time_env(torch.stack(self.images))
        proprio = _flatten_time_env(torch.stack(self.proprio))
        actions = _flatten_time_env(torch.stack(self.actions))
        log_probs = _flatten_time_env(torch.stack(self.log_probs))
        values = _flatten_time_env(torch.stack(self.values))
        ref_log_probs = _flatten_time_env(torch.stack(self.ref_log_probs))
        advantages = _flatten_time_env(self.advantages)
        returns = _flatten_time_env(self.returns)

        n = int(actions.shape[0])
        order = torch.randperm(n)
        batch_size = int(max(1, minibatch_size))
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield RolloutBatch(
                images=images[idx].to(device),
                proprio=proprio[idx].to(device),
                actions=actions[idx].to(device),
                old_log_probs=log_probs[idx].to(device),
                advantages=advantages[idx].to(device),
                returns=returns[idx].to(device),
                old_values=values[idx].to(device),
                ref_log_probs=ref_log_probs[idx].to(device),
            )


def _flatten_time_env(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading time and environment dimensions."""

    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
