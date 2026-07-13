"""On-policy rollout buffers for ReinFlow and the Gaussian PPO baseline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    """Flattened minibatch for the legacy Gaussian baseline."""

    images: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    ref_log_probs: torch.Tensor


class PPORolloutBuffer:
    """List-backed rollout buffer retained for the Gaussian baseline."""

    def __init__(self) -> None:
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
        """Append one vectorized baseline rollout step."""

        for storage, value_t in (
            (self.images, image),
            (self.proprio, proprio),
            (self.actions, action),
            (self.log_probs, log_prob),
            (self.rewards, reward),
            (self.dones, done),
            (self.values, value),
            (self.ref_log_probs, ref_log_prob),
        ):
            storage.append(value_t.detach().cpu())

    def compute_returns_and_advantages(
        self,
        *,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute conventional one-step GAE for the baseline."""

        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        last_value_cpu = last_value.detach().cpu()
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(last_value_cpu)
        for step in reversed(range(rewards.shape[0])):
            next_value = last_value_cpu if step == rewards.shape[0] - 1 else values[step + 1]
            continuation = 1.0 - dones[step]
            delta = rewards[step] + float(gamma) * next_value * continuation - values[step]
            last_gae = delta + float(gamma) * float(gae_lambda) * continuation * last_gae
            advantages[step] = last_gae
        self.returns = advantages + values
        self.advantages = _normalize_advantages(advantages)

    def minibatches(self, minibatch_size: int, device: torch.device) -> Iterator[RolloutBatch]:
        """Yield shuffled baseline minibatches."""

        if self.advantages is None or self.returns is None:
            raise RuntimeError("Compute returns before requesting minibatches.")
        tensors = [
            _flatten_time_env(torch.stack(values))
            for values in (
                self.images,
                self.proprio,
                self.actions,
                self.log_probs,
                self.values,
                self.ref_log_probs,
            )
        ]
        advantages = _flatten_time_env(self.advantages)
        returns = _flatten_time_env(self.returns)
        for indices in _minibatch_indices(tensors[2].shape[0], minibatch_size):
            yield RolloutBatch(
                images=tensors[0][indices].to(device),
                proprio=tensors[1][indices].to(device),
                actions=tensors[2][indices].to(device),
                old_log_probs=tensors[3][indices].to(device),
                old_values=tensors[4][indices].to(device),
                ref_log_probs=tensors[5][indices].to(device),
                advantages=advantages[indices].to(device),
                returns=returns[indices].to(device),
            )


@dataclass(frozen=True)
class ReinFlowRolloutBatch:
    """Flattened macro-transition minibatch for ReinFlow PPO."""

    images: torch.Tensor
    proprio: torch.Tensor
    paths: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    durations: torch.Tensor


class ReinFlowRolloutBuffer:
    """Fixed-capacity tensor storage for variable-duration macro transitions."""

    def __init__(self, *, capacity: int, num_envs: int, storage_device: torch.device) -> None:
        """Create an empty lazily allocated rollout buffer."""

        if capacity < 1 or num_envs < 1:
            raise ValueError("ReinFlow buffer capacity and num_envs must be positive.")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.storage_device = storage_device
        self.position = 0
        self._storage: dict[str, torch.Tensor] = {}
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        *,
        image: torch.Tensor,
        proprio: torch.Tensor,
        path: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        bootstrap_discount: torch.Tensor,
        trace_continue: torch.Tensor,
        duration: torch.Tensor,
    ) -> None:
        """Store one vectorized macro transition."""

        if self.position >= self.capacity:
            raise RuntimeError("ReinFlow rollout buffer is full.")
        values = {
            "images": image,
            "proprio": proprio,
            "paths": path,
            "log_probs": log_prob,
            "rewards": reward,
            "values": value,
            "next_values": next_value,
            "bootstrap_discounts": bootstrap_discount,
            "trace_continues": trace_continue,
            "durations": duration,
        }
        if not self._storage:
            self._allocate(values)
        for name, value_t in values.items():
            expected = self._storage[name][self.position]
            if tuple(value_t.shape) != tuple(expected.shape):
                raise ValueError(f"{name} shape mismatch: expected {tuple(expected.shape)}, got {tuple(value_t.shape)}.")
            expected.copy_(value_t.detach().to(self.storage_device))
        self.position += 1

    def compute_returns_and_advantages(self, *, gae_lambda: float) -> None:
        """Compute variable-discount GAE without crossing episode resets."""

        if self.position == 0:
            raise RuntimeError("Cannot compute GAE for an empty rollout buffer.")
        rewards = self._used("rewards")
        values = self._used("values")
        next_values = self._used("next_values")
        discounts = self._used("bootstrap_discounts")
        trace = self._used("trace_continues")
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_envs, device=self.storage_device)
        for step in reversed(range(self.position)):
            delta = rewards[step] + discounts[step] * next_values[step] - values[step]
            last_gae = delta + discounts[step] * float(gae_lambda) * trace[step] * last_gae
            advantages[step] = last_gae
        self.returns = advantages + values
        self.advantages = _normalize_advantages(advantages)

    def minibatches(self, minibatch_size: int, device: torch.device) -> Iterator[ReinFlowRolloutBatch]:
        """Yield shuffled flattened minibatches without rebuilding storage."""

        if self.advantages is None or self.returns is None:
            raise RuntimeError("Compute returns before requesting minibatches.")
        flattened = {name: _flatten_time_env(self._used(name)) for name in self._storage}
        advantages = _flatten_time_env(self.advantages)
        returns = _flatten_time_env(self.returns)
        count = int(self.position * self.num_envs)
        for indices in _minibatch_indices(count, minibatch_size):
            yield ReinFlowRolloutBatch(
                images=flattened["images"][indices].to(device, non_blocking=True),
                proprio=flattened["proprio"][indices].to(device, non_blocking=True),
                paths=flattened["paths"][indices].to(device, non_blocking=True),
                old_log_probs=flattened["log_probs"][indices].to(device, non_blocking=True),
                old_values=flattened["values"][indices].to(device, non_blocking=True),
                durations=flattened["durations"][indices].to(device, non_blocking=True),
                advantages=advantages[indices].to(device, non_blocking=True),
                returns=returns[indices].to(device, non_blocking=True),
            )

    @property
    def old_values(self) -> torch.Tensor:
        """Return values recorded during rollout for diagnostics."""

        if "values" not in self._storage:
            raise RuntimeError("The rollout buffer is empty.")
        return self._used("values")

    def _allocate(self, values: dict[str, torch.Tensor]) -> None:
        """Allocate all rollout tensors from the first transition's shapes."""

        for name, value in values.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"{name} must have leading num_envs={self.num_envs}.")
            shape = (self.capacity, *value.shape)
            self._storage[name] = torch.empty(shape, dtype=value.dtype, device=self.storage_device)

    def _used(self, name: str) -> torch.Tensor:
        """Return the populated prefix of one storage tensor."""

        return self._storage[name][: self.position]


FlowPPOBuffer = ReinFlowRolloutBuffer
FlowRolloutBatch = ReinFlowRolloutBatch


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages over all rollout decisions and environments."""

    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def _flatten_time_env(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading rollout-time and environment dimensions."""

    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _minibatch_indices(count: int, minibatch_size: int) -> Iterator[torch.Tensor]:
    """Yield shuffled index slices for one optimization epoch."""

    order = torch.randperm(int(count))
    size = max(1, int(minibatch_size))
    for start in range(0, int(count), size):
        yield order[start : start + size]
