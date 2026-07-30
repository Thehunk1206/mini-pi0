"""Small batched simulator contract and serial compatibility wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mini_pi0.sim.base import SimulatorAdapter

Observation = dict[str, np.ndarray]
Info = dict[str, object]


@dataclass(frozen=True)
class BatchedStepOutput:
    """One primitive simulator step for a batch of environments."""

    observations: list[Observation]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    successes: np.ndarray
    infos: list[Info]


class BatchedSimulatorAdapter(Protocol):
    """Simulator operations needed by the backend-independent RL runner."""

    backend_name: str
    num_envs: int

    def reset(self, seeds: Sequence[int]) -> list[Observation]:
        """Reset all environments with one seed per environment."""

    def reset_at(self, indices: Sequence[int], seeds: Sequence[int]) -> dict[int, Observation]:
        """Reset selected environments and return observations by index."""

    def step(self, actions: np.ndarray, active: np.ndarray) -> BatchedStepOutput:
        """Step active environments once and leave inactive entries unchanged."""

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return shared one-environment action bounds."""

    def close(self) -> None:
        """Release simulator resources."""


class SerialBatchAdapter:
    """Run existing scalar simulator adapters behind the batched contract."""

    def __init__(self, adapters: Sequence[SimulatorAdapter]) -> None:
        """Wrap a non-empty sequence of compatible scalar adapters."""

        if not adapters:
            raise ValueError("SerialBatchAdapter requires at least one environment.")
        self.adapters = list(adapters)
        self.backend_name = self.adapters[0].backend_name
        self.num_envs = len(self.adapters)
        self._observations: list[Observation | None] = [None] * self.num_envs
        self._low, self._high = _shared_action_spec(self.adapters)

    def reset(self, seeds: Sequence[int]) -> list[Observation]:
        """Reset every wrapped environment."""

        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} reset seeds, got {len(seeds)}.")
        observations = [adapter.reset(seed=int(seed)) for adapter, seed in zip(self.adapters, seeds, strict=True)]
        self._observations = list(observations)
        return observations

    def reset_at(self, indices: Sequence[int], seeds: Sequence[int]) -> dict[int, Observation]:
        """Reset selected wrapped environments."""

        if len(indices) != len(seeds):
            raise ValueError("Reset indices and seeds must have the same length.")
        result: dict[int, Observation] = {}
        for index, seed in zip(indices, seeds, strict=True):
            env_index = int(index)
            observation = self.adapters[env_index].reset(seed=int(seed))
            self._observations[env_index] = observation
            result[env_index] = observation
        return result

    def step(self, actions: np.ndarray, active: np.ndarray) -> BatchedStepOutput:
        """Step each active scalar environment exactly once."""

        action_batch = np.asarray(actions, dtype=np.float32)
        active_mask = np.asarray(active, dtype=bool).reshape(-1)
        expected_shape = (self.num_envs, self._low.size)
        if action_batch.shape != expected_shape:
            raise ValueError(f"Expected batched actions shaped {expected_shape}, got {action_batch.shape}.")
        if active_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected active mask shaped {(self.num_envs,)}, got {active_mask.shape}.")

        observations = self._current_observations()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        successes = np.zeros(self.num_envs, dtype=bool)
        infos: list[Info] = [{} for _ in range(self.num_envs)]
        for index in np.flatnonzero(active_mask):
            step = self.adapters[index].step(np.clip(action_batch[index], self._low, self._high))
            observations[index] = step.obs
            rewards[index] = float(step.reward)
            terminated[index] = bool(step.terminated)
            truncated[index] = bool(step.truncated)
            successes[index] = bool(self.adapters[index].check_success(step.info, step.obs))
            infos[index] = dict(step.info)
        self._observations = list(observations)
        return BatchedStepOutput(observations, rewards, terminated, truncated, successes, infos)

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return validated shared action bounds."""

        return self._low.copy(), self._high.copy()

    def close(self) -> None:
        """Close every wrapped adapter."""

        for adapter in self.adapters:
            adapter.close()

    def _current_observations(self) -> list[Observation]:
        """Return initialized observations or fail before an invalid step."""

        if any(observation is None for observation in self._observations):
            raise RuntimeError("Reset the batched adapter before stepping it.")
        return [observation for observation in self._observations if observation is not None]


def _shared_action_spec(adapters: Sequence[SimulatorAdapter]) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite, equal action bounds across scalar environments."""

    first_low, first_high = _valid_action_spec(*adapters[0].action_spec())
    for index, adapter in enumerate(adapters[1:], start=1):
        low, high = _valid_action_spec(*adapter.action_spec())
        if not np.array_equal(low, first_low) or not np.array_equal(high, first_high):
            raise ValueError(f"Environment {index} action bounds differ from environment 0.")
    return first_low, first_high


def _valid_action_spec(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate one action-bound pair."""

    lower = np.asarray(low, dtype=np.float32).reshape(-1)
    upper = np.asarray(high, dtype=np.float32).reshape(-1)
    if lower.shape != upper.shape or lower.size == 0:
        raise ValueError("Simulator action bounds must be non-empty vectors with matching shapes.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.all(lower < upper):
        raise ValueError("Simulator action bounds must be finite and satisfy low < high elementwise.")
    return lower, upper
