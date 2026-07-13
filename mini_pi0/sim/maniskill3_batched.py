"""Native ManiSkill vector adapter for ReinFlow rollouts."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.batched import BatchedStepOutput, Info, Observation
from mini_pi0.sim.maniskill3_adapter import ManiSkill3Adapter


class ManiSkill3BatchedAdapter:
    """Use one ManiSkill vector environment for all RL environments."""

    backend_name = "maniskill3"

    def __init__(self, cfg: RootConfig) -> None:
        """Create one native vector environment with ``rl.num_envs`` rows."""

        vector_cfg = copy.deepcopy(cfg)
        self.num_envs = int(cfg.rl.num_envs)
        vector_cfg.simulator.env_kwargs = dict(vector_cfg.simulator.env_kwargs or {})
        vector_cfg.simulator.env_kwargs["num_envs"] = self.num_envs
        self.scalar = ManiSkill3Adapter(vector_cfg)
        self.env = self.scalar.env
        self._observations: list[Observation | None] = [None] * self.num_envs
        self._seeds = [int(cfg.experiment.seed) + index for index in range(self.num_envs)]
        self._low, self._high = self._action_spec()

    def reset(self, seeds: Sequence[int]) -> list[Observation]:
        """Reset every native environment with explicit seeds."""

        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} reset seeds, got {len(seeds)}.")
        self._seeds = [int(seed) for seed in seeds]
        raw_obs, _info = self.env.reset(seed=self._seeds)
        self.scalar._last_raw_obs = raw_obs  # noqa: SLF001 - shared adapter implementation.
        self.scalar._reset_peg_diagnostics(list(range(self.num_envs)))  # noqa: SLF001
        observations = self._canonical_batch()
        self._observations = list(observations)
        return observations

    def reset_at(self, indices: Sequence[int], seeds: Sequence[int]) -> dict[int, Observation]:
        """Partially reset selected ManiSkill environment rows."""

        if len(indices) != len(seeds):
            raise ValueError("Reset indices and seeds must have the same length.")
        env_indices = [int(index) for index in indices]
        for index, seed in zip(env_indices, seeds, strict=True):
            self._seeds[index] = int(seed)
        index_tensor = torch.as_tensor(env_indices, dtype=torch.long, device=self.scalar.unwrapped.device)
        raw_obs, _info = self.env.reset(seed=self._seeds, options={"env_idx": index_tensor})
        self.scalar._last_raw_obs = raw_obs  # noqa: SLF001
        self.scalar._reset_peg_diagnostics(env_indices)  # noqa: SLF001
        canonical = self._canonical_batch()
        result: dict[int, Observation] = {}
        for index in env_indices:
            self._observations[index] = canonical[index]
            result[index] = canonical[index]
        return result

    def step(self, actions: np.ndarray, active: np.ndarray) -> BatchedStepOutput:
        """Step the vector environment once and mask completed macro rows."""

        action_batch = np.asarray(actions, dtype=np.float32)
        active_mask = np.asarray(active, dtype=bool).reshape(-1)
        expected = (self.num_envs, self._low.size)
        if action_batch.shape != expected or active_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected actions {expected} and active mask {(self.num_envs,)}.")
        clipped = np.clip(action_batch, self._low, self._high)
        clipped[~active_mask] = 0.0
        device = self.scalar.unwrapped.device
        raw_obs, reward, terminated, truncated, raw_info = self.env.step(
            torch.as_tensor(clipped, dtype=torch.float32, device=device)
        )
        self.scalar._last_raw_obs = raw_obs  # noqa: SLF001
        diagnostics = [
            self.scalar._peg_diagnostics(index, update_jam=True)  # noqa: SLF001
            for index in range(self.num_envs)
        ]
        canonical = self._canonical_batch(diagnostics)
        previous = self._current_observations()
        observations = [canonical[index] if active_mask[index] else previous[index] for index in range(self.num_envs)]
        infos = [_info_at(raw_info, index) for index in range(self.num_envs)]
        for info, values in zip(infos, diagnostics, strict=True):
            info.update(
                {
                    key.removeprefix("observation.state."): float(value[0])
                    for key, value in values.items()
                }
            )
        successes = self._successes(infos, observations) & active_mask
        rewards = _vector(reward, self.num_envs, dtype=np.float32)
        terminations = _vector(terminated, self.num_envs, dtype=bool)
        truncations = _vector(truncated, self.num_envs, dtype=bool)
        rewards[~active_mask] = 0.0
        terminations[~active_mask] = False
        truncations[~active_mask] = False
        self._observations = list(observations)
        return BatchedStepOutput(observations, rewards, terminations, truncations, successes, infos)

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one-environment action bounds."""

        return self._low.copy(), self._high.copy()

    def close(self) -> None:
        """Close the single native vector environment."""

        self.scalar.close()

    def _canonical_batch(
        self,
        diagnostics: list[dict[str, np.ndarray]] | None = None,
    ) -> list[Observation]:
        """Convert every vector row with the scalar adapter's canonical mapping."""

        rows = diagnostics if diagnostics is not None else [None] * self.num_envs
        return [
            self.scalar._canonical_obs_from_env(index, peg_diagnostics=rows[index])  # noqa: SLF001
            for index in range(self.num_envs)
        ]

    def _successes(self, infos: list[Info], observations: list[Observation]) -> np.ndarray:
        """Read per-row success from task diagnostics or canonical progress."""

        values = np.zeros(self.num_envs, dtype=bool)
        evaluation = self.scalar._evaluate_task()  # noqa: SLF001
        for index, (info, observation) in enumerate(zip(infos, observations, strict=True)):
            candidate = info.get("success", info.get("is_success"))
            if candidate is None and "success" in evaluation:
                candidate = _row_value(evaluation["success"], index)
            if candidate is None:
                progress = observation.get("observation.state.task_progress", np.zeros(1))
                candidate = float(np.asarray(progress).reshape(-1)[0]) >= 1.0
            values[index] = bool(candidate)
        return values

    def _action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Validate native single-environment action bounds."""

        space = getattr(self.env, "single_action_space", None)
        if space is None:
            space = self.env.action_space
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        if low.ndim == 2 and low.shape[0] == self.num_envs:
            low = low[0]
            high = high[0]
        low = low.reshape(-1)
        high = high.reshape(-1)
        if low.shape != high.shape or not np.all(low < high):
            raise ValueError("Invalid ManiSkill vector action bounds.")
        return low, high

    def _current_observations(self) -> list[Observation]:
        """Return initialized observations."""

        if any(observation is None for observation in self._observations):
            raise RuntimeError("Reset the ManiSkill vector adapter before stepping.")
        return [observation for observation in self._observations if observation is not None]


def _info_at(info: object, env_index: int) -> Info:
    """Normalize one row of a nested ManiSkill info dictionary."""

    if not isinstance(info, dict):
        return {}
    return {str(key): _row_value(value, env_index) for key, value in info.items()}


def _row_value(value: Any, env_index: int) -> object:
    """Convert one batched tensor/array row into a Python value."""

    if isinstance(value, dict):
        return {str(key): _row_value(child, env_index) for key, child in value.items()}
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim > 0 and array.shape[0] > env_index:
        array = array[env_index]
    if np.asarray(array).ndim == 0:
        return np.asarray(array).item()
    return np.asarray(array).tolist()


def _vector(value: object, size: int, *, dtype: np.dtype[Any] | type[bool]) -> np.ndarray:
    """Convert a simulator tensor to a flat vector of known size."""

    raw = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.asarray(raw, dtype=dtype).reshape(-1)
    if array.size != int(size):
        raise ValueError(f"Expected simulator vector of size {size}, got {array.size}.")
    return array
