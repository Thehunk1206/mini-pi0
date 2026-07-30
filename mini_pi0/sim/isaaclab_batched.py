"""Native Isaac Lab vector adapter loaded only inside an Isaac runtime."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.batched import BatchedStepOutput, Info, Observation
from mini_pi0.sim.isaaclab_adapter import IsaacLabAdapter, _finite_action_bounds, _to_numpy


class IsaacLabBatchedAdapter:
    """Use one Isaac Lab vector environment and one SimulationApp."""

    backend_name = "isaaclab"

    def __init__(self, cfg: RootConfig) -> None:
        """Construct a single Isaac environment with ``rl.num_envs`` rows."""

        vector_cfg = copy.deepcopy(cfg)
        self.num_envs = int(cfg.rl.num_envs)
        vector_cfg.simulator.env_kwargs = dict(vector_cfg.simulator.env_kwargs or {})
        vector_cfg.simulator.env_kwargs["num_envs"] = self.num_envs
        self.scalar = IsaacLabAdapter(vector_cfg)
        self.env = self.scalar.env
        self._observations: list[Observation | None] = [None] * self.num_envs
        self._low, self._high = self._action_spec()

    def reset(self, seeds: Sequence[int]) -> list[Observation]:
        """Reset all Isaac rows from the first deterministic vector seed."""

        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} reset seeds, got {len(seeds)}.")
        raw = self.env.reset(seed=int(seeds[0]))
        raw_obs, _info = raw if isinstance(raw, tuple) and len(raw) == 2 else (raw, {})
        self.scalar._last_raw_obs = raw_obs  # noqa: SLF001 - shared canonical mapper.
        observations = self._canonical_batch(raw_obs)
        self._observations = list(observations)
        return observations

    def reset_at(self, indices: Sequence[int], seeds: Sequence[int]) -> dict[int, Observation]:
        """Return rows already auto-reset by Isaac Lab after termination."""

        if len(indices) != len(seeds):
            raise ValueError("Reset indices and seeds must have the same length.")
        observations = self._current_observations()
        return {int(index): observations[int(index)] for index in indices}

    def step(self, actions: np.ndarray, active: np.ndarray) -> BatchedStepOutput:
        """Step all Isaac rows once and mask completed macro rows."""

        import torch

        action_batch = np.asarray(actions, dtype=np.float32)
        active_mask = np.asarray(active, dtype=bool).reshape(-1)
        expected = (self.num_envs, self._low.size)
        if action_batch.shape != expected or active_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected actions {expected} and active mask {(self.num_envs,)}.")
        clipped = np.clip(action_batch, self._low, self._high)
        clipped[~active_mask] = 0.0
        device = getattr(getattr(self.env, "unwrapped", self.env), "device", None)
        raw_obs, reward, terminated, truncated, raw_info = self.env.step(
            torch.as_tensor(clipped, dtype=torch.float32, device=device)
        )
        self.scalar._last_raw_obs = raw_obs  # noqa: SLF001
        canonical = self._canonical_batch(raw_obs)
        previous = self._current_observations()
        observations = [canonical[index] if active_mask[index] else previous[index] for index in range(self.num_envs)]
        infos = [_info_at(raw_info, index) for index in range(self.num_envs)]
        successes = np.asarray(
            [self.scalar._success_from_info_or_obs(info, obs) for info, obs in zip(infos, observations, strict=True)],  # noqa: SLF001
            dtype=bool,
        )
        rewards = _vector(reward, self.num_envs, dtype=np.float32)
        terminations = _vector(terminated, self.num_envs, dtype=bool)
        truncations = _vector(truncated, self.num_envs, dtype=bool)
        rewards[~active_mask] = 0.0
        terminations[~active_mask] = False
        truncations[~active_mask] = False
        successes &= active_mask
        self._observations = list(observations)
        return BatchedStepOutput(observations, rewards, terminations, truncations, successes, infos)

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one-environment action bounds."""

        return self._low.copy(), self._high.copy()

    def close(self) -> None:
        """Close the Isaac environment without modifying host software."""

        self.scalar.close()

    def _canonical_batch(self, raw_obs: Any) -> list[Observation]:
        """Canonicalize each row of an Isaac observation tree."""

        return [self.scalar._canonical_obs(raw_obs, env_index=index) for index in range(self.num_envs)]  # noqa: SLF001

    def _action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Read and validate Isaac's one-environment action bounds."""

        space = getattr(self.env, "single_action_space", None)
        if space is None:
            space = self.env.action_space
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        if low.ndim == 2 and low.shape[0] == self.num_envs:
            space = _ArraySpace(low[0], high[0])
        return _finite_action_bounds(space, fallback_dim=int(self.scalar.cfg.robot.action_dim))

    def _current_observations(self) -> list[Observation]:
        """Return initialized observations."""

        if any(observation is None for observation in self._observations):
            raise RuntimeError("Reset the Isaac vector adapter before stepping.")
        return [observation for observation in self._observations if observation is not None]


class _ArraySpace:
    """Minimal action-space view for one row of a vector Box."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low
        self.high = high


def _info_at(info: object, env_index: int) -> Info:
    """Normalize one row of nested Isaac diagnostics."""

    if not isinstance(info, dict):
        return {}
    return {str(key): _row_value(value, env_index) for key, value in info.items()}


def _row_value(value: Any, env_index: int) -> object:
    """Convert one tensor-tree row into JSON-friendly Python values."""

    if isinstance(value, dict):
        return {str(key): _row_value(child, env_index) for key, child in value.items()}
    array = np.asarray(_to_numpy(value))
    if array.ndim > 0 and array.shape[0] > env_index:
        array = array[env_index]
    return array.item() if np.asarray(array).ndim == 0 else np.asarray(array).tolist()


def _vector(value: object, size: int, *, dtype: np.dtype[Any] | type[bool]) -> np.ndarray:
    """Convert an Isaac tensor to one flat vector."""

    array = np.asarray(_to_numpy(value), dtype=dtype).reshape(-1)
    if array.size != int(size):
        raise ValueError(f"Expected simulator vector of size {size}, got {array.size}.")
    return array
