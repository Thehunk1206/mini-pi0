"""Reward post-processing helpers for RL fine-tuning."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

Observation = dict[str, np.ndarray]


def success_bonus(info: dict[str, Any], *, bonus: float = 1.0) -> float:
    """Return a sparse success bonus from simulator info.

    Args:
        info: Simulator info dictionary.
        bonus: Reward value emitted when success is true.

    Returns:
        ``bonus`` when a common success flag is present and true, else ``0``.
    """

    for key in ("success", "is_success", "terminated_success"):
        if key in info and bool(info[key]):
            return float(bonus)
    return 0.0


def shaped_reward(base_reward: float, info: dict[str, Any]) -> float:
    """Return base simulator reward plus a conservative success bonus."""

    return float(base_reward) + success_bonus(info, bonus=1.0)


class RewardStrategy(Protocol):
    """Post-process one macro transition without changing simulator code."""

    def macro_reward(
        self,
        native_reward: float,
        previous_obs: Observation,
        next_obs: Observation,
        duration: int,
        gamma: float,
    ) -> float:
        """Return the reward stored for one policy decision."""


class NativeReward:
    """Pass simulator rewards through unchanged."""

    def macro_reward(
        self,
        native_reward: float,
        previous_obs: Observation,
        next_obs: Observation,
        duration: int,
        gamma: float,
    ) -> float:
        """Return the accumulated native macro reward."""

        del previous_obs, next_obs, duration, gamma
        return float(native_reward)


class PegPotentialReward:
    """Add the predeclared potential-based PegInsertion shaping ablation."""

    def __init__(self, *, grasp_weight: float, alignment_weight: float, insertion_weight: float) -> None:
        self.grasp_weight = float(grasp_weight)
        self.alignment_weight = float(alignment_weight)
        self.insertion_weight = float(insertion_weight)

    def macro_reward(
        self,
        native_reward: float,
        previous_obs: Observation,
        next_obs: Observation,
        duration: int,
        gamma: float,
    ) -> float:
        """Apply duration-aware potential shaping to native reward."""

        previous = self._potential(previous_obs)
        following = self._potential(next_obs)
        return float(native_reward) + float(gamma) ** int(duration) * following - previous

    def _potential(self, obs: Observation) -> float:
        """Compute grasp, alignment, and normalized insertion potential."""

        grasped = _scalar(obs, "observation.state.peg_grasped")
        alignment = max(0.0, _scalar(obs, "observation.state.peg_hole_alignment_error"))
        insertion = max(0.0, _scalar(obs, "observation.state.insertion_depth"))
        hole_depth = max(1e-6, _scalar(obs, "observation.state.hole_depth", default=1.0))
        return (
            self.grasp_weight * float(grasped > 0.5)
            + self.alignment_weight * (1.0 - float(np.tanh(20.0 * alignment)))
            + self.insertion_weight * float(np.clip(insertion / hole_depth, 0.0, 1.0))
        )


def make_reward_strategy(
    name: str,
    *,
    grasp_weight: float = 1.0,
    alignment_weight: float = 2.0,
    insertion_weight: float = 4.0,
) -> RewardStrategy:
    """Create the configured reward strategy."""

    normalized = str(name).strip().lower()
    if normalized == "native":
        return NativeReward()
    if normalized == "peg_potential":
        return PegPotentialReward(
            grasp_weight=grasp_weight,
            alignment_weight=alignment_weight,
            insertion_weight=insertion_weight,
        )
    raise ValueError(f"Unknown reward strategy: {name}")


def _scalar(obs: Observation, key: str, *, default: float = 0.0) -> float:
    """Read one scalar observation feature with a stable default."""

    value = obs.get(key)
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0]) if array.size else float(default)
