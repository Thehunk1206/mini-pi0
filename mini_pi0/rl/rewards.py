from __future__ import annotations

"""Reward post-processing helpers for RL fine-tuning."""

from typing import Any


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
