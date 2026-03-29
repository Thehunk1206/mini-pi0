from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepOutput:
    """Canonical result returned by simulator adapters for one environment step.

    Attributes:
        obs: Canonical observation dictionary for the next timestep.
        reward: Scalar reward emitted by the simulator.
        done: Episode termination flag from the simulator.
        info: Backend-specific diagnostic info dictionary.
    """

    obs: dict[str, np.ndarray]
    reward: float
    done: bool
    info: dict[str, Any]


class SimulatorAdapter(ABC):
    """Abstract simulator interface consumed by train/eval/deploy runners."""

    backend_name: str = "unknown"

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset environment state and return initial canonical observation.

        Args:
            seed: Optional seed for deterministic reset behavior.

        Returns:
            Canonical observation dictionary.
        """

        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> StepOutput:
        """Advance simulator by one control step.

        Args:
            action: Raw action vector to apply.

        Returns:
            Canonical step output.
        """

        raise NotImplementedError

    @abstractmethod
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return action bounds.

        Returns:
            Tuple ``(low, high)`` arrays each shaped ``[action_dim]``.
        """

        raise NotImplementedError

    @abstractmethod
    def render(self, camera: str = "agentview", width: int = 512, height: int = 512) -> np.ndarray:
        """Render one RGB frame from the environment.

        Args:
            camera: Camera name.
            width: Frame width.
            height: Frame height.

        Returns:
            Rendered frame as numpy array.
        """

        raise NotImplementedError

    @abstractmethod
    def check_success(self, info: dict[str, Any] | None = None, obs: dict[str, np.ndarray] | None = None) -> bool:
        """Evaluate whether task success has been achieved.

        Args:
            info: Optional backend info dict from latest step.
            obs: Optional latest canonical observation.

        Returns:
            ``True`` when task success criteria are met.
        """

        raise NotImplementedError

    def set_object_pose(self, **kwargs) -> bool:
        """Optionally randomize/override object pose for domain randomization.

        Args:
            **kwargs: Backend-specific pose controls.

        Returns:
            ``True`` if pose override was applied, else ``False``.
        """

        return False

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

        raise NotImplementedError
