from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionStats:
    """Container for per-dimension action normalization statistics."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """Create an action statistics object.

        Args:
            mean: Per-dimension action mean.
            std: Per-dimension action standard deviation.
        """

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def from_actions(cls, actions: np.ndarray) -> "ActionStats":
        """Estimate normalization statistics from a full action array.

        Args:
            actions: Action matrix shaped ``[N, action_dim]``.

        Returns:
            New ``ActionStats`` with epsilon-stabilized standard deviation.
        """

        mean = actions.mean(axis=0)
        std = actions.std(axis=0) + 1e-6
        return cls(mean=mean, std=std)

    @classmethod
    def load(cls, path: str) -> "ActionStats":
        """Load action statistics from JSON.

        Args:
            path: JSON file path containing ``mean`` and ``std`` arrays.

        Returns:
            Loaded stats object.
        """

        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))

    def save(self, path: str) -> None:
        """Persist action statistics to JSON.

        Args:
            path: Destination path.
        """

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f, indent=2)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions using stored stats.

        Args:
            actions: Raw action array.

        Returns:
            Normalized actions.
        """

        return (actions - self.mean) / self.std

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Convert normalized actions back to raw action scale.

        Args:
            actions: Normalized action array.

        Returns:
            Denormalized actions.
        """

        return actions * self.std + self.mean
