from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeData:
    """Canonical in-memory representation of one demonstration episode.

    Attributes:
        obs: Time-ordered list of observation dictionaries.
        actions: ``[T, action_dim]`` float32 action array.
    """

    obs: list[dict[str, np.ndarray]]
    actions: np.ndarray
