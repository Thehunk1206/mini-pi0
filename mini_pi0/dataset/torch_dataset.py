from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_pi0.dataset.episodes import EpisodeData
from mini_pi0.dataset.stats import ActionStats


class ActionChunkDataset(Dataset):
    """Torch dataset of observation + fixed-horizon action chunks.

    Each sample uses observation at timestep ``t`` and predicts normalized
    action sequence ``[t, t + chunk_size)``.
    """

    def __init__(
        self,
        episodes: list[EpisodeData],
        chunk_size: int,
        image_key: str,
        proprio_keys: list[str],
        action_stats: ActionStats,
        observation_key: str | None = None,
    ):
        """Build dataset samples from episodic demonstrations.

        Args:
            episodes: Canonical demonstration episodes.
            chunk_size: Number of consecutive actions predicted per sample.
            image_key: Observation image key.
            proprio_keys: Ordered proprioception keys for vector concatenation.
            action_stats: Statistics used to normalize action targets.
            observation_key: Override observation key used as model visual input.
                Use this for precomputed feature mode.
        """

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.chunk_size = int(chunk_size)
        obs_key = str(observation_key or image_key)

        for ep in episodes:
            obs_seq = ep.obs
            act_seq = action_stats.normalize(np.asarray(ep.actions, dtype=np.float32))
            n = max(0, len(obs_seq) - self.chunk_size + 1)
            for t in range(n):
                obs_t = obs_seq[t]
                visual = np.asarray(obs_t[obs_key])
                if visual.ndim >= 2:
                    visual = visual.astype(np.uint8)
                else:
                    visual = visual.astype(np.float32).reshape(-1)
                parts = [np.asarray(obs_t[k], dtype=np.float32).reshape(-1) for k in proprio_keys]
                prop = np.concatenate(parts, axis=0).astype(np.float32)
                chunk = act_seq[t : t + self.chunk_size].astype(np.float32)
                self.samples.append((visual, prop, chunk))

    def __len__(self) -> int:
        """Return dataset size in samples."""

        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample as model-ready tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(img_t, prop_t, chunk_t)`` with image/proprio/action tensors.
        """

        img, prop, chunk = self.samples[idx]
        img_t = torch.from_numpy(img)
        if img_t.ndim == 3:
            img_t = img_t.float().permute(2, 0, 1) / 255.0
        else:
            img_t = img_t.float()
        prop_t = torch.from_numpy(prop).float()
        chunk_t = torch.from_numpy(chunk).float()
        return img_t, prop_t, chunk_t
