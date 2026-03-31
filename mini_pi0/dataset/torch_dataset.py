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
        image_keys: list[str] | None,
        proprio_keys: list[str],
        action_stats: ActionStats,
        observation_key: str | None = None,
    ):
        """Build dataset samples from episodic demonstrations.

        Args:
            episodes: Canonical demonstration episodes.
            chunk_size: Number of consecutive actions predicted per sample.
            image_key: Backward-compatible single image key.
            image_keys: Optional list of image keys for multi-camera conditioning.
            proprio_keys: Ordered proprioception keys for vector concatenation.
            action_stats: Statistics used to normalize action targets.
            observation_key: Override observation key used as model visual input.
                Use this for precomputed feature mode.
        """

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.chunk_size = int(chunk_size)
        obs_key = str(observation_key or image_key)
        cam_keys = [str(k).strip() for k in (image_keys or []) if str(k).strip()]
        if not cam_keys:
            cam_keys = [str(image_key)]

        for ep in episodes:
            obs_seq = ep.obs
            act_seq = action_stats.normalize(np.asarray(ep.actions, dtype=np.float32))
            n = max(0, len(obs_seq) - self.chunk_size + 1)
            for t in range(n):
                obs_t = obs_seq[t]
                if observation_key is not None:
                    visual = np.asarray(obs_t[obs_key])
                elif len(cam_keys) == 1:
                    visual = np.asarray(obs_t[cam_keys[0]])
                else:
                    visual_parts = [np.asarray(obs_t[k]) for k in cam_keys]
                    if all(v.ndim >= 2 for v in visual_parts):
                        h = visual_parts[0].shape[0]
                        c = visual_parts[0].shape[2] if visual_parts[0].ndim >= 3 else 1
                        for idx, part in enumerate(visual_parts[1:], start=1):
                            part_h = part.shape[0]
                            part_c = part.shape[2] if part.ndim >= 3 else 1
                            if part_h != h or part_c != c:
                                raise ValueError(
                                    "All image_keys must share height and channels for image fusion. "
                                    f"Got {visual_parts[0].shape} and {part.shape} at index {idx}."
                                )
                        # Keep 3 channels by stitching cameras along width.
                        visual = np.concatenate([v.astype(np.uint8) for v in visual_parts], axis=1)
                    elif all(v.ndim == 1 for v in visual_parts):
                        visual = np.concatenate([v.astype(np.float32).reshape(-1) for v in visual_parts], axis=0)
                    else:
                        raise ValueError(
                            "Mixed visual tensor ranks across image_keys are not supported. "
                            f"Shapes: {[tuple(v.shape) for v in visual_parts]}"
                        )
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
