from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_pi0.dataset.episodes import EpisodeData
from mini_pi0.dataset.image_transforms import resize_image_tensor
from mini_pi0.dataset.stats import ActionStats, StateStats


class ActionChunkDataset(Dataset):
    """Torch dataset of observation + fixed-horizon action chunks.

    Each sample uses observation at timestep ``t`` and predicts normalized
    action sequence ``[t, t + chunk_size)``.
    """

    actions_are_normalized = True

    def __init__(
        self,
        episodes: list[EpisodeData],
        chunk_size: int,
        image_key: str,
        image_keys: list[str] | None,
        proprio_keys: list[str],
        action_stats: ActionStats,
        obs_horizon: int = 1,
        preserve_camera_dim: bool = False,
        state_stats: StateStats | None = None,
        normalize_states: bool = False,
        image_resize_hw: list[int] | None = None,
        image_resize_mode: str = "letterbox",
        sample_stride: int = 1,
    ):
        """Build dataset samples from episodic demonstrations.

        Args:
            episodes: Canonical demonstration episodes.
            chunk_size: Number of consecutive actions predicted per sample.
            image_key: Backward-compatible single image key.
            image_keys: Optional list of image keys for multi-camera conditioning.
            proprio_keys: Ordered proprioception keys for vector concatenation.
            action_stats: Statistics used to normalize action targets.
            obs_horizon: Number of current/past observations to include.
                ``1`` preserves legacy single-observation samples.
            preserve_camera_dim: Keep image cameras as a separate axis instead
                of legacy width-stitching.
        """

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.chunk_size = int(chunk_size)
        self.obs_horizon = int(max(1, obs_horizon))
        self.preserve_camera_dim = bool(preserve_camera_dim)
        self.sample_stride = int(max(1, sample_stride))
        self.states_are_normalized = bool(normalize_states)
        cam_keys = [str(k).strip() for k in (image_keys or []) if str(k).strip()]
        if not cam_keys:
            cam_keys = [str(image_key)]

        def _resize_hwc(value: np.ndarray) -> np.ndarray:
            arr = np.asarray(value)
            if arr.ndim != 3 or arr.shape[-1] not in {1, 3, 4}:
                raise ValueError(f"Only HWC images are supported, got shape {arr.shape}.")
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            tensor = resize_image_tensor(tensor, image_resize_hw, image_resize_mode)
            return tensor.permute(1, 2, 0).contiguous().numpy()

        def _visual_at(obs_t: dict[str, np.ndarray]) -> np.ndarray:
            visual_parts = [_resize_hwc(np.asarray(obs_t[k])) for k in cam_keys]
            if len(cam_keys) == 1:
                visual_arr = visual_parts[0]
                if self.preserve_camera_dim and visual_arr.ndim >= 2:
                    visual_arr = visual_arr[None, ...]
            elif self.preserve_camera_dim:
                visual_arr = np.stack(visual_parts, axis=0)
            else:
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
                    visual_arr = np.concatenate([v.astype(np.uint8) for v in visual_parts], axis=1)
                else:
                    raise ValueError(
                        "Only raw image observations are supported. "
                        f"Shapes: {[tuple(v.shape) for v in visual_parts]}"
                    )
            if visual_arr.ndim >= 2:
                return visual_arr.astype(np.uint8)
            raise ValueError(f"Only raw image observations are supported, got shape {visual_arr.shape}.")

        def _prop_at(obs_t: dict[str, np.ndarray]) -> np.ndarray:
            parts = [np.asarray(obs_t[k], dtype=np.float32).reshape(-1) for k in proprio_keys]
            prop = np.concatenate(parts, axis=0).astype(np.float32)
            if normalize_states:
                if state_stats is None:
                    raise ValueError("state_stats are required when normalize_states=true.")
                prop = state_stats.normalize(prop).astype(np.float32)
            return prop

        for ep in episodes:
            obs_seq = ep.obs
            act_seq = action_stats.normalize(np.asarray(ep.actions, dtype=np.float32))
            n = max(0, len(obs_seq) - self.chunk_size + 1)
            for t in range(0, n, self.sample_stride):
                hist_indices = [max(0, t - self.obs_horizon + 1 + offset) for offset in range(self.obs_horizon)]
                visual_hist = [_visual_at(obs_seq[idx]) for idx in hist_indices]
                prop_hist = [_prop_at(obs_seq[idx]) for idx in hist_indices]
                if self.obs_horizon == 1:
                    visual = visual_hist[0]
                    prop = prop_hist[0]
                else:
                    visual = np.stack(visual_hist, axis=0)
                    prop = np.stack(prop_hist, axis=0).astype(np.float32)
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
        elif img_t.ndim == 4 and img_t.shape[-1] in {1, 3, 4}:
            img_t = img_t.float().permute(0, 3, 1, 2) / 255.0
        elif img_t.ndim == 5:
            img_t = img_t.float().permute(0, 1, 4, 2, 3) / 255.0
        else:
            img_t = img_t.float()
        prop_t = torch.from_numpy(prop).float()
        chunk_t = torch.from_numpy(chunk).float()
        return img_t, prop_t, chunk_t
