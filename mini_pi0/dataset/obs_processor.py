from __future__ import annotations

import numpy as np
import torch
from collections import deque
from collections.abc import Iterable

from mini_pi0.dataset.stats import ActionStats
from mini_pi0.utils.device import resolve_device


class ObsProcessor:
    """Convert raw environment observations/actions to model-ready tensors.

    The processor also owns action normalization tensors loaded from
    ``ActionStats`` so inference uses the same scaling as training.
    """

    def __init__(
        self,
        action_stats_path: str,
        image_key: str | None,
        image_keys: list[str] | None,
        proprio_keys: list[str],
        device: str = "auto",
        obs_horizon: int = 1,
        preserve_camera_dim: bool = False,
    ):
        """Initialize observation processor and action normalization tensors.

        Args:
            action_stats_path: JSON path containing ``mean`` and ``std`` action stats.
            image_key: Backward-compatible single observation key for image tensors.
            image_keys: Optional ordered image observation keys for multi-camera input.
            proprio_keys: Ordered proprioception keys concatenated into one vector.
            device: Torch device string (``auto``, ``cpu``, ``cuda``, ``mps``).
            obs_horizon: Number of current/past observations to feed the model.
            preserve_camera_dim: Keep cameras as an explicit tensor axis instead
                of legacy width stitching.
        """

        self.device = resolve_device(device)
        keys = [str(k).strip() for k in (image_keys or []) if str(k).strip()]
        if not keys:
            if image_key is None:
                raise ValueError("ObsProcessor requires image_key or image_keys.")
            keys = [str(image_key)]
        self.image_keys = keys
        self.image_key = keys[0]
        self.proprio_keys = proprio_keys
        self.obs_horizon = int(max(1, obs_horizon))
        self.preserve_camera_dim = bool(preserve_camera_dim)
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self.obs_horizon)
        self._batch_history: dict[int, deque[dict[str, np.ndarray]]] = {}
        stats = ActionStats.load(action_stats_path)
        self.action_mean = torch.tensor(stats.mean, dtype=torch.float32, device=self.device)
        self.action_std = torch.tensor(stats.std, dtype=torch.float32, device=self.device)

    def reset_history(self, obs: dict[str, np.ndarray]) -> None:
        """Reset sequential rollout history using repeat padding."""

        self._history.clear()
        for _ in range(self.obs_horizon):
            self._history.append(obs)

    def reset_batch_history(self, obs_batch: list[dict[str, np.ndarray]]) -> None:
        """Reset vectorized rollout histories using repeat padding."""

        self._batch_history = {}
        for idx, obs in enumerate(obs_batch):
            hist: deque[dict[str, np.ndarray]] = deque(maxlen=self.obs_horizon)
            for _ in range(self.obs_horizon):
                hist.append(obs)
            self._batch_history[int(idx)] = hist

    def _single_obs_to_arrays(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Convert one observation into unbatched visual/proprio arrays."""

        imgs = [np.asarray(obs[key], dtype=np.uint8) for key in self.image_keys]
        if len(imgs) > 1:
            h = imgs[0].shape[0]
            c = imgs[0].shape[2] if imgs[0].ndim >= 3 else 1
            for idx, part in enumerate(imgs[1:], start=1):
                part_h = part.shape[0]
                part_c = part.shape[2] if part.ndim >= 3 else 1
                if part_h != h or part_c != c:
                    raise ValueError(
                        "All image_keys must share height and channels for image fusion. "
                        f"Got {imgs[0].shape} and {part.shape} at index {idx}."
                    )
        if self.preserve_camera_dim:
            visual = np.stack(imgs, axis=0)
        else:
            visual = np.concatenate(imgs, axis=1)

        prop = np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.proprio_keys],
            axis=0,
        )
        return visual, prop

    def _history_to_tensors(self, history: Iterable[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert an observation history into batched model tensors."""

        visual_hist: list[np.ndarray] = []
        prop_hist: list[np.ndarray] = []
        for obs in history:
            visual, prop = self._single_obs_to_arrays(obs)
            visual_hist.append(visual)
            prop_hist.append(prop)

        if self.obs_horizon == 1:
            visual_arr = visual_hist[-1]
            prop_arr = prop_hist[-1]
        else:
            visual_arr = np.stack(visual_hist, axis=0)
            prop_arr = np.stack(prop_hist, axis=0).astype(np.float32)

        img = torch.from_numpy(np.asarray(visual_arr))
        if img.ndim == 3:
            img = img.float().permute(2, 0, 1).unsqueeze(0) / 255.0
        elif img.ndim == 4 and img.shape[-1] in {1, 3, 4}:
            img = img.float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        elif img.ndim == 5:
            img = img.float().permute(0, 1, 4, 2, 3).unsqueeze(0) / 255.0
        else:
            img = img.float().reshape(1, *img.shape)

        prop = torch.from_numpy(np.asarray(prop_arr, dtype=np.float32)).float().unsqueeze(0)
        return img.to(self.device), prop.to(self.device)

    def obs_to_tensors(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert one raw observation dict to batched model input tensors.

        Args:
            obs: Canonical observation dictionary.

        Returns:
            Tuple ``(img, prop)`` where:
            - ``img`` is ``[1, 3, H, W]`` float tensor in ``[0, 1]``
            - ``prop`` is ``[1, P]`` float tensor of concatenated proprio values.
        """

        if not self._history:
            self.reset_history(obs)
        else:
            self._history.append(obs)
        return self._history_to_tensors(self._history)

    def obs_batch_to_tensors(
        self,
        obs_batch: list[dict[str, np.ndarray]],
        env_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of canonical observations to model input tensors.

        Args:
            obs_batch: List of canonical observation dictionaries, one per
                simulator sub-environment.

        Returns:
            Tuple ``(img, prop)`` where the leading dimension is batch size.
        """

        if not obs_batch:
            raise ValueError("obs_batch_to_tensors requires at least one observation.")
        imgs: list[torch.Tensor] = []
        props: list[torch.Tensor] = []
        if env_indices is None:
            env_indices = list(range(len(obs_batch)))
        if len(env_indices) != len(obs_batch):
            raise ValueError("env_indices length must match obs_batch length.")
        for env_idx, obs in zip(env_indices, obs_batch, strict=True):
            hist = self._batch_history.get(int(env_idx))
            if hist is None:
                hist = deque(maxlen=self.obs_horizon)
                for _ in range(self.obs_horizon):
                    hist.append(obs)
                self._batch_history[int(env_idx)] = hist
            else:
                hist.append(obs)
            img, prop = self._history_to_tensors(hist)
            imgs.append(img)
            props.append(prop)
        return torch.cat(imgs, dim=0), torch.cat(props, dim=0)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized model actions back to environment action scale.

        Args:
            actions: Normalized action tensor.

        Returns:
            Denormalized action tensor.
        """

        return actions * self.action_std + self.action_mean

    def clip(self, actions: torch.Tensor, low: np.ndarray, high: np.ndarray) -> torch.Tensor:
        """Clip actions to simulator action bounds.

        Args:
            actions: Action tensor to clip.
            low: Per-dimension lower bounds.
            high: Per-dimension upper bounds.

        Returns:
            Clipped action tensor.
        """

        lo = torch.tensor(low, device=self.device, dtype=torch.float32)
        hi = torch.tensor(high, device=self.device, dtype=torch.float32)
        return torch.clamp(actions, lo, hi)
