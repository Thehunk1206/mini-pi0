from __future__ import annotations

import numpy as np
import torch

from mini_pi0.dataset.stats import ActionStats
from mini_pi0.utils.device import resolve_device
from mini_pi0.vision.encoders import VisionFeatureExtractor, images_to_tensor


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
        observation_mode: str = "image",
        feature_key: str = "vision_feat",
        feature_extractor: VisionFeatureExtractor | None = None,
    ):
        """Initialize observation processor and action normalization tensors.

        Args:
            action_stats_path: JSON path containing ``mean`` and ``std`` action stats.
            image_key: Backward-compatible single observation key for image tensors.
            image_keys: Optional ordered image observation keys for multi-camera input.
            proprio_keys: Ordered proprioception keys concatenated into one vector.
            device: Torch device string (``auto``, ``cpu``, ``cuda``, ``mps``).
            observation_mode: ``image`` or ``precomputed``.
            feature_key: Observation key used to read precomputed features from obs dict.
            feature_extractor: Optional runtime encoder used when model expects features.
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
        self.observation_mode = str(observation_mode).strip().lower()
        self.feature_key = feature_key
        self.feature_extractor = feature_extractor
        stats = ActionStats.load(action_stats_path)
        self.action_mean = torch.tensor(stats.mean, dtype=torch.float32, device=self.device)
        self.action_std = torch.tensor(stats.std, dtype=torch.float32, device=self.device)

    def _encode_runtime_features(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        """Encode one or multiple camera images with runtime feature extractor."""

        if self.feature_extractor is None:
            raise KeyError(
                f"Observation key '{self.feature_key}' missing and no runtime feature extractor configured."
            )
        imgs = [np.asarray(obs[key], dtype=np.uint8) for key in self.image_keys]
        x = images_to_tensor(imgs, device=self.device)
        with torch.no_grad():
            feats = self.feature_extractor(x).float()
        if feats.ndim != 2:
            raise ValueError(f"Runtime extractor output must be 2D [N,D], got shape {tuple(feats.shape)}")
        if feats.shape[0] != len(self.image_keys):
            raise ValueError(
                f"Runtime extractor batch mismatch: expected {len(self.image_keys)} features, got {feats.shape[0]}"
            )
        return feats.reshape(1, -1)

    def obs_to_tensors(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert one raw observation dict to batched model input tensors.

        Args:
            obs: Canonical observation dictionary.

        Returns:
            Tuple ``(img, prop)`` where:
            - ``img`` is ``[1, 3, H, W]`` float tensor in ``[0, 1]``
            - ``prop`` is ``[1, P]`` float tensor of concatenated proprio values.
        """

        if self.observation_mode in {"precomputed", "feature", "features"}:
            if self.feature_key in obs:
                img = torch.from_numpy(np.asarray(obs[self.feature_key], dtype=np.float32)).float().unsqueeze(0)
            else:
                img = self._encode_runtime_features(obs)
        else:
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
            merged = np.concatenate(imgs, axis=1)
            img = torch.from_numpy(merged).float()
            img = img.permute(2, 0, 1).unsqueeze(0) / 255.0

        prop = np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.proprio_keys],
            axis=0,
        )
        prop = torch.from_numpy(prop).float().unsqueeze(0)

        return img.to(self.device), prop.to(self.device)

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
