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
        image_key: str,
        proprio_keys: list[str],
        device: str = "auto",
        observation_mode: str = "image",
        feature_key: str = "vision_feat",
        feature_extractor: VisionFeatureExtractor | None = None,
    ):
        """Initialize observation processor and action normalization tensors.

        Args:
            action_stats_path: JSON path containing ``mean`` and ``std`` action stats.
            image_key: Observation key used for image tensors.
            proprio_keys: Ordered proprioception keys concatenated into one vector.
            device: Torch device string (``auto``, ``cpu``, ``cuda``, ``mps``).
            observation_mode: ``image`` or ``precomputed``.
            feature_key: Observation key used to read precomputed features from obs dict.
            feature_extractor: Optional runtime encoder used when model expects features.
        """

        self.device = resolve_device(device)
        self.image_key = image_key
        self.proprio_keys = proprio_keys
        self.observation_mode = str(observation_mode).strip().lower()
        self.feature_key = feature_key
        self.feature_extractor = feature_extractor
        stats = ActionStats.load(action_stats_path)
        self.action_mean = torch.tensor(stats.mean, dtype=torch.float32, device=self.device)
        self.action_std = torch.tensor(stats.std, dtype=torch.float32, device=self.device)

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
                if self.feature_extractor is None:
                    raise KeyError(
                        f"Observation key '{self.feature_key}' missing and no runtime feature extractor configured."
                    )
                raw = np.asarray(obs[self.image_key], dtype=np.uint8)
                x = images_to_tensor([raw], device=self.device)
                with torch.no_grad():
                    img = self.feature_extractor(x).float()
        else:
            img = torch.from_numpy(np.asarray(obs[self.image_key], dtype=np.uint8)).float()
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
