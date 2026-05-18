from __future__ import annotations

"""Training-time augmentation utilities."""

import torch

from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.stats import ActionStats


def _random_resized_crop_batch(img: torch.Tensor, scale_min: float) -> torch.Tensor:
    """Apply per-sample random crop then resize back to original resolution."""
    bsz, _, height, width = img.shape
    scale = float(max(0.1, min(1.0, scale_min)))
    if scale >= 1.0:
        return img

    crop_h = max(1, int(round(height * scale)))
    crop_w = max(1, int(round(width * scale)))
    max_y = max(0, height - crop_h)
    max_x = max(0, width - crop_w)

    crops: list[torch.Tensor] = []
    for i in range(bsz):
        y0 = int(torch.randint(0, max_y + 1, (1,), device=img.device).item()) if max_y > 0 else 0
        x0 = int(torch.randint(0, max_x + 1, (1,), device=img.device).item()) if max_x > 0 else 0
        crop = img[i : i + 1, :, y0 : y0 + crop_h, x0 : x0 + crop_w]
        crop = torch.nn.functional.interpolate(
            crop,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(crop)
    return torch.cat(crops, dim=0)


def augment_image_batch(img: torch.Tensor, cfg: RootConfig) -> torch.Tensor:
    """Apply lightweight image augmentations to training image batches."""
    if img.ndim != 4:
        return img
    if not bool(getattr(cfg.train, "image_aug_enable", False)):
        return img

    out = img
    crop_scale = float(getattr(cfg.train, "image_aug_crop_scale", 1.0))
    if crop_scale < 1.0:
        out = _random_resized_crop_batch(out, crop_scale)

    bsz = out.shape[0]
    brightness = float(max(0.0, getattr(cfg.train, "image_aug_brightness", 0.0)))
    contrast = float(max(0.0, getattr(cfg.train, "image_aug_contrast", 0.0)))
    saturation = float(max(0.0, getattr(cfg.train, "image_aug_saturation", 0.0)))

    if brightness > 0.0:
        factor = 1.0 + (torch.rand((bsz, 1, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * brightness
        out = out * factor
    if contrast > 0.0:
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        factor = 1.0 + (torch.rand((bsz, 1, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * contrast
        out = (out - mean) * factor + mean
    if saturation > 0.0:
        gray = out.mean(dim=1, keepdim=True)
        factor = 1.0 + (torch.rand((bsz, 1, 1, 1), device=out.device, dtype=out.dtype) * 2.0 - 1.0) * saturation
        out = gray + (out - gray) * factor
    return out.clamp(0.0, 1.0)


def augment_actions(actions: torch.Tensor, cfg: RootConfig) -> torch.Tensor:
    """Inject Gaussian noise into normalized action targets."""
    std = float(max(0.0, getattr(cfg.train, "action_noise_std", 0.0)))
    if std <= 0.0:
        return actions

    out = actions + torch.randn_like(actions) * std
    clip = float(max(0.0, getattr(cfg.train, "action_noise_clip", 0.0)))
    if clip > 0.0:
        out = out.clamp(-clip, clip)
    return out


class GpuBatchProcessor:
    """Move batches to a device and run tensor-only preprocessing there."""

    def __init__(
        self,
        *,
        cfg: RootConfig,
        device: torch.device,
        action_stats: ActionStats | None,
        normalize_actions: bool,
    ) -> None:
        """Create a GPU-aware batch processor.

        Args:
            cfg: Root training config.
            device: Target training device.
            action_stats: Optional action normalization statistics.
            normalize_actions: Whether incoming action chunks are raw actions.
        """

        self.cfg = cfg
        self.device = device
        self.normalize_actions = bool(normalize_actions)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if action_stats is not None:
            self.mean = torch.as_tensor(action_stats.mean, device=device, dtype=torch.float32)
            self.std = torch.as_tensor(action_stats.std, device=device, dtype=torch.float32)

    def train_batch(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transfer and preprocess a training batch."""

        img, prop, actions = self._transfer(img, prop, actions)
        img = self._prepare_images(img)
        actions = self._prepare_actions(actions)
        img = augment_image_batch(img, self.cfg)
        actions = augment_actions(actions, self.cfg)
        return img, prop.float(), actions

    def eval_batch(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transfer and preprocess a validation batch without augmentation."""

        img, prop, actions = self._transfer(img, prop, actions)
        return self._prepare_images(img), prop.float(), self._prepare_actions(actions)

    def _transfer(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Asynchronously transfer tensors when pinned memory is available."""

        non_blocking = self.device.type == "cuda"
        return (
            img.to(self.device, non_blocking=non_blocking),
            prop.to(self.device, non_blocking=non_blocking),
            actions.to(self.device, non_blocking=non_blocking),
        )

    def _prepare_images(self, img: torch.Tensor) -> torch.Tensor:
        """Convert image tensors to contiguous float data on device."""

        if img.dtype == torch.uint8:
            return img.float().div_(255.0).contiguous()
        return img.float().contiguous()

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize raw action chunks on device when needed."""

        out = actions.float()
        if not self.normalize_actions:
            return out
        if self.mean is None or self.std is None:
            raise RuntimeError("Action stats are required to normalize raw action chunks.")
        return (out - self.mean.view(*([1] * (out.ndim - 1)), -1)) / self.std.view(*([1] * (out.ndim - 1)), -1)
