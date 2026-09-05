from __future__ import annotations

"""Shared image geometry transforms for training and deployment parity."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def normalize_image_hw(value: Sequence[int] | None) -> tuple[int, int] | None:
    """Validate and normalize an optional ``[height, width]`` image size."""

    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"image_resize_hw must contain [height, width], got {list(value)!r}.")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_resize_hw values must be positive, got {[height, width]!r}.")
    return height, width


def resize_image_tensor(
    image: torch.Tensor,
    target_hw: Sequence[int] | None,
    mode: str = "letterbox",
) -> torch.Tensor:
    """Resize ``[..., C, H, W]`` images while preserving leading dimensions.

    ``letterbox`` preserves the complete field of view and uses replicated edge
    pixels for padding. ``center_crop`` preserves aspect ratio but crops the
    source to the target aspect ratio. ``stretch`` directly resizes both axes.
    """

    target = normalize_image_hw(target_hw)
    if target is None:
        return image
    if image.ndim < 3:
        raise ValueError(f"Expected image tensor shaped [..., C, H, W], got {tuple(image.shape)}.")

    target_h, target_w = target
    source_h, source_w = int(image.shape[-2]), int(image.shape[-1])
    if (source_h, source_w) == target:
        return image

    original_dtype = image.dtype
    flat = image.reshape(-1, *image.shape[-3:]).float()
    resize_mode = str(mode or "letterbox").strip().lower()

    if resize_mode == "stretch":
        out = F.interpolate(flat, size=target, mode="bilinear", align_corners=False)
    elif resize_mode == "center_crop":
        source_ratio = source_w / source_h
        target_ratio = target_w / target_h
        if source_ratio > target_ratio:
            crop_w = max(1, int(round(source_h * target_ratio)))
            left = max(0, (source_w - crop_w) // 2)
            flat = flat[..., :, left : left + crop_w]
        elif source_ratio < target_ratio:
            crop_h = max(1, int(round(source_w / target_ratio)))
            top = max(0, (source_h - crop_h) // 2)
            flat = flat[..., top : top + crop_h, :]
        out = F.interpolate(flat, size=target, mode="bilinear", align_corners=False)
    elif resize_mode == "letterbox":
        scale = min(target_h / source_h, target_w / source_w)
        resized_h = max(1, min(target_h, int(round(source_h * scale))))
        resized_w = max(1, min(target_w, int(round(source_w * scale))))
        out = F.interpolate(flat, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        out = F.pad(out, (left, right, top, bottom), mode="replicate")
    else:
        raise ValueError("image_resize_mode must be 'letterbox', 'center_crop', or 'stretch'.")

    out = out.reshape(*image.shape[:-3], *out.shape[-3:])
    if original_dtype == torch.uint8:
        return out.round().clamp_(0, 255).to(dtype=original_dtype)
    return out.to(dtype=original_dtype)


__all__ = ["normalize_image_hw", "resize_image_tensor"]
