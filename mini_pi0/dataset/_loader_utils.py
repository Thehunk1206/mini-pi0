from __future__ import annotations

from typing import Any

import numpy as np


KEY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("observation.images.base_0_rgb", "agentview_image"),
    ("observation.images.right_wrist_0_rgb", "observation.images.wrist_0_rgb", "robot0_eye_in_hand_image"),
    ("observation.state.eef_pos", "robot0_eef_pos"),
    ("observation.state.eef_quat", "robot0_eef_quat"),
    ("observation.state.tool", "robot0_gripper_qpos"),
    ("observation.state.object", "object-state", "object"),
)


def alias_candidates(key: str) -> tuple[str, ...]:
    """Return equivalent key candidates (including ``key``) in preference order."""

    for group in KEY_ALIAS_GROUPS:
        if key in group:
            return group
    return (key,)


def resolve_alias_key(container: Any, key: str) -> str:
    """Resolve a key against known aliases when possible."""

    for candidate in alias_candidates(key):
        if candidate in container:
            return candidate
    return key


def resolve_alias_keys(container: Any, keys: list[str]) -> list[str]:
    """Resolve a list of keys against aliases while preserving order."""

    return [resolve_alias_key(container, key) for key in keys]


def to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of tensor-like values to numpy arrays."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy()  # torch.Tensor-like
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass
    return np.asarray(value)


def extract_key(sample: dict[str, Any], key: str) -> Any:
    """Resolve direct and dotted nested keys, including aliases."""

    direct = resolve_alias_key(sample, key)
    if direct in sample:
        return sample[direct]

    cur: Any = sample
    dotted = resolve_alias_key(sample, key)
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        available = ", ".join(sorted(sample.keys())[:12])
        raise KeyError(
            f"Missing key '{key}' in sample. "
            f"If using LeRobot, set robot/image/proprio keys to match dataset features. "
            f"Top-level keys seen: {available}"
        )
    return cur


def to_uint8_image(arr: np.ndarray, fallback_hw: tuple[int, int]) -> np.ndarray:
    """Convert arbitrary image-like arrays into ``uint8 HxWx3`` tensors."""

    image = np.asarray(arr)
    if image.ndim == 1:
        h, w = fallback_hw
        return np.zeros((h, w, 3), dtype=np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        h, w = fallback_hw
        return np.zeros((h, w, 3), dtype=np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[-1] >= 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def validate_observation_keys(*, image_keys: list[str], proprio_keys: list[str]) -> None:
    """Validate required key lists used by all dataset loaders."""

    if not image_keys:
        raise ValueError("image_keys must contain at least one observation key.")
    if not proprio_keys:
        raise ValueError("proprio_keys must contain at least one observation key.")

