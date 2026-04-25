"""Dataset episode loading facade.

This module intentionally keeps a small public API used across the project:

- :class:`EpisodeData`
- :func:`list_supported_dataset_formats`
- :func:`load_episodes_robomimic`
- :func:`load_episodes_lerobot`
- :func:`iter_lerobot_episode_images`
- :func:`load_episodes_from_config`

Implementation details live in focused helper modules under ``mini_pi0.dataset``.
This keeps external behavior stable while making internals easier to maintain.
"""

from __future__ import annotations

from typing import Any

from mini_pi0.config.schema import effective_image_keys, effective_state_keys
from mini_pi0.dataset._feature_attach import maybe_attach_precomputed_features
from mini_pi0.dataset._lerobot_loader import iter_lerobot_episode_images, load_episodes_lerobot
from mini_pi0.dataset._robomimic_loader import load_episodes_robomimic
from mini_pi0.dataset.types import EpisodeData


SUPPORTED_DATASET_FORMATS = ("robomimic_hdf5", "lerobot_hf")


def list_supported_dataset_formats() -> list[str]:
    """Return dataset format identifiers supported by the data loader stack."""

    return list(SUPPORTED_DATASET_FORMATS)


def _normalize_dataset_format(fmt: str) -> str:
    """Normalize user-facing format aliases into canonical format ids."""

    value = str(fmt).strip().lower()
    if value in {"robomimic", "robomimic_hdf5", "hdf5"}:
        return "robomimic_hdf5"
    if value in {"lerobot", "lerobot_hf", "hf"}:
        return "lerobot_hf"
    return value


def _resolve_fallback_image_hw(cfg: Any) -> tuple[int, int]:
    """Resolve and validate ``data.fallback_image_hw`` from config."""

    hw = tuple(cfg.data.fallback_image_hw)
    if len(hw) != 2:
        raise ValueError("data.fallback_image_hw must be [H, W]")
    return int(hw[0]), int(hw[1])


def _load_from_robomimic(cfg: Any, *, image_keys: list[str], state_keys: list[str], fallback_hw: tuple[int, int]) -> list[EpisodeData]:
    """Load episodes from robomimic based on config."""

    hdf5_path = cfg.data.robomimic_hdf5
    if not hdf5_path:
        raise ValueError("data.robomimic_hdf5 must be set when data.format=robomimic_hdf5")
    return load_episodes_robomimic(
        hdf5_path=hdf5_path,
        image_keys=image_keys,
        proprio_keys=state_keys,
        limit=cfg.data.n_demos,
        data_group=cfg.data.robomimic_data_group,
        fallback_image_hw=fallback_hw,
    )


def _load_from_lerobot(cfg: Any, *, image_keys: list[str], state_keys: list[str], fallback_hw: tuple[int, int]) -> list[EpisodeData]:
    """Load episodes from LeRobot based on config."""

    repo_id = cfg.data.lerobot_repo_id
    if not repo_id:
        raise ValueError("data.lerobot_repo_id must be set when data.format=lerobot_hf")
    obs_mode = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    load_images = obs_mode not in {"precomputed", "feature", "features"}
    return load_episodes_lerobot(
        repo_id=repo_id,
        image_keys=image_keys,
        proprio_keys=state_keys,
        action_key=cfg.data.lerobot_action_key,
        episode_index_key=cfg.data.lerobot_episode_index_key,
        limit=cfg.data.n_demos,
        fallback_image_hw=fallback_hw,
        local_files_only=bool(cfg.data.lerobot_local_files_only),
        video_backend=cfg.data.lerobot_video_backend,
        load_images=load_images,
    )


def load_episodes_from_config(cfg: Any) -> list[EpisodeData]:
    """Load episodes according to typed repository config.

    Args:
        cfg: Root configuration with ``data`` and ``robot`` sections populated.

    Returns:
        List of episodes in canonical schema.

    Raises:
        ValueError: If dataset format is unsupported or required fields are missing.
    """

    normalized_format = _normalize_dataset_format(getattr(cfg.data, "format", "robomimic_hdf5"))
    state_keys = effective_state_keys(cfg.robot)
    image_keys = effective_image_keys(cfg.robot)
    fallback_hw = _resolve_fallback_image_hw(cfg)

    if normalized_format == "robomimic_hdf5":
        episodes = _load_from_robomimic(cfg, image_keys=image_keys, state_keys=state_keys, fallback_hw=fallback_hw)
        return maybe_attach_precomputed_features(episodes, cfg)

    if normalized_format == "lerobot_hf":
        episodes = _load_from_lerobot(cfg, image_keys=image_keys, state_keys=state_keys, fallback_hw=fallback_hw)
        return maybe_attach_precomputed_features(episodes, cfg)

    raise ValueError(
        f"Unsupported data.format '{cfg.data.format}'. "
        f"Supported formats: {', '.join(list_supported_dataset_formats())}"
    )
