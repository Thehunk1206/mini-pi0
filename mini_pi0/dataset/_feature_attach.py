from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from mini_pi0.dataset.types import EpisodeData


def _validate_feature_matrix(*, features: np.ndarray, feature_id: str, expected_steps: int) -> np.ndarray:
    """Validate shape compatibility for one episode feature matrix."""

    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array for '{feature_id}', got shape {arr.shape}")
    if len(arr) != expected_steps:
        raise ValueError(
            f"Feature length mismatch for '{feature_id}': features={len(arr)} vs episode_obs={expected_steps}"
        )
    return arr


def _attach_feature_matrix_to_episode(
    *,
    episode: EpisodeData,
    feature_matrix: np.ndarray,
    feature_key: str,
) -> None:
    """Attach one per-timestep feature matrix to one episode in place."""

    for t, obs_t in enumerate(episode.obs):
        obs_t[feature_key] = feature_matrix[t]


def _attach_from_feature_directory(
    *,
    episodes: list[EpisodeData],
    feature_dir: Path,
    feature_key: str,
) -> list[EpisodeData]:
    """Attach features from ``ep_XXXXXX.npy`` files inside a directory."""

    print(f"[data] Attaching precomputed features from directory: {feature_dir}", flush=True)
    for ep_idx, episode in enumerate(episodes):
        episode_key = f"ep_{ep_idx:06d}"
        episode_file = feature_dir / f"{episode_key}.npy"
        if not episode_file.exists():
            raise KeyError(
                f"Missing feature file '{episode_file}'. "
                "Re-run precompute for the same dataset ordering/limit."
            )
        features = _validate_feature_matrix(
            features=np.load(episode_file),
            feature_id=str(episode_file),
            expected_steps=len(episode.obs),
        )
        _attach_feature_matrix_to_episode(
            episode=episode,
            feature_matrix=features,
            feature_key=feature_key,
        )
        if (ep_idx + 1) % 20 == 0 or (ep_idx + 1) == len(episodes):
            print(f"[data] Attached features for {ep_idx + 1}/{len(episodes)} episodes", flush=True)
    return episodes


def _attach_from_feature_archive(
    *,
    episodes: list[EpisodeData],
    feature_path: str,
    feature_key: str,
) -> list[EpisodeData]:
    """Attach features from `.npz` archive keyed by ``ep_XXXXXX``."""

    print(f"[data] Attaching precomputed features from archive: {feature_path}", flush=True)
    with np.load(feature_path) as store:
        for ep_idx, episode in enumerate(episodes):
            episode_key = f"ep_{ep_idx:06d}"
            if episode_key not in store:
                raise KeyError(
                    f"Missing feature array '{episode_key}' in {feature_path}. "
                    "Re-run precompute for the same dataset ordering/limit."
                )
            features = _validate_feature_matrix(
                features=store[episode_key],
                feature_id=episode_key,
                expected_steps=len(episode.obs),
            )
            _attach_feature_matrix_to_episode(
                episode=episode,
                feature_matrix=features,
                feature_key=feature_key,
            )
            if (ep_idx + 1) % 20 == 0 or (ep_idx + 1) == len(episodes):
                print(f"[data] Attached features for {ep_idx + 1}/{len(episodes)} episodes", flush=True)
    return episodes


def maybe_attach_precomputed_features(episodes: list[EpisodeData], cfg: Any) -> list[EpisodeData]:
    """Attach cached vision features to loaded episodes when requested.

    The expected format stores per-episode arrays under keys:
    ``ep_000000``, ``ep_000001``, ...
    """

    mode = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    if mode not in {"precomputed", "feature", "features"}:
        return episodes

    feature_path = getattr(cfg.data, "precomputed_features_path", None)
    if not feature_path:
        raise ValueError("data.precomputed_features_path must be set when using precomputed observation mode.")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Precomputed features not found: {feature_path}")

    feature_key = str(getattr(cfg.data, "precomputed_feature_key", "vision_feat"))
    feature_path_obj = Path(feature_path)
    if feature_path_obj.is_dir():
        return _attach_from_feature_directory(
            episodes=episodes,
            feature_dir=feature_path_obj,
            feature_key=feature_key,
        )
    return _attach_from_feature_archive(
        episodes=episodes,
        feature_path=feature_path,
        feature_key=feature_key,
    )
