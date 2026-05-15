from __future__ import annotations

"""Dataset prep, curation, and training metadata helpers."""

import json
import os
import random
from typing import Any

import numpy as np
import torch

from mini_pi0.config.schema import RootConfig, to_dict
from mini_pi0.dataset.episodes import EpisodeData
from mini_pi0.models.registry import count_params, pretty_print_model_tree


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_workers(value: int) -> int:
    """Resolve configured DataLoader workers, supporting ``-1`` auto mode."""
    if value is None or int(value) < 0:
        return int(min(4, os.cpu_count() or 1))
    return int(max(0, int(value)))


def infer_action_dim(episodes: list[EpisodeData]) -> int:
    """Infer a consistent action dimension from loaded episodes."""
    dims = {int(np.asarray(ep.actions).shape[-1]) for ep in episodes if np.asarray(ep.actions).ndim == 2}
    if not dims:
        raise ValueError("Unable to infer action dimension from dataset episodes.")
    if len(dims) != 1:
        raise ValueError(f"Inconsistent action dimensions across episodes: {sorted(dims)}")
    return int(next(iter(dims)))


def infer_prop_dim(obs: dict[str, np.ndarray], proprio_keys: list[str]) -> int:
    """Infer concatenated proprio dimension from one observation sample."""
    return int(sum(np.asarray(obs[k], dtype=np.float32).reshape(-1).shape[0] for k in proprio_keys))


def validate_image_observations(obs: dict[str, np.ndarray], image_keys: list[str]) -> None:
    """Validate that configured visual observations are image tensors."""
    visual_parts = [np.asarray(obs[k]) for k in image_keys]
    if not visual_parts:
        raise ValueError("At least one image observation key is required.")
    if not all(v.ndim >= 2 for v in visual_parts):
        raise ValueError(
            "Only raw image observations are supported. "
            f"Shapes: {[tuple(v.shape) for v in visual_parts]}"
        )
    h, w = visual_parts[0].shape[:2]
    c = visual_parts[0].shape[2] if visual_parts[0].ndim >= 3 else 1
    for idx, part in enumerate(visual_parts[1:], start=1):
        part_c = part.shape[2] if part.ndim >= 3 else 1
        if part.shape[:2] != (h, w) or part_c != c:
            raise ValueError(
                "All image_keys must share shape/channels for image fusion. "
                f"Got {visual_parts[0].shape} and {part.shape} at index {idx}."
            )


def split_train_val(dataset: Any, val_ratio: float, seed: int) -> tuple[Any, Any | None]:
    """Split dataset into train and validation subsets."""
    ratio = float(max(0.0, min(0.9, val_ratio)))
    n_total = int(len(dataset))
    if ratio <= 0.0 or n_total < 2:
        return dataset, None

    n_val = int(round(n_total * ratio))
    n_val = max(1, min(n_total - 1, n_val))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n_total).tolist()
    return torch.utils.data.Subset(dataset, indices[n_val:]), torch.utils.data.Subset(dataset, indices[:n_val])


def _infer_progress_key(first_obs: dict[str, np.ndarray], preferred_key: str | None) -> str | None:
    """Pick an observation key used for simple episode progress filtering."""
    candidates: list[str] = []
    if preferred_key:
        candidates.append(str(preferred_key))
    candidates.extend(
        [
            "observation.state.object",
            "object-state",
            "object",
            "observation.state.eef_pos",
            "robot0_eef_pos",
        ]
    )
    for key in candidates:
        if key in first_obs:
            return key
    return None


def _has_nonfinite_episode(ep: EpisodeData) -> bool:
    """Return ``True`` when episode contains NaN/Inf values."""
    if not np.isfinite(np.asarray(ep.actions)).all():
        return True
    for obs_t in ep.obs:
        for v in obs_t.values():
            if not np.isfinite(np.asarray(v)).all():
                return True
    return False


def _episode_progress_delta(ep: EpisodeData, key: str | None) -> float | None:
    """Compute start/end state delta for one episode and key."""
    if key is None or not ep.obs or key not in ep.obs[0] or key not in ep.obs[-1]:
        return None
    a = np.asarray(ep.obs[0][key], dtype=np.float32).reshape(-1)
    b = np.asarray(ep.obs[-1][key], dtype=np.float32).reshape(-1)
    d = min(a.shape[0], b.shape[0])
    if d <= 0:
        return None
    return float(np.linalg.norm(b[:d] - a[:d]))


def curate_episodes(episodes: list[EpisodeData], cfg: RootConfig) -> tuple[list[EpisodeData], dict[str, Any]]:
    """Apply lightweight data-quality filtering for small-data robustness."""
    min_len = int(max(0, getattr(cfg.data, "filter_min_episode_length", 0)))
    min_action_std = float(max(0.0, getattr(cfg.data, "filter_min_action_std", 0.0)))
    min_state_delta = float(max(0.0, getattr(cfg.data, "filter_min_state_delta", 0.0)))
    preferred_state_key = getattr(cfg.data, "filter_state_delta_key", None)
    drop_nan = bool(getattr(cfg.data, "filter_drop_nan", True))

    summary: dict[str, Any] = {
        "enabled": bool(drop_nan or min_len > 0 or min_action_std > 0.0 or min_state_delta > 0.0),
        "before_episodes": int(len(episodes)),
        "after_episodes": int(len(episodes)),
        "removed_episodes": 0,
        "reasons": {},
        "progress_key": None,
        "thresholds": {
            "drop_nan": drop_nan,
            "min_episode_length": min_len,
            "min_action_std": min_action_std,
            "min_state_delta": min_state_delta,
            "preferred_state_delta_key": preferred_state_key,
        },
    }
    if not episodes or not summary["enabled"]:
        return episodes, summary

    progress_key = _infer_progress_key(episodes[0].obs[0], preferred_state_key) if episodes[0].obs else None
    summary["progress_key"] = progress_key
    keep: list[EpisodeData] = []
    reasons: dict[str, int] = {}

    for ep in episodes:
        reason: str | None = None
        t = int(np.asarray(ep.actions).shape[0]) if np.asarray(ep.actions).ndim >= 1 else 0
        if min_len > 0 and t < min_len:
            reason = "short_episode"
        elif drop_nan and _has_nonfinite_episode(ep):
            reason = "non_finite_values"
        elif min_action_std > 0.0:
            act_std = float(np.asarray(ep.actions, dtype=np.float32).std())
            if act_std < min_action_std:
                reason = "low_action_std"

        if reason is None and min_state_delta > 0.0:
            delta = _episode_progress_delta(ep, progress_key)
            if delta is not None and delta < min_state_delta:
                reason = "low_state_progress"

        if reason is None:
            keep.append(ep)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    summary["after_episodes"] = int(len(keep))
    summary["removed_episodes"] = int(len(episodes) - len(keep))
    summary["reasons"] = reasons
    if not keep:
        raise ValueError(
            "Data curation removed all episodes. Relax filters under data.filter_* "
            f"(summary={summary})."
        )
    return keep, summary


def print_train_header(
    cfg: RootConfig,
    resolved_device: torch.device,
    n_episodes: int,
    n_samples: int,
    model: torch.nn.Module,
) -> None:
    """Print resolved config and model summary before training."""
    total_params, trainable_params = count_params(model)
    cfg_dict = to_dict(cfg)
    cfg_dict["train"]["resolved_device"] = str(resolved_device)

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(json.dumps(cfg_dict, indent=2, sort_keys=True))
    print("-" * 80)
    print(f"Episodes loaded      : {n_episodes}")
    print(f"Training samples     : {n_samples}")
    print(f"Model params (total) : {total_params:,}")
    print(f"Model params (train) : {trainable_params:,}")
    print("-" * 80)
    print("Model Architecture (Pretty Tree)")
    print("-" * 80)
    pretty_print_model_tree(model, max_depth=max(0, int(cfg.train.model_print_depth)))
    print("=" * 80)
