from __future__ import annotations

from importlib import import_module
import inspect
from typing import Any, Iterator

import numpy as np

from mini_pi0.dataset._loader_utils import extract_key, to_numpy, to_uint8_image, validate_observation_keys
from mini_pi0.dataset.types import EpisodeData


def _import_lerobot_dataset_class():
    """Import ``LeRobotDataset`` from supported module paths."""

    candidates = [
        ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),
        ("lerobot.common.datasets.lerobot_dataset", "LeRobotDataset"),
    ]
    errors: list[str] = []
    for mod_name, class_name in candidates:
        try:
            mod = import_module(mod_name)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
            errors.append(f"{mod_name}:{class_name} not found")
        except Exception as e:
            errors.append(f"{mod_name} import failed: {type(e).__name__}: {e}")
    raise RuntimeError(
        "LeRobot dataset support requires the `lerobot` package. "
        "Install it (for example: `pip install lerobot`) and retry. "
        f"Import attempts: {' | '.join(errors)}"
    )


def make_lerobot_dataset(repo_id: str, local_files_only: bool, video_backend: str | None = None) -> Any:
    """Instantiate a LeRobot dataset with compatible constructor fallbacks."""

    cls = _import_lerobot_dataset_class()
    try:
        sig = inspect.signature(cls)
        supported = set(sig.parameters.keys())
    except Exception:
        supported = set()

    kwargs: dict[str, Any] = {}
    if "repo_id" in supported or not supported:
        kwargs["repo_id"] = repo_id
    if "split" in supported:
        kwargs["split"] = "train"
    if "local_files_only" in supported:
        kwargs["local_files_only"] = bool(local_files_only)
    if video_backend and "video_backend" in supported:
        kwargs["video_backend"] = str(video_backend)

    attempts = [
        lambda: cls(**kwargs) if kwargs else cls(repo_id=repo_id),
        lambda: cls(repo_id=repo_id, split="train"),
        lambda: cls(repo_id=repo_id),
        lambda: cls(repo_id),
    ]
    errors: list[str] = []
    for builder in attempts:
        try:
            dataset = builder()
            if video_backend and hasattr(dataset, "video_backend"):
                try:
                    dataset.video_backend = str(video_backend)
                except Exception:
                    pass
            return dataset
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise RuntimeError(
        f"Failed to create LeRobotDataset for repo '{repo_id}'. "
        f"Tried multiple constructor signatures. Errors: {' | '.join(errors)}"
    )


def _extract_episode_index(sample: dict[str, Any], episode_index_key: str, sample_index: int) -> int:
    """Extract and validate scalar episode index from one sample."""

    episode_raw = extract_key(sample, episode_index_key)
    episode_flat = to_numpy(episode_raw).reshape(-1)
    if episode_flat.size != 1:
        raise ValueError(
            f"LeRobot episode index key '{episode_index_key}' must be scalar. "
            f"Got shape {tuple(episode_flat.shape)} at sample {sample_index}."
        )
    return int(episode_flat.item())


def _extract_obs_t(
    *,
    sample: dict[str, Any],
    image_keys: list[str],
    proprio_keys: list[str],
    fallback_image_hw: tuple[int, int],
    load_images: bool,
) -> dict[str, np.ndarray]:
    """Extract one timestep observation dict in canonical key space."""

    obs_t: dict[str, np.ndarray] = {}
    if load_images:
        h, w = fallback_image_hw
        for key in image_keys:
            image = to_uint8_image(to_numpy(extract_key(sample, key)), fallback_hw=(h, w))
            obs_t[key] = image
    for key in proprio_keys:
        obs_t[key] = np.asarray(to_numpy(extract_key(sample, key)), dtype=np.float32).reshape(-1)
    return obs_t


def load_episodes_lerobot(
    repo_id: str,
    image_keys: list[str],
    proprio_keys: list[str],
    action_key: str = "action",
    episode_index_key: str = "episode_index",
    limit: int | None = None,
    fallback_image_hw: tuple[int, int] = (84, 84),
    local_files_only: bool = False,
    video_backend: str | None = "pyav",
    load_images: bool = True,
) -> list[EpisodeData]:
    """Load demonstrations from a native Hugging Face LeRobot dataset."""

    if not repo_id:
        raise ValueError("LeRobot repo_id must be a non-empty string.")
    validate_observation_keys(image_keys=image_keys, proprio_keys=proprio_keys)

    dataset = make_lerobot_dataset(
        repo_id=repo_id,
        local_files_only=bool(local_files_only),
        video_backend=(str(video_backend).strip() if video_backend else None),
    )
    n_samples = len(dataset)
    if n_samples <= 0:
        raise FileNotFoundError(f"No samples found in LeRobot dataset: {repo_id}")

    source = dataset
    if not load_images and hasattr(dataset, "hf_dataset"):
        # Fast path for feature-only training: use tabular rows and skip video decode.
        source = dataset.hf_dataset
        n_samples = len(source)

    episode_order: list[int] = []
    obs_by_episode: dict[int, list[dict[str, np.ndarray]]] = {}
    actions_by_episode: dict[int, list[np.ndarray]] = {}

    for i in range(n_samples):
        sample = source[i]
        if not isinstance(sample, dict):
            raise ValueError(f"LeRobot sample at index {i} is not a dict (got {type(sample).__name__}).")
        episode_idx = _extract_episode_index(sample, episode_index_key=episode_index_key, sample_index=i)

        if episode_idx not in obs_by_episode:
            if limit is not None and len(episode_order) >= int(limit):
                break
            episode_order.append(episode_idx)
            obs_by_episode[episode_idx] = []
            actions_by_episode[episode_idx] = []

        action = np.asarray(to_numpy(extract_key(sample, action_key)), dtype=np.float32).reshape(-1)
        obs_t = _extract_obs_t(
            sample=sample,
            image_keys=image_keys,
            proprio_keys=proprio_keys,
            fallback_image_hw=fallback_image_hw,
            load_images=load_images,
        )
        obs_by_episode[episode_idx].append(obs_t)
        actions_by_episode[episode_idx].append(action)

    episodes: list[EpisodeData] = []
    for episode_idx in episode_order:
        obs_seq = obs_by_episode.get(episode_idx, [])
        action_seq = actions_by_episode.get(episode_idx, [])
        if not obs_seq or not action_seq:
            continue
        actions = np.stack(action_seq, axis=0).astype(np.float32)
        episodes.append(EpisodeData(obs=obs_seq, actions=actions))

    if not episodes:
        raise FileNotFoundError(f"No valid episodes loaded from LeRobot dataset: {repo_id}")
    return episodes


def _iter_episode_frame_groups(
    *,
    dataset: Any,
    image_keys: list[str],
    episode_index_key: str,
    fallback_image_hw: tuple[int, int],
    limit: int | None,
) -> Iterator[tuple[int, dict[str, list[np.ndarray]]]]:
    """Yield image frame groups for each episode in stream order."""

    current_episode_id: int | None = None
    current_frames: dict[str, list[np.ndarray]] = {key: [] for key in image_keys}
    emitted = 0
    h, w = fallback_image_hw

    for i in range(len(dataset)):
        sample = dataset[i]
        if not isinstance(sample, dict):
            raise ValueError(f"LeRobot sample at index {i} is not a dict (got {type(sample).__name__}).")
        episode_idx = _extract_episode_index(sample, episode_index_key=episode_index_key, sample_index=i)

        if current_episode_id is None:
            current_episode_id = episode_idx
        elif episode_idx < current_episode_id:
            raise ValueError(
                "LeRobot samples are not grouped by episode_index in stream order. "
                "Streaming precompute requires monotonic episode ordering."
            )

        if episode_idx != current_episode_id:
            if any(current_frames[k] for k in image_keys):
                yield emitted, current_frames
                emitted += 1
                if limit is not None and emitted >= int(limit):
                    return
            current_episode_id = episode_idx
            current_frames = {key: [] for key in image_keys}

        for key in image_keys:
            image = to_uint8_image(to_numpy(extract_key(sample, key)), fallback_hw=(h, w))
            current_frames[key].append(image)

    if any(current_frames[k] for k in image_keys) and (limit is None or emitted < int(limit)):
        yield emitted, current_frames


def iter_lerobot_episode_images(
    repo_id: str,
    image_keys: list[str],
    episode_index_key: str = "episode_index",
    limit: int | None = None,
    fallback_image_hw: tuple[int, int] = (84, 84),
    local_files_only: bool = False,
    video_backend: str | None = "pyav",
) -> tuple[object, dict[str, int | None]]:
    """Stream LeRobot images grouped by episode without loading full dataset in RAM."""

    if not repo_id:
        raise ValueError("LeRobot repo_id must be a non-empty string.")
    if not image_keys:
        raise ValueError("image_keys must contain at least one observation key.")

    dataset = make_lerobot_dataset(
        repo_id=repo_id,
        local_files_only=bool(local_files_only),
        video_backend=(str(video_backend).strip() if video_backend else None),
    )
    meta_obj = getattr(dataset, "meta", None)
    meta: dict[str, int | None] = {
        "total_frames": int(getattr(meta_obj, "total_frames", 0) or 0) if meta_obj is not None else None,
        "total_episodes": int(getattr(meta_obj, "total_episodes", 0) or 0) if meta_obj is not None else None,
    }

    iterator = _iter_episode_frame_groups(
        dataset=dataset,
        image_keys=image_keys,
        episode_index_key=episode_index_key,
        fallback_image_hw=fallback_image_hw,
        limit=limit,
    )
    return iterator, meta
