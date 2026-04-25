from __future__ import annotations

import os

import numpy as np

from mini_pi0.dataset._loader_utils import resolve_alias_key, resolve_alias_keys, to_uint8_image, validate_observation_keys
from mini_pi0.dataset.types import EpisodeData


def _build_robomimic_episode(
    *,
    obs_group,
    actions: np.ndarray,
    image_keys: list[str],
    proprio_keys: list[str],
    fallback_image_hw: tuple[int, int],
) -> EpisodeData | None:
    """Build one canonical episode from a robomimic ``obs`` group."""

    t = int(actions.shape[0])
    if t <= 0:
        return None

    missing_prop = [k for k in proprio_keys if resolve_alias_key(obs_group, k) not in obs_group]
    if missing_prop:
        raise KeyError(
            f"obs group missing proprio keys {missing_prop}. "
            "Set robot.state_keys (or robot.proprio_keys) to match your dataset keys."
        )

    prop_arrays = {
        key: np.asarray(obs_group[resolve_alias_key(obs_group, key)], dtype=np.float32)
        for key in proprio_keys
    }
    image_source_keys = resolve_alias_keys(obs_group, image_keys)
    image_arrays = {
        key: (np.asarray(obs_group[source_key]) if source_key in obs_group else None)
        for key, source_key in zip(image_keys, image_source_keys, strict=True)
    }

    t = min(t, *[len(arr) for arr in prop_arrays.values()])
    for key in image_keys:
        if image_arrays[key] is not None:
            t = min(t, len(image_arrays[key]))

    h, w = fallback_image_hw
    obs_seq: list[dict[str, np.ndarray]] = []
    for i in range(t):
        obs_t: dict[str, np.ndarray] = {}
        for key in image_keys:
            src = image_arrays[key]
            if src is None:
                obs_t[key] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                obs_t[key] = to_uint8_image(src[i], fallback_hw=(h, w))
        for key in proprio_keys:
            obs_t[key] = np.asarray(prop_arrays[key][i], dtype=np.float32)
        obs_seq.append(obs_t)
    return EpisodeData(obs=obs_seq, actions=actions[:t])


def load_episodes_robomimic(
    hdf5_path: str,
    image_keys: list[str],
    proprio_keys: list[str],
    limit: int | None = None,
    data_group: str = "data",
    fallback_image_hw: tuple[int, int] = (84, 84),
) -> list[EpisodeData]:
    """Load demonstrations from a robomimic-style HDF5 file."""

    try:
        import h5py
    except Exception as e:
        raise RuntimeError("h5py is required for robomimic_hdf5 dataset format.") from e

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Robomimic HDF5 not found: {hdf5_path}")
    validate_observation_keys(image_keys=image_keys, proprio_keys=proprio_keys)

    episodes: list[EpisodeData] = []
    with h5py.File(hdf5_path, "r") as f:
        if data_group not in f:
            raise KeyError(f"Expected top-level group '{data_group}' in {hdf5_path}")
        data = f[data_group]
        demo_keys = sorted(list(data.keys()))
        if limit is not None:
            demo_keys = demo_keys[:limit]
        if not demo_keys:
            raise FileNotFoundError(f"No demo groups found under '{data_group}' in {hdf5_path}")

        for demo_key in demo_keys:
            demo = data[demo_key]
            if "actions" not in demo:
                raise KeyError(f"{hdf5_path}:{data_group}/{demo_key} missing 'actions'")
            if "obs" not in demo:
                raise KeyError(f"{hdf5_path}:{data_group}/{demo_key} missing 'obs'")

            actions = np.asarray(demo["actions"], dtype=np.float32)
            episode = _build_robomimic_episode(
                obs_group=demo["obs"],
                actions=actions,
                image_keys=image_keys,
                proprio_keys=proprio_keys,
                fallback_image_hw=fallback_image_hw,
            )
            if episode is not None:
                episodes.append(episode)

    if not episodes:
        raise FileNotFoundError(f"No valid episodes loaded from {hdf5_path}")
    return episodes
