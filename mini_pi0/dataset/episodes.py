from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from mini_pi0.config.schema import effective_state_keys


_KEY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("observation.images.base_0_rgb", "agentview_image"),
    ("observation.images.right_wrist_0_rgb", "observation.images.wrist_0_rgb", "robot0_eye_in_hand_image"),
    ("observation.state.eef_pos", "robot0_eef_pos"),
    ("observation.state.eef_quat", "robot0_eef_quat"),
    ("observation.state.tool", "robot0_gripper_qpos"),
    ("observation.state.object", "object-state", "object"),
)


def _alias_candidates(key: str) -> tuple[str, ...]:
    """Return equivalent key candidates (including ``key``) in preference order."""

    for group in _KEY_ALIAS_GROUPS:
        if key in group:
            return group
    return (key,)


def _resolve_alias_key(container: Any, key: str) -> str:
    """Resolve a key against a keyable container with dataset key aliases.

    Args:
        container: Mapping-like object supporting ``in`` membership checks.
        key: Requested key.

    Returns:
        Resolved key present in ``container`` when possible, otherwise original key.
    """

    for cand in _alias_candidates(key):
        if cand in container:
            return cand
    return key


@dataclass
class EpisodeData:
    """Canonical in-memory representation of one demonstration episode.

    Attributes:
        obs: Time-ordered list of observation dictionaries.
        actions: ``[T, action_dim]`` float32 action array.
    """

    obs: list[dict[str, np.ndarray]]
    actions: np.ndarray


def list_supported_dataset_formats() -> list[str]:
    """Return dataset format identifiers supported by the data loader stack.

    Returns:
        List of supported format keys.
    """

    return ["robomimic_hdf5", "lerobot_hf"]


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of tensor-like values to numpy arrays.

    Args:
        value: Source object (NumPy, torch tensor, scalar, list, etc).

    Returns:
        Converted numpy array.
    """

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


def _extract_key(sample: dict[str, Any], key: str) -> Any:
    """Resolve a key from a sample supporting dotted nested lookup.

    Args:
        sample: Sample dictionary.
        key: Direct key or dotted nested key.

    Returns:
        Value for ``key``.

    Raises:
        KeyError: If key cannot be resolved.
    """

    direct = _resolve_alias_key(sample, key)
    if direct in sample:
        return sample[direct]

    cur: Any = sample
    dotted_key = _resolve_alias_key(sample, key)
    for part in dotted_key.split("."):
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


def _import_lerobot_dataset_class():
    """Import ``LeRobotDataset`` from supported module paths.

    Returns:
        LeRobotDataset class object.

    Raises:
        RuntimeError: If package/class cannot be imported.
    """

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


def _make_lerobot_dataset(repo_id: str, local_files_only: bool, video_backend: str | None = None) -> Any:
    """Instantiate a LeRobotDataset with compatible constructor fallbacks.

    Args:
        repo_id: Hugging Face dataset repository id.
        local_files_only: Restrict loading to local cache.
        video_backend: Optional LeRobot video backend override.

    Returns:
        Constructed dataset object.

    Raises:
        RuntimeError: If constructor signatures are incompatible.
    """

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
            return builder()
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise RuntimeError(
        f"Failed to create LeRobotDataset for repo '{repo_id}'. "
        f"Tried multiple constructor signatures. Errors: {' | '.join(errors)}"
    )


def _to_uint8_image(arr: np.ndarray, fallback_hw: tuple[int, int]) -> np.ndarray:
    """Convert arbitrary image-like arrays into ``uint8 HxWx3`` tensors.

    Args:
        arr: Source array in unknown layout/range.
        fallback_hw: ``(H, W)`` used for zero-image fallback when conversion fails.

    Returns:
        RGB frame in ``uint8`` format.
    """

    img = np.asarray(arr)
    if img.ndim == 1:
        h, w = fallback_hw
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3:
        h, w = fallback_hw
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[-1] >= 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_episodes_robomimic(
    hdf5_path: str,
    image_key: str,
    proprio_keys: list[str],
    limit: int | None = None,
    data_group: str = "data",
    fallback_image_hw: tuple[int, int] = (84, 84),
) -> list[EpisodeData]:
    """Load demonstrations from a robomimic-style HDF5 file.

    Expected layout:
    - ``/<data_group>/<demo_key>/actions``
    - ``/<data_group>/<demo_key>/obs/<obs_key>``
    - ``data_group`` defaults to ``data`` (robomimic canonical convention).

    Additional robomimic content such as ``states``, ``next_obs``, ``rewards``,
    ``dones``, dataset attributes (for example ``env_args``), and ``/mask``
    filter keys are valid but not required by this loader.

    Args:
        hdf5_path: Path to input HDF5 file.
        image_key: Observation image key under each demo's ``obs`` group.
        proprio_keys: Proprio keys under each demo's ``obs`` group.
        limit: Optional max number of demo groups to load.
        data_group: Top-level data group name (default ``data``).
        fallback_image_hw: ``(H, W)`` for zero-frame fallback if image key is absent.

    Returns:
        List of parsed episodes.

    Raises:
        RuntimeError: If ``h5py`` is unavailable.
        FileNotFoundError: If file or valid episodes are missing.
        KeyError: If required robomimic groups/datasets are missing.
    """

    try:
        import h5py
    except Exception as e:
        raise RuntimeError("h5py is required for robomimic_hdf5 dataset format.") from e

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Robomimic HDF5 not found: {hdf5_path}")

    episodes: list[EpisodeData] = []
    if not proprio_keys:
        raise ValueError("proprio_keys must contain at least one observation key.")
    with h5py.File(hdf5_path, "r") as f:
        if data_group not in f:
            raise KeyError(f"Expected top-level group '{data_group}' in {hdf5_path}")
        grp = f[data_group]
        demo_keys = sorted(list(grp.keys()))
        if limit is not None:
            demo_keys = demo_keys[:limit]

        if not demo_keys:
            raise FileNotFoundError(f"No demo groups found under '{data_group}' in {hdf5_path}")

        for demo_key in demo_keys:
            d = grp[demo_key]
            if "actions" not in d:
                raise KeyError(f"{hdf5_path}:{data_group}/{demo_key} missing 'actions'")
            if "obs" not in d:
                raise KeyError(f"{hdf5_path}:{data_group}/{demo_key} missing 'obs'")

            obs_grp = d["obs"]
            actions = np.asarray(d["actions"], dtype=np.float32)
            t = int(actions.shape[0])
            if t <= 0:
                continue

            missing_prop = [k for k in proprio_keys if _resolve_alias_key(obs_grp, k) not in obs_grp]
            if missing_prop:
                raise KeyError(
                    f"{hdf5_path}:{data_group}/{demo_key}/obs missing proprio keys {missing_prop}. "
                    "Set robot.state_keys (or robot.proprio_keys) to match your dataset keys."
                )

            prop_arrays = {
                k: np.asarray(obs_grp[_resolve_alias_key(obs_grp, k)], dtype=np.float32)
                for k in proprio_keys
            }

            image_src_key = _resolve_alias_key(obs_grp, image_key)
            has_image = image_src_key in obs_grp
            images = np.asarray(obs_grp[image_src_key]) if has_image else None

            t = min(t, *[len(arr) for arr in prop_arrays.values()])
            if has_image and images is not None:
                t = min(t, len(images))

            obs_seq: list[dict[str, np.ndarray]] = []
            h, w = fallback_image_hw
            for i in range(t):
                if has_image and images is not None:
                    img = _to_uint8_image(images[i], fallback_hw=(h, w))
                else:
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                obs_t: dict[str, np.ndarray] = {image_key: img}
                for key in proprio_keys:
                    obs_t[key] = np.asarray(prop_arrays[key][i], dtype=np.float32)
                obs_seq.append(obs_t)

            episodes.append(EpisodeData(obs=obs_seq, actions=actions[:t]))

    if not episodes:
        raise FileNotFoundError(f"No valid episodes loaded from {hdf5_path}")
    return episodes


def load_episodes_lerobot(
    repo_id: str,
    image_key: str,
    proprio_keys: list[str],
    action_key: str = "action",
    episode_index_key: str = "episode_index",
    limit: int | None = None,
    fallback_image_hw: tuple[int, int] = (84, 84),
    local_files_only: bool = False,
    video_backend: str | None = "pyav",
    load_images: bool = True,
) -> list[EpisodeData]:
    """Load demonstrations from a native Hugging Face LeRobot dataset.

    Args:
        repo_id: Hugging Face LeRobot dataset repo id.
        image_key: Feature key used as image observation (supports dotted key paths).
        proprio_keys: Ordered feature keys concatenated into proprio vector.
        action_key: Feature key containing action vectors.
        episode_index_key: Feature key identifying episode id per frame/sample.
        limit: Optional max number of episodes to load.
        fallback_image_hw: ``(H, W)`` fallback image shape if source image is missing/invalid.
        local_files_only: If ``True``, do not hit network and use local HF cache only.
        video_backend: Video decoding backend override. ``pyav`` is the recommended
            default on macOS to avoid torchcodec / FFmpeg runtime incompatibilities.
        load_images: Whether to decode and include image observations. Set ``False``
            for precomputed-feature training to avoid expensive video decoding.

    Returns:
        Parsed episodes in canonical schema.
    """

    if not repo_id:
        raise ValueError("LeRobot repo_id must be a non-empty string.")

    ds = _make_lerobot_dataset(
        repo_id=repo_id,
        local_files_only=bool(local_files_only),
        video_backend=(str(video_backend).strip() if video_backend else None),
    )
    if video_backend and hasattr(ds, "video_backend"):
        # Some lerobot versions may ignore unknown constructor kwargs;
        # enforce backend explicitly after construction when attribute exists.
        try:
            ds.video_backend = str(video_backend)
        except Exception:
            pass
    n = len(ds)
    if n <= 0:
        raise FileNotFoundError(f"No samples found in LeRobot dataset: {repo_id}")
    source = ds
    if not load_images and hasattr(ds, "hf_dataset"):
        # Fast path for feature-only training: iterate the raw tabular dataset
        # (no video columns), which avoids expensive per-sample video decode.
        source = ds.hf_dataset
        n = len(source)

    h, w = fallback_image_hw
    order: list[int] = []
    obs_by_ep: dict[int, list[dict[str, np.ndarray]]] = {}
    act_by_ep: dict[int, list[np.ndarray]] = {}

    for i in range(n):
        sample = source[i]
        if not isinstance(sample, dict):
            raise ValueError(f"LeRobot sample at index {i} is not a dict (got {type(sample).__name__}).")

        ep_raw = _extract_key(sample, episode_index_key)
        ep_flat = _to_numpy(ep_raw).reshape(-1)
        if ep_flat.size != 1:
            raise ValueError(
                f"LeRobot episode index key '{episode_index_key}' must be scalar. "
                f"Got shape {tuple(ep_flat.shape)} at sample {i}."
            )
        ep_idx = int(ep_flat.item())

        if ep_idx not in obs_by_ep:
            if limit is not None and len(order) >= int(limit):
                break
            order.append(ep_idx)
            obs_by_ep[ep_idx] = []
            act_by_ep[ep_idx] = []

        action = np.asarray(_to_numpy(_extract_key(sample, action_key)), dtype=np.float32).reshape(-1)
        obs_t: dict[str, np.ndarray] = {}
        if load_images:
            image = _to_uint8_image(_to_numpy(_extract_key(sample, image_key)), fallback_hw=(h, w))
            obs_t[image_key] = image
        for key in proprio_keys:
            obs_t[key] = np.asarray(_to_numpy(_extract_key(sample, key)), dtype=np.float32).reshape(-1)

        obs_by_ep[ep_idx].append(obs_t)
        act_by_ep[ep_idx].append(action)

    episodes: list[EpisodeData] = []
    for ep_idx in order:
        ep_obs = obs_by_ep.get(ep_idx, [])
        ep_act = act_by_ep.get(ep_idx, [])
        if not ep_obs or not ep_act:
            continue
        actions = np.stack(ep_act, axis=0).astype(np.float32)
        episodes.append(EpisodeData(obs=ep_obs, actions=actions))

    if not episodes:
        raise FileNotFoundError(f"No valid episodes loaded from LeRobot dataset: {repo_id}")
    return episodes


def iter_lerobot_episode_images(
    repo_id: str,
    image_key: str,
    episode_index_key: str = "episode_index",
    limit: int | None = None,
    fallback_image_hw: tuple[int, int] = (84, 84),
    local_files_only: bool = False,
    video_backend: str | None = "pyav",
) -> tuple[object, dict[str, int | None]]:
    """Stream LeRobot images grouped by episode without loading full dataset in RAM.

    Args:
        repo_id: Hugging Face LeRobot dataset repo id.
        image_key: Feature key used as image observation.
        episode_index_key: Feature key identifying episode id per sample.
        limit: Optional max number of episodes to stream.
        fallback_image_hw: Fallback image shape for invalid/missing frames.
        local_files_only: If ``True``, load from local cache only when supported.
        video_backend: Decoder backend override (`pyav` recommended on macOS).

    Returns:
        Tuple of:
        - an iterator yielding ``(episode_seq_idx, frames)`` where ``frames`` is a list of ``uint8 HxWx3``
        - metadata dictionary with ``total_frames`` and ``total_episodes`` hints

    Raises:
        ValueError: If episode ids are non-monotonic in dataset stream order.
    """

    if not repo_id:
        raise ValueError("LeRobot repo_id must be a non-empty string.")

    ds = _make_lerobot_dataset(
        repo_id=repo_id,
        local_files_only=bool(local_files_only),
        video_backend=(str(video_backend).strip() if video_backend else None),
    )
    if video_backend and hasattr(ds, "video_backend"):
        try:
            ds.video_backend = str(video_backend)
        except Exception:
            pass

    meta_obj = getattr(ds, "meta", None)
    meta: dict[str, int | None] = {
        "total_frames": int(getattr(meta_obj, "total_frames", 0) or 0) if meta_obj is not None else None,
        "total_episodes": int(getattr(meta_obj, "total_episodes", 0) or 0) if meta_obj is not None else None,
    }

    def _iter():
        h, w = fallback_image_hw
        n = len(ds)
        current_ep_id: int | None = None
        current_frames: list[np.ndarray] = []
        emitted = 0

        for i in range(n):
            sample = ds[i]
            if not isinstance(sample, dict):
                raise ValueError(f"LeRobot sample at index {i} is not a dict (got {type(sample).__name__}).")

            ep_raw = _extract_key(sample, episode_index_key)
            ep_flat = _to_numpy(ep_raw).reshape(-1)
            if ep_flat.size != 1:
                raise ValueError(
                    f"LeRobot episode index key '{episode_index_key}' must be scalar. "
                    f"Got shape {tuple(ep_flat.shape)} at sample {i}."
                )
            ep_idx = int(ep_flat.item())

            if current_ep_id is None:
                current_ep_id = ep_idx
            elif ep_idx < current_ep_id:
                raise ValueError(
                    "LeRobot samples are not grouped by episode_index in stream order. "
                    "Streaming precompute requires monotonic episode ordering."
                )

            if ep_idx != current_ep_id:
                if current_frames:
                    yield emitted, current_frames
                    emitted += 1
                    if limit is not None and emitted >= int(limit):
                        return
                current_ep_id = ep_idx
                current_frames = []

            image = _to_uint8_image(_to_numpy(_extract_key(sample, image_key)), fallback_hw=(h, w))
            current_frames.append(image)

        if current_frames and (limit is None or emitted < int(limit)):
            yield emitted, current_frames

    return _iter(), meta


def load_episodes_from_config(cfg) -> list[EpisodeData]:
    """Load episodes according to typed repository config.

    Args:
        cfg: Root configuration with ``data`` and ``robot`` sections populated.

    Returns:
        List of episodes in canonical schema.

    Raises:
        ValueError: If dataset format is unsupported or required fields are missing.
    """

    fmt = str(getattr(cfg.data, "format", "robomimic_hdf5")).strip().lower()
    state_keys = effective_state_keys(cfg.robot)
    hw = tuple(cfg.data.fallback_image_hw)
    if len(hw) != 2:
        raise ValueError("data.fallback_image_hw must be [H, W]")
    fallback_hw = (int(hw[0]), int(hw[1]))

    if fmt in {"robomimic", "robomimic_hdf5", "hdf5"}:
        hdf5_path = cfg.data.robomimic_hdf5
        if not hdf5_path:
            raise ValueError("data.robomimic_hdf5 must be set when data.format=robomimic_hdf5")
        episodes = load_episodes_robomimic(
            hdf5_path=hdf5_path,
            image_key=cfg.robot.image_key,
            proprio_keys=state_keys,
            limit=cfg.data.n_demos,
            data_group=cfg.data.robomimic_data_group,
            fallback_image_hw=fallback_hw,
        )
        return _maybe_attach_precomputed_features(episodes, cfg)

    if fmt in {"lerobot", "lerobot_hf", "hf"}:
        repo_id = cfg.data.lerobot_repo_id
        if not repo_id:
            raise ValueError("data.lerobot_repo_id must be set when data.format=lerobot_hf")
        obs_mode = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
        load_images = obs_mode not in {"precomputed", "feature", "features"}
        episodes = load_episodes_lerobot(
            repo_id=repo_id,
            image_key=cfg.robot.image_key,
            proprio_keys=state_keys,
            action_key=cfg.data.lerobot_action_key,
            episode_index_key=cfg.data.lerobot_episode_index_key,
            limit=cfg.data.n_demos,
            fallback_image_hw=fallback_hw,
            local_files_only=bool(cfg.data.lerobot_local_files_only),
            video_backend=cfg.data.lerobot_video_backend,
            load_images=load_images,
        )
        return _maybe_attach_precomputed_features(episodes, cfg)

    raise ValueError(
        f"Unsupported data.format '{cfg.data.format}'. "
        f"Supported formats: {', '.join(list_supported_dataset_formats())}"
    )


def _maybe_attach_precomputed_features(episodes: list[EpisodeData], cfg) -> list[EpisodeData]:
    """Attach cached vision features to loaded episodes when requested.

    The expected `.npz` format stores per-episode arrays using keys:
    ``ep_000000``, ``ep_000001``, ...

    Args:
        episodes: Episodes loaded from source dataset.
        cfg: Root config containing data section.

    Returns:
        Episodes with feature vectors attached to each timestep observation.
    """

    mode = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    if mode not in {"precomputed", "feature", "features"}:
        return episodes
    feat_path = getattr(cfg.data, "precomputed_features_path", None)
    if not feat_path:
        raise ValueError("data.precomputed_features_path must be set when using precomputed observation mode.")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Precomputed features not found: {feat_path}")

    feature_key = str(getattr(cfg.data, "precomputed_feature_key", "vision_feat"))

    feat_p = Path(feat_path)
    if feat_p.is_dir():
        print(f"[data] Attaching precomputed features from directory: {feat_p}", flush=True)
        for ep_idx, ep in enumerate(episodes):
            key = f"ep_{ep_idx:06d}"
            ep_file = feat_p / f"{key}.npy"
            if not ep_file.exists():
                raise KeyError(
                    f"Missing feature file '{ep_file}'. "
                    "Re-run precompute for the same dataset ordering/limit."
                )
            feats = np.asarray(np.load(ep_file), dtype=np.float32)
            if feats.ndim != 2:
                raise ValueError(f"Expected 2D feature array for '{ep_file}', got shape {feats.shape}")
            if len(feats) != len(ep.obs):
                raise ValueError(
                    f"Feature length mismatch for '{ep_file.name}': features={len(feats)} vs episode_obs={len(ep.obs)}"
                )
            for t, obs_t in enumerate(ep.obs):
                obs_t[feature_key] = feats[t]
            if (ep_idx + 1) % 20 == 0 or (ep_idx + 1) == len(episodes):
                print(f"[data] Attached features for {ep_idx + 1}/{len(episodes)} episodes", flush=True)
        return episodes

    print(f"[data] Attaching precomputed features from archive: {feat_path}", flush=True)
    with np.load(feat_path) as store:
        for ep_idx, ep in enumerate(episodes):
            key = f"ep_{ep_idx:06d}"
            if key not in store:
                raise KeyError(
                    f"Missing feature array '{key}' in {feat_path}. "
                    "Re-run precompute for the same dataset ordering/limit."
                )
            feats = np.asarray(store[key], dtype=np.float32)
            if feats.ndim != 2:
                raise ValueError(f"Expected 2D feature array for '{key}', got shape {feats.shape}")
            if len(feats) != len(ep.obs):
                raise ValueError(
                    f"Feature length mismatch for '{key}': features={len(feats)} vs episode_obs={len(ep.obs)}"
                )
            for t, obs_t in enumerate(ep.obs):
                obs_t[feature_key] = feats[t]
            if (ep_idx + 1) % 20 == 0 or (ep_idx + 1) == len(episodes):
                print(f"[data] Attached features for {ep_idx + 1}/{len(episodes)} episodes", flush=True)
    return episodes
