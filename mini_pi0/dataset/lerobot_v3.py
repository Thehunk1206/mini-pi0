from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


LEROBOT_V3_FORMATS = {"lerobot_v3", "lerobot_hf", "lerobot", "hf"}


@dataclass(frozen=True)
class LeRobotV3OpenConfig:
    """Configuration required to open a LeRobot v3 dataset."""

    repo_id: str
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None
    local_files_only: bool = False
    video_backend: str | None = "pyav"


@dataclass(frozen=True)
class LeRobotV3DatasetInfo:
    """Small summary used to align model dimensions with LeRobot data."""

    action_dim: int
    prop_dim: int
    episode_count: int
    sample_count: int
    image_keys: tuple[str, ...]
    state_key: str


@dataclass(frozen=True)
class LeRobotFeatureSpec:
    """Feature keys consumed by the policy."""

    action_key: str = "action"
    state_key: str = "observation.state"
    image_keys: tuple[str, ...] = ("observation.images.agentview_image",)
    episode_index_key: str = "episode_index"

    @classmethod
    def from_keys(
        cls,
        *,
        action_key: str,
        state_key: str,
        image_keys: Iterable[str],
        episode_index_key: str = "episode_index",
    ) -> "LeRobotFeatureSpec":
        """Create a normalized feature spec from config values."""

        return cls(
            action_key=str(action_key),
            state_key=str(state_key),
            image_keys=tuple(cls.image_key(key) for key in image_keys),
            episode_index_key=str(episode_index_key),
        )

    @staticmethod
    def image_key(key: str) -> str:
        """Return a canonical LeRobot image feature key."""

        clean = str(key).strip()
        if clean.startswith("observation.images."):
            return clean
        return f"observation.images.{clean}"

    def validate(self, dataset: Any) -> None:
        """Validate that required features are present.

        Args:
            dataset: Opened LeRobot dataset.

        Raises:
            KeyError: If a required feature key is missing.
        """

        features = set(getattr(dataset, "features", {}) or {})
        missing = [key for key in (self.action_key, self.state_key, *self.image_keys) if key not in features]
        if missing:
            raise KeyError(f"LeRobot dataset is missing required feature keys: {missing}")


@dataclass(frozen=True)
class LeRobotTemporalConfig:
    """Temporal window configuration expressed as LeRobot delta timestamps."""

    fps: int
    obs_horizon: int
    chunk_size: int

    def delta_timestamps(self, spec: LeRobotFeatureSpec) -> dict[str, list[float]]:
        """Build LeRobot delta timestamp mapping for observations and actions."""

        fps = float(max(1, int(self.fps)))
        obs_h = max(1, int(self.obs_horizon))
        chunk = max(1, int(self.chunk_size))
        obs_deltas = [(idx - obs_h + 1) / fps for idx in range(obs_h)]
        action_deltas = [idx / fps for idx in range(chunk)]
        deltas = {spec.action_key: action_deltas, spec.state_key: obs_deltas}
        for image_key in spec.image_keys:
            deltas[image_key] = obs_deltas
        return deltas


class LeRobotDatasetFactory:
    """Factory for official LeRobotDataset instances."""

    def __init__(self, open_config: LeRobotV3OpenConfig, temporal_config: LeRobotTemporalConfig | None = None):
        """Create a dataset factory."""

        self.open_config = open_config
        self.temporal_config = temporal_config

    def open(self, spec: LeRobotFeatureSpec | None = None) -> Any:
        """Open a LeRobot dataset, optionally with temporal windows."""

        cls = self._dataset_class()
        kwargs: dict[str, Any] = {
            "repo_id": self.open_config.repo_id,
            "root": self.open_config.root,
            "episodes": self.open_config.episodes,
            "revision": self.open_config.revision,
            "video_backend": self.open_config.video_backend,
        }
        if self.temporal_config is not None and spec is not None:
            kwargs["delta_timestamps"] = self.temporal_config.delta_timestamps(spec)
        if self.open_config.local_files_only:
            kwargs["force_cache_sync"] = False
        return cls(**{key: value for key, value in kwargs.items() if value is not None})

    @staticmethod
    def _dataset_class() -> type:
        """Import the official LeRobotDataset class."""

        try:
            module = import_module("lerobot.datasets.lerobot_dataset")
            cls = getattr(module, "LeRobotDataset")
        except Exception as exc:
            raise RuntimeError("LeRobot v3 support requires `lerobot==0.4.4` or a compatible version.") from exc
        return cls


class LeRobotActionStatsComputer:
    """Compute action statistics without decoding image/video features."""

    def __init__(self, dataset: Any, action_key: str = "action") -> None:
        """Create a stats computer for an opened LeRobot dataset."""

        self.dataset = dataset
        self.action_key = action_key

    def iter_actions(self) -> Iterator[np.ndarray]:
        """Yield action arrays from low-dimensional dataset storage."""

        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is not None and self.action_key in getattr(hf_dataset, "column_names", []):
            for value in hf_dataset[self.action_key]:
                yield _to_numpy(value).astype(np.float32)
            return
        for idx in range(len(self.dataset)):
            yield _to_numpy(self.dataset[idx][self.action_key]).astype(np.float32)


class LeRobotPolicyDataset(Dataset):
    """Thin adapter from LeRobot samples to mini-pi0 policy batches.

    Action chunks are intentionally returned raw. The training loop normalizes
    them on the target device to avoid CPU-side work in the dataset hot path.
    """

    actions_are_normalized = False
    prefers_locality_sampler = True

    def __init__(
        self,
        *,
        dataset: Any,
        spec: LeRobotFeatureSpec,
        chunk_size: int,
        obs_horizon: int,
        preserve_camera_dim: bool,
    ) -> None:
        """Create a policy dataset wrapper around an opened LeRobot dataset."""

        self.dataset = dataset
        self.spec = spec
        self.chunk_size = int(chunk_size)
        self.obs_horizon = int(max(1, obs_horizon))
        self.preserve_camera_dim = bool(preserve_camera_dim)
        self.valid_indices = self._valid_indices()
        if not self.valid_indices:
            raise ValueError(f"No LeRobot samples available for chunk_size={self.chunk_size}.")

    def __len__(self) -> int:
        """Return number of valid non-padded samples."""

        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw image/state/action tensors for one policy sample."""

        sample = self.dataset[self.valid_indices[idx]]
        return self._images(sample), self._state(sample), self._actions(sample)

    def info(self) -> LeRobotV3DatasetInfo:
        """Infer dataset dimensions from the first valid sample."""

        sample = self.dataset[self.valid_indices[0]]
        action = _to_tensor(sample[self.spec.action_key]).reshape(self.chunk_size, -1)
        state = _to_tensor(sample[self.spec.state_key])
        state_dim = int(state.reshape(self.obs_horizon, -1).shape[-1])
        return LeRobotV3DatasetInfo(
            action_dim=int(action.shape[-1]),
            prop_dim=state_dim,
            episode_count=self._episode_count(),
            sample_count=len(self.valid_indices),
            image_keys=self.spec.image_keys,
            state_key=self.spec.state_key,
        )

    def _images(self, sample: dict[str, Any]) -> torch.Tensor:
        """Format images into the existing model contract."""

        images = [_image_to_tchw(sample[key], self.obs_horizon) for key in self.spec.image_keys]
        if self.obs_horizon == 1:
            single_t = [img[0] for img in images]
            if len(single_t) == 1:
                return single_t[0].unsqueeze(0) if self.preserve_camera_dim else single_t[0]
            return torch.stack(single_t, dim=0) if self.preserve_camera_dim else torch.cat(single_t, dim=-1)
        if len(images) == 1:
            return images[0].unsqueeze(1) if self.preserve_camera_dim else images[0]
        if self.preserve_camera_dim:
            return torch.stack(images, dim=1)
        return torch.cat(images, dim=-1)

    def _state(self, sample: dict[str, Any]) -> torch.Tensor:
        """Format state into ``[prop]`` or ``[T, prop]``."""

        state = _to_tensor(sample[self.spec.state_key]).float()
        if self.obs_horizon == 1:
            return state.reshape(-1)
        return state.reshape(self.obs_horizon, -1)

    def _actions(self, sample: dict[str, Any]) -> torch.Tensor:
        """Return raw action chunk shaped ``[chunk, action_dim]``."""

        return _to_tensor(sample[self.spec.action_key]).float().reshape(self.chunk_size, -1)

    def _valid_indices(self) -> list[int]:
        """Return starts with complete future action chunks."""

        episode_ids = self._episode_ids_from_hf_dataset()
        if episode_ids is not None:
            return self._valid_indices_from_episode_ids(episode_ids)

        meta = getattr(self.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            indices: list[int] = []
            for episode in episodes:
                start = int(episode["dataset_from_index"])
                end = int(episode["dataset_to_index"])
                indices.extend(range(start, max(start, end - self.chunk_size + 1)))
            return indices

        pad_key = f"{self.spec.action_key}_is_pad"
        return [
            idx
            for idx in range(len(self.dataset))
            if pad_key not in self.dataset[idx] or not bool(_to_tensor(self.dataset[idx][pad_key]).any().item())
        ]

    def _episode_count(self) -> int:
        """Return number of distinct loaded episodes."""

        episode_ids = self._episode_ids_from_hf_dataset()
        if episode_ids is not None:
            return len(set(episode_ids))

        meta = getattr(self.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            return int(len(episodes))
        sampled_episode_ids = [
            int(_to_tensor(self.dataset[idx][self.spec.episode_index_key]).reshape(-1)[0].item())
            for idx in range(len(self.dataset))
        ]
        return len(set(sampled_episode_ids))

    def _valid_indices_from_episode_ids(self, episode_ids: list[int]) -> list[int]:
        """Build relative sample starts from loaded episode ids."""

        indices: list[int] = []
        current_episode = episode_ids[0] if episode_ids else None
        start = 0
        for idx, episode_id in enumerate([*episode_ids, None]):
            if episode_id != current_episode:
                indices.extend(range(start, max(start, idx - self.chunk_size + 1)))
                start = idx
                current_episode = episode_id
        return indices

    def _episode_ids_from_hf_dataset(self) -> list[int] | None:
        """Read episode ids without decoding images when possible."""

        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is None or self.spec.episode_index_key not in getattr(hf_dataset, "column_names", []):
            return None
        return [
            int(_to_tensor(value).reshape(-1)[0].item())
            for value in hf_dataset[self.spec.episode_index_key]
        ]


def open_lerobot_v3_dataset(cfg: LeRobotV3OpenConfig) -> Any:
    """Backward-compatible helper for opening a plain LeRobot dataset."""

    return LeRobotDatasetFactory(cfg).open()


def lerobot_image_key(key: str) -> str:
    """Backward-compatible helper for canonical image feature names."""

    return LeRobotFeatureSpec.image_key(key)


def iter_lerobot_actions(dataset: Any, action_key: str) -> Iterator[np.ndarray]:
    """Backward-compatible helper for streaming actions."""

    yield from LeRobotActionStatsComputer(dataset, action_key).iter_actions()


def build_lerobot_v3_dataset(
    *,
    repo_id: str,
    root: str | None,
    revision: str | None,
    episodes: list[int] | None,
    local_files_only: bool,
    video_backend: str | None,
    action_key: str,
    state_key: str,
    image_keys: list[str],
    chunk_size: int,
    action_stats: Any | None = None,
    obs_horizon: int,
    preserve_camera_dim: bool,
) -> LeRobotPolicyDataset:
    """Backward-compatible factory used by older call sites/tests."""

    spec = LeRobotFeatureSpec.from_keys(action_key=action_key, state_key=state_key, image_keys=image_keys)
    temporal = LeRobotTemporalConfig(fps=20, obs_horizon=obs_horizon, chunk_size=chunk_size)
    dataset = LeRobotDatasetFactory(
        LeRobotV3OpenConfig(
            repo_id=repo_id,
            root=root,
            revision=revision,
            episodes=episodes,
            local_files_only=local_files_only,
            video_backend=video_backend,
        ),
        temporal,
    ).open(spec)
    return LeRobotPolicyDataset(
        dataset=dataset,
        spec=spec,
        chunk_size=chunk_size,
        obs_horizon=obs_horizon,
        preserve_camera_dim=preserve_camera_dim,
    )


def infer_lerobot_v3_info(
    *,
    dataset: Any,
    action_key: str,
    state_key: str,
    image_keys: list[str],
    chunk_size: int,
) -> LeRobotV3DatasetInfo:
    """Backward-compatible info helper for an opened temporal LeRobot dataset."""

    spec = LeRobotFeatureSpec.from_keys(action_key=action_key, state_key=state_key, image_keys=image_keys)
    policy_dataset = LeRobotPolicyDataset(
        dataset=dataset,
        spec=spec,
        chunk_size=chunk_size,
        obs_horizon=1,
        preserve_camera_dim=False,
    )
    return policy_dataset.info()


LeRobotV3ActionChunkDataset = LeRobotPolicyDataset


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a scalar/tensor/list value to a NumPy array."""

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass
    return np.asarray(value)


def _to_tensor(value: Any) -> torch.Tensor:
    """Convert LeRobot values to CPU tensors without expensive transforms."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(_to_numpy(value))


def _image_to_tchw(value: Any, obs_horizon: int) -> torch.Tensor:
    """Convert image value to ``[T,C,H,W]``."""

    tensor = _to_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4 and tensor.shape[1] in {1, 3, 4}:
        pass
    elif tensor.ndim == 4 and tensor.shape[-1] in {1, 3, 4}:
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected LeRobot image as CHW/HWC or TCHW/THWC, got {tuple(tensor.shape)}.")
    if tensor.shape[0] != int(obs_horizon):
        raise ValueError(f"Expected {obs_horizon} image frames, got {tensor.shape[0]}.")
    if tensor.shape[1] == 4:
        tensor = tensor[:, :3]
    return tensor.contiguous()
