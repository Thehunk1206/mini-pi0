"""Class-based robomimic HDF5 to LeRobot v3 conversion."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


DEFAULT_LEROBOT_STATE_KEY = "observation.state"
DEFAULT_LEROBOT_ACTION_KEY = "action"


@dataclass(frozen=True)
class RobomimicToLeRobotConfig:
    """Configuration for converting robomimic HDF5 to LeRobot v3."""

    input_hdf5: str
    output_dir: str
    repo_id: str
    data_group: str = "data"
    task_name: str = "robot manipulation"
    fps: int = 20
    robot_type: str = "panda"
    image_keys: tuple[str, ...] = ("agentview_image",)
    state_keys: tuple[str, ...] = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    limit: int | None = None
    use_videos: bool = True
    streaming_encoding: bool = False
    overwrite: bool = False
    log_every: int = 25
    show_progress: bool = True
    quiet_ffmpeg: bool = True


@dataclass(frozen=True)
class RobomimicEpisode:
    """One robomimic demonstration episode."""

    key: str
    actions: np.ndarray
    images: dict[str, np.ndarray]
    states: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        """Return episode length."""

        return int(self.actions.shape[0])

    def state_vector(self, step: int, state_keys: tuple[str, ...]) -> np.ndarray:
        """Concatenate configured state keys for one timestep."""

        parts = [np.asarray(self.states[key][step], dtype=np.float32).reshape(-1) for key in state_keys]
        return np.concatenate(parts, axis=0).astype(np.float32)


class RobomimicHdf5Reader:
    """Read and validate robomimic-style HDF5 demonstrations."""

    def __init__(self, cfg: RobomimicToLeRobotConfig) -> None:
        """Create a reader for one source HDF5."""

        self.cfg = cfg
        self.path = Path(cfg.input_hdf5)
        if not self.path.exists():
            raise FileNotFoundError(f"robomimic HDF5 not found: {self.path}")

    def episodes(self) -> Iterator[RobomimicEpisode]:
        """Yield validated robomimic episodes."""

        with h5py.File(self.path, "r") as src:
            data = self._data_group(src)
            for demo_key in self.demo_keys():
                yield self._read_episode(demo_key, data[demo_key])

    def first_episode(self) -> RobomimicEpisode:
        """Return the first episode for schema inference."""

        return next(self.episodes())

    def demo_keys(self) -> list[str]:
        """Return selected demo keys without reading episode arrays."""

        with h5py.File(self.path, "r") as src:
            data = self._data_group(src)
            demo_keys = self._demo_keys(data)
        if self.cfg.limit is not None:
            demo_keys = demo_keys[: int(self.cfg.limit)]
        if not demo_keys:
            raise ValueError(f"No demos found in {self.path}/{self.cfg.data_group}.")
        return demo_keys

    def _data_group(self, src: h5py.File) -> h5py.Group:
        """Return the configured HDF5 data group."""

        if self.cfg.data_group not in src:
            raise ValueError(f"Missing HDF5 group '{self.cfg.data_group}' in {self.path}.")
        return src[self.cfg.data_group]

    def _read_episode(self, demo_key: str, demo: h5py.Group) -> RobomimicEpisode:
        """Read one demo group into memory for conversion."""

        if "actions" not in demo or "obs" not in demo:
            raise ValueError(f"Demo '{demo_key}' must contain 'actions' and 'obs'.")
        obs = demo["obs"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        images = {key: self._read_obs_array(obs, key, demo_key) for key in self.cfg.image_keys}
        states = {key: self._read_obs_array(obs, key, demo_key) for key in self.cfg.state_keys}
        self._validate_lengths(demo_key, actions, images, states)
        return RobomimicEpisode(key=demo_key, actions=actions, images=images, states=states)

    @staticmethod
    def _read_obs_array(obs: h5py.Group, key: str, demo_key: str) -> np.ndarray:
        """Read one observation dataset."""

        if key not in obs:
            raise ValueError(f"Demo '{demo_key}' is missing obs key: {key}")
        return np.asarray(obs[key])

    @staticmethod
    def _validate_lengths(
        demo_key: str,
        actions: np.ndarray,
        images: dict[str, np.ndarray],
        states: dict[str, np.ndarray],
    ) -> None:
        """Validate all arrays share the action sequence length."""

        length = int(actions.shape[0])
        for key, value in {**images, **states}.items():
            if int(value.shape[0]) != length:
                raise ValueError(f"Demo '{demo_key}' key '{key}' length {value.shape[0]} != actions length {length}.")

    @staticmethod
    def _demo_keys(data_group: h5py.Group) -> list[str]:
        """Return sorted robomimic demo keys."""

        def sort_key(name: str) -> tuple[int, str]:
            suffix = name.rsplit("_", maxsplit=1)[-1]
            return (int(suffix) if suffix.isdigit() else 10**9, name)

        return sorted((key for key in data_group.keys() if key.startswith("demo_")), key=sort_key)


class LeRobotFeatureBuilder:
    """Build LeRobot feature metadata from robomimic episodes."""

    def __init__(self, cfg: RobomimicToLeRobotConfig) -> None:
        """Create a feature builder."""

        self.cfg = cfg

    def build(self, episode: RobomimicEpisode) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Build LeRobot features and state component names."""

        state_names = self._state_names(episode)
        features: dict[str, dict[str, Any]] = {
            DEFAULT_LEROBOT_ACTION_KEY: {
                "dtype": "float32",
                "shape": (int(episode.actions.reshape(episode.length, -1).shape[-1]),),
                "names": [f"action_{idx}" for idx in range(episode.actions.reshape(episode.length, -1).shape[-1])],
            },
            DEFAULT_LEROBOT_STATE_KEY: {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }
        for key in self.cfg.image_keys:
            frame = self._image_frame(episode.images[key][0])
            features[self.image_key(key)] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": tuple(int(v) for v in frame.shape),
                "names": ["height", "width", "channels"],
            }
        return features, state_names

    def _state_names(self, episode: RobomimicEpisode) -> list[str]:
        """Return flattened state component names."""

        names: list[str] = []
        for key in self.cfg.state_keys:
            width = int(np.asarray(episode.states[key][0]).reshape(-1).shape[0])
            names.extend(f"{key}.{idx}" for idx in range(width))
        return names

    @staticmethod
    def image_key(image_key: str) -> str:
        """Map a robomimic image key to LeRobot image feature naming."""

        clean = str(image_key).strip()
        if clean.startswith("observation.images."):
            return clean
        return f"observation.images.{clean}"

    @staticmethod
    def _image_frame(value: np.ndarray) -> np.ndarray:
        """Normalize image arrays to uint8 HWC RGB-compatible data."""

        frame = np.asarray(value)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Expected HWC image with 1/3/4 channels, got shape {frame.shape}.")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame


class RobomimicToLeRobotConverter:
    """Convert robomimic HDF5 episodes to a LeRobot v3 dataset."""

    def __init__(self, cfg: RobomimicToLeRobotConfig) -> None:
        """Create a converter."""

        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.reader = RobomimicHdf5Reader(cfg)
        self.features = LeRobotFeatureBuilder(cfg)

    def convert(self) -> dict[str, Any]:
        """Run conversion and return a summary."""

        self._prepare_output_dir()
        demo_keys = self.reader.demo_keys()
        total_episodes = len(demo_keys)
        self._log_start(total_episodes)
        first_episode = self.reader.first_episode()
        feature_schema, state_names = self.features.build(first_episode)
        dataset = self._create_dataset(feature_schema)
        total_samples = 0
        episode_count = 0
        try:
            episodes = self._episode_iterator(total_episodes)
            for episode_count, episode in enumerate(episodes, start=1):
                self._write_episode(dataset, episode)
                total_samples += episode.length
                self._log_progress(episode_count, total_episodes, total_samples)
            if hasattr(dataset, "finalize"):
                dataset.finalize()
        except KeyboardInterrupt:
            self._print_partial_failure(episode_count, total_episodes, total_samples)
            raise
        except Exception as exc:
            message = self._partial_failure_message(episode_count, total_episodes, total_samples)
            raise RuntimeError(message) from exc
        print(
            "[convert] Finished LeRobot conversion | "
            f"episodes={episode_count}/{total_episodes} frames={total_samples} output={self.output_dir}",
            flush=True,
        )
        return {
            "input_hdf5": str(self.reader.path),
            "output_dir": str(self.output_dir),
            "repo_id": self.cfg.repo_id,
            "episodes": episode_count,
            "total_samples": total_samples,
            "image_keys": [self.features.image_key(key) for key in self.cfg.image_keys],
            "state_key": DEFAULT_LEROBOT_STATE_KEY,
            "state_names": state_names,
            "action_key": DEFAULT_LEROBOT_ACTION_KEY,
            "use_videos": bool(self.cfg.use_videos),
        }

    def _write_episode(self, dataset: Any, episode: RobomimicEpisode) -> None:
        """Write one episode to LeRobot."""

        for step in range(episode.length):
            frame: dict[str, Any] = {
                DEFAULT_LEROBOT_ACTION_KEY: episode.actions[step],
                DEFAULT_LEROBOT_STATE_KEY: episode.state_vector(step, self.cfg.state_keys),
                "task": self.cfg.task_name,
            }
            for key in self.cfg.image_keys:
                frame[self.features.image_key(key)] = self.features._image_frame(episode.images[key][step])
            dataset.add_frame(frame)
        with self._maybe_silence_ffmpeg():
            dataset.save_episode()

    def _create_dataset(self, features: dict[str, dict[str, Any]]) -> Any:
        """Create the destination LeRobot dataset."""

        cls = self._dataset_class()
        return cls.create(
            repo_id=self.cfg.repo_id,
            fps=int(self.cfg.fps),
            features=features,
            root=self.output_dir,
            robot_type=self.cfg.robot_type,
            use_videos=bool(self.cfg.use_videos),
            video_backend="pyav",
            vcodec="h264",
            streaming_encoding=bool(self.cfg.streaming_encoding),
        )

    @staticmethod
    def _dataset_class() -> type:
        """Import the official LeRobotDataset class."""

        try:
            module = import_module("lerobot.datasets.lerobot_dataset")
            cls = getattr(module, "LeRobotDataset")
        except Exception as exc:
            raise RuntimeError("robomimic-to-LeRobot conversion requires `lerobot==0.4.4`.") from exc
        for attr in ("create", "add_frame", "save_episode"):
            if not hasattr(cls, attr):
                raise RuntimeError(f"Installed LeRobotDataset is missing required method: {attr}")
        return cls

    def _prepare_output_dir(self) -> None:
        """Validate or clear the output directory."""

        if self.output_dir.exists():
            if not self.cfg.overwrite:
                raise FileExistsError(f"Output directory already exists: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)

    def _episode_iterator(self, total_episodes: int) -> Iterator[RobomimicEpisode]:
        """Return an episode iterator with optional progress display."""

        episodes = self.reader.episodes()
        if not self.cfg.show_progress:
            return episodes
        try:
            from tqdm.auto import tqdm
        except Exception:
            return episodes
        return iter(tqdm(episodes, total=total_episodes, desc="Converting episodes", unit="ep"))

    def _log_start(self, total_episodes: int) -> None:
        """Print conversion setup."""

        print(
            "[convert] Starting robomimic -> LeRobot v3 | "
            f"input={self.reader.path} output={self.output_dir} episodes={total_episodes} "
            f"cameras={len(self.cfg.image_keys)} use_videos={self.cfg.use_videos} "
            f"quiet_ffmpeg={self.cfg.quiet_ffmpeg}",
            flush=True,
        )

    def _log_progress(self, episode_count: int, total_episodes: int, total_samples: int) -> None:
        """Print periodic progress for non-interactive logs."""

        log_every = int(max(0, self.cfg.log_every))
        if log_every <= 0 and episode_count != total_episodes:
            return
        if episode_count != total_episodes and episode_count % log_every != 0:
            return
        print(
            "[convert] Progress | "
            f"episodes={episode_count}/{total_episodes} frames={total_samples} output={self.output_dir}",
            flush=True,
        )

    def _print_partial_failure(self, episode_count: int, total_episodes: int, total_samples: int) -> None:
        """Print partial-output guidance after interruption."""

        print(self._partial_failure_message(episode_count, total_episodes, total_samples), file=sys.stderr, flush=True)

    def _partial_failure_message(self, episode_count: int, total_episodes: int, total_samples: int) -> str:
        """Build actionable partial-output error text."""

        return (
            "[convert] LeRobot conversion did not finish cleanly. "
            f"Completed episodes before failure: {episode_count}/{total_episodes}; frames={total_samples}. "
            f"Output directory may be partial and should not be used for training: {self.output_dir}. "
            "Restart with --overwrite after removing/overwriting the partial output."
        )

    @contextlib.contextmanager
    def _maybe_silence_ffmpeg(self) -> Iterator[None]:
        """Temporarily silence native stdout/stderr during video encoding."""

        if not (self.cfg.quiet_ffmpeg and self.cfg.use_videos):
            yield
            return
        with _SilenceNativeOutput():
            yield


class _SilenceNativeOutput:
    """Context manager that redirects OS-level stdout/stderr to ``os.devnull``."""

    def __enter__(self) -> "_SilenceNativeOutput":
        """Redirect file descriptors 1 and 2."""

        sys.stdout.flush()
        sys.stderr.flush()
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._devnull_fd, 1)
        os.dup2(self._devnull_fd, 2)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore file descriptors 1 and 2."""

        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._devnull_fd)


def convert_robomimic_to_lerobot(cfg: RobomimicToLeRobotConfig) -> dict[str, Any]:
    """Convert a robomimic-style HDF5 file to a local LeRobot v3 dataset."""

    return RobomimicToLeRobotConverter(cfg).convert()
