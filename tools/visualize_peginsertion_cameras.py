"""Visualize PegInsertionSide sensor views from a replay trajectory.

The tool loads recorded ManiSkill ``env_states`` and renders the same episode
with ``MiniPi0PegInsertionSide-v1`` so the additional insertion cameras can be
checked without editing ManiSkill itself.

Example:
    .venv/bin/python tools/visualize_peginsertion_cameras.py \
      --traj-path demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
      --out-dir tmp/peginsertion_hole_camera_debug
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import mini_pi0.sim.maniskill3_peginsertion_env  # noqa: F401


DEFAULT_CAMERAS: tuple[str, ...] = ("base_camera", "hand_camera", "hole_left_camera", "hole_right_camera")


@dataclass(frozen=True)
class CameraReplayConfig:
    """Runtime configuration for PegInsertion camera visualization."""

    traj_path: Path
    out_dir: Path
    env_id: str
    traj_key: str
    control_mode: str
    obs_mode: str
    sim_backend: str
    sensor_width: int
    sensor_height: int
    cameras: tuple[str, ...]
    frames: tuple[int, ...] | None
    seed: int


class CameraReplayError(RuntimeError):
    """Raised when a trajectory cannot be visualized."""


class ManiSkillStateReader:
    """Read per-frame ManiSkill state dictionaries from HDF5."""

    def __init__(self, traj_path: Path, traj_key: str) -> None:
        self.traj_path = traj_path
        self.traj_key = traj_key

    def frame_count(self) -> int:
        """Return the number of recorded environment states."""

        with h5py.File(self.traj_path, "r") as root:
            traj = self._trajectory_group(root)
            return self._infer_frame_count(traj)

    def read_state(self, frame_idx: int, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
        """Read one state dictionary with a preserved batch dimension.

        Args:
            frame_idx: Environment-state frame index to read.
            device: Torch device used by the ManiSkill environment.

        Returns:
            State dictionary compatible with ``env.unwrapped.set_state_dict``.

        Raises:
            CameraReplayError: If the trajectory is missing env state data.
        """

        with h5py.File(self.traj_path, "r") as root:
            traj = self._trajectory_group(root)
            if "env_states" not in traj:
                raise CameraReplayError(f"{self.traj_key} has no env_states group")
            env_states = traj["env_states"]
            return {
                "actors": self._read_state_group(env_states["actors"], frame_idx, device),
                "articulations": self._read_state_group(env_states["articulations"], frame_idx, device),
            }

    def _trajectory_group(self, root: h5py.File) -> h5py.Group:
        if self.traj_key not in root:
            raise CameraReplayError(f"Missing trajectory group: {self.traj_key}")
        group = root[self.traj_key]
        if not isinstance(group, h5py.Group):
            raise CameraReplayError(f"{self.traj_key} is not an HDF5 group")
        return group

    @staticmethod
    def _infer_frame_count(traj: h5py.Group) -> int:
        if "env_states" not in traj:
            raise CameraReplayError(f"{traj.name} has no env_states group")
        actors = traj["env_states"].get("actors")
        if not isinstance(actors, h5py.Group) or not actors:
            raise CameraReplayError(f"{traj.name}/env_states/actors is empty")
        first = next(iter(actors.values()))
        if not isinstance(first, h5py.Dataset):
            raise CameraReplayError(f"{traj.name}/env_states/actors has no datasets")
        return int(first.shape[0])

    @staticmethod
    def _read_state_group(group: h5py.Group, frame_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, dataset in group.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            arr = np.asarray(dataset[frame_idx : frame_idx + 1], dtype=np.float32)
            out[name] = torch.as_tensor(arr, device=device)
        return out


class PegInsertionCameraRenderer:
    """Render configured cameras from a local PegInsertion environment."""

    def __init__(self, cfg: CameraReplayConfig) -> None:
        self.cfg = cfg
        self.env = gym.make(
            cfg.env_id,
            obs_mode=cfg.obs_mode,
            control_mode=cfg.control_mode,
            render_mode="rgb_array",
            sim_backend=cfg.sim_backend,
            robot_uids="panda_wristcam",
            num_envs=1,
            sensor_width=cfg.sensor_width,
            sensor_height=cfg.sensor_height,
        )

    @property
    def device(self) -> torch.device:
        """Return the torch device used by the underlying ManiSkill env."""

        return torch.device(self.env.unwrapped.device)

    def reset(self, seed: int) -> None:
        """Reset the environment before applying recorded states."""

        self.env.reset(seed=seed, options={"reconfigure": True})

    def render_state(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, np.ndarray]:
        """Apply one state and render all requested cameras."""

        self.env.unwrapped.set_state_dict(dict(state))
        obs = self.env.unwrapped.get_obs({})
        sensor_data = obs.get("sensor_data", {})
        if not isinstance(sensor_data, Mapping):
            raise CameraReplayError("Rendered observation has no sensor_data mapping")

        frames: dict[str, np.ndarray] = {}
        for camera in self.cfg.cameras:
            payload = sensor_data.get(camera)
            if not isinstance(payload, Mapping) or "rgb" not in payload:
                raise CameraReplayError(f"Rendered observation is missing {camera}/rgb")
            frames[camera] = _rgb_to_uint8(payload["rgb"])
        return frames

    def close(self) -> None:
        """Close the ManiSkill environment."""

        self.env.close()


class CameraReplayVisualizer:
    """Coordinate trajectory-state rendering and contact-sheet writing."""

    def __init__(self, cfg: CameraReplayConfig) -> None:
        self.cfg = cfg
        self.reader = ManiSkillStateReader(cfg.traj_path, cfg.traj_key)

    def run(self) -> list[Path]:
        """Render selected frames and write a combined contact sheet."""

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        frame_count = self.reader.frame_count()
        frame_indices = self._resolve_frames(frame_count)

        renderer = PegInsertionCameraRenderer(self.cfg)
        written: list[Path] = []
        try:
            renderer.reset(self.cfg.seed)
            rows: list[np.ndarray] = []
            for frame_idx in frame_indices:
                state = self.reader.read_state(frame_idx, renderer.device)
                frames = renderer.render_state(state)
                row_images: list[np.ndarray] = []
                for camera in self.cfg.cameras:
                    image = frames[camera]
                    out_path = self.cfg.out_dir / f"{self.cfg.traj_key}_frame{frame_idx:04d}_{camera}.png"
                    imageio.imwrite(out_path, image)
                    written.append(out_path)
                    row_images.append(image)
                rows.append(_join_horizontally(row_images))
        finally:
            renderer.close()

        sheet = _join_vertically(rows)
        sheet_path = self.cfg.out_dir / f"{self.cfg.traj_key}_camera_sheet.png"
        imageio.imwrite(sheet_path, sheet)
        written.append(sheet_path)
        return written

    def _resolve_frames(self, frame_count: int) -> tuple[int, ...]:
        if frame_count <= 0:
            raise CameraReplayError("Trajectory has no frames")
        if self.cfg.frames is not None:
            return tuple(_bounded_frame_index(idx, frame_count) for idx in self.cfg.frames)
        return (0, frame_count // 2, frame_count - 1)


def _rgb_to_uint8(value: object) -> np.ndarray:
    """Convert a ManiSkill RGB tensor/array to ``[H, W, 3]`` uint8."""

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise CameraReplayError(f"Expected RGB image with shape [H, W, C], got {arr.shape}")
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def _join_horizontally(images: Sequence[np.ndarray]) -> np.ndarray:
    """Join images into one row with thin black separators."""

    if not images:
        raise CameraReplayError("No images to join")
    height = max(int(img.shape[0]) for img in images)
    padded = [_pad_to_height(img, height) for img in images]
    separator = np.zeros((height, 4, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for idx, image in enumerate(padded):
        if idx:
            parts.append(separator)
        parts.append(image)
    return np.concatenate(parts, axis=1)


def _join_vertically(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Join image rows with thin black separators."""

    if not rows:
        raise CameraReplayError("No rows to join")
    width = max(int(row.shape[1]) for row in rows)
    padded = [_pad_to_width(row, width) for row in rows]
    separator = np.zeros((4, width, 3), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for idx, row in enumerate(padded):
        if idx:
            parts.append(separator)
        parts.append(row)
    return np.concatenate(parts, axis=0)


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def _pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    pad = np.zeros((image.shape[0], width - image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=1)


def _bounded_frame_index(frame_idx: int, frame_count: int) -> int:
    if frame_idx < 0:
        return max(0, frame_count + frame_idx)
    return min(frame_idx, frame_count - 1)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_frame_csv(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> CameraReplayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--env-id", default="MiniPi0PegInsertionSide-v1")
    parser.add_argument("--traj-key", default="traj_0")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--sensor-width", type=int, default=224)
    parser.add_argument("--sensor-height", type=int, default=224)
    parser.add_argument("--cameras", default="base_camera,hand_camera,hole_left_camera,hole_right_camera")
    parser.add_argument("--frames", default=None, help="Comma-separated env-state frame indices. Defaults to start, middle, end.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return CameraReplayConfig(
        traj_path=args.traj_path,
        out_dir=args.out_dir,
        env_id=str(args.env_id),
        traj_key=str(args.traj_key),
        control_mode=str(args.control_mode),
        obs_mode=str(args.obs_mode),
        sim_backend=str(args.sim_backend),
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        cameras=_parse_csv(str(args.cameras)),
        frames=_parse_frame_csv(args.frames),
        seed=int(args.seed),
    )


def main() -> int:
    """Run the camera visualizer CLI."""

    cfg = _parse_args()
    written = CameraReplayVisualizer(cfg).run()
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
