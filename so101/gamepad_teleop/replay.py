"""Replay one local gamepad episode in the recording-oriented Rerun layout.

This entrypoint is read-only and hardware-free. It displays the articulated
SO-101 mesh, saved camera streams, and only the measured/commanded ``.pos``
channels from the LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so101.teleop.model_assets import DEFAULT_MODEL_CACHE, ensure_model_cache
from so101.teleop.urdf_model import URDFKinematicModel
from so101.teleop.visualization import configure_rerun_batching

from .recording import build_recording_blueprint, log_joint_positions


DEFAULT_DATASET_ROOT = Path("data/lerobot/so101_pick_cube")
DEFAULT_REPO_ID = "local/so101-pick-cube"


@dataclass(frozen=True)
class PositionChannels:
    joint_names: tuple[str, ...]
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]


def position_channels(features: dict[str, Any]) -> PositionChannels:
    """Resolve matching state/action ``<joint>.pos`` dimensions by name."""

    try:
        state_names = list(features["observation.state"]["names"])
        action_names = list(features["action"]["names"])
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset has no named observation.state/action features") from exc

    state_lookup = {
        str(name): index
        for index, name in enumerate(state_names)
        if str(name).endswith(".pos")
    }
    action_lookup = {
        str(name): index
        for index, name in enumerate(action_names)
        if str(name).endswith(".pos")
    }
    names = tuple(name for name in action_lookup if name in state_lookup)
    if not names:
        raise ValueError("dataset contains no matching state/action .pos channels")
    return PositionChannels(
        joint_names=tuple(name.removesuffix(".pos") for name in names),
        state_indices=tuple(state_lookup[name] for name in names),
        action_indices=tuple(action_lookup[name] for name in names),
    )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def to_hwc_uint8(value: Any) -> np.ndarray:
    """Convert a LeRobot CHW tensor (or HWC array) to a Rerun RGB image."""

    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"camera frame must have three dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


class ReplayMeshLogger:
    """Log the official measured mesh and translucent commanded ghost."""

    def __init__(self, model: URDFKinematicModel) -> None:
        self.model = model

    def initialize(self) -> None:
        import rerun as rr

        if not self.model.visuals:
            raise ValueError("the official SO-101 URDF has no mesh geometry")
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for index, visual in enumerate(self.model.visuals):
            visual_transform = visual.origin_transform
            for stream, color in (
                ("measured", list(visual.rgba)),
                ("commanded", [255, 176, 32, 72]),
            ):
                entity = f"world/robot_mesh/{stream}/{visual.link}/visual_{index}"
                rr.log(
                    entity,
                    rr.Transform3D(
                        translation=visual_transform[:3, 3],
                        mat3x3=visual_transform[:3, :3],
                    ),
                    static=True,
                )
                rr.log(
                    entity,
                    rr.Asset3D(path=visual.mesh_path, albedo_factor=color),
                    static=True,
                )

    def log(
        self,
        measured_positions: dict[str, float],
        commanded_positions: dict[str, float],
    ) -> None:
        import rerun as rr

        for stream, positions in (
            ("measured", measured_positions),
            ("commanded", commanded_positions),
        ):
            transforms = self.model.lerobot_link_transforms(positions)
            for link, transform in transforms.items():
                rr.log(
                    f"world/robot_mesh/{stream}/{link}",
                    rr.Transform3D(
                        translation=transform[:3, 3],
                        mat3x3=transform[:3, :3],
                    ),
                )


def _camera_keys(features: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and feature.get("dtype") in {"video", "image"}
    )


def _validate_episode(root: Path, episode_index: int) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    total = int(info.get("total_episodes", 0))
    if episode_index < 0 or episode_index >= total:
        raise ValueError(
            f"episode {episode_index} is unavailable; dataset currently has "
            f"{total} finalized episode(s). Save/finalize the active recording first."
        )
    return info


def replay_episode(
    *,
    repo_id: str,
    root: Path,
    episode_index: int,
) -> None:
    """Load and send a finalized episode to a spawned Rerun viewer."""

    root = root.expanduser().resolve()
    info = _validate_episode(root, episode_index)
    features = dict(info["features"])
    channels = position_channels(features)
    camera_keys = _camera_keys(features)
    camera_names = tuple(key.removeprefix("observation.images.") for key in camera_keys)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    import rerun as rr

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode_index],
        video_backend="pyav",
        return_uint8=True,
    )
    if not len(dataset):
        raise ValueError(f"episode {episode_index} contains no frames")

    model_metadata = ensure_model_cache()
    visual_model = URDFKinematicModel.from_file(
        DEFAULT_MODEL_CACHE / model_metadata["urdf"]
    )

    configure_rerun_batching()
    init_rerun(session_name=f"so101_dataset_episode_{episode_index}")
    blueprint = build_recording_blueprint(
        camera_names,
        status_view_name="Dataset replay",
    )
    rr.send_blueprint(blueprint)
    log_rerun_data.blueprint = blueprint

    mesh = ReplayMeshLogger(visual_model)
    mesh.initialize()
    rr.log(
        "recording/status",
        rr.TextDocument(
            "# Dataset replay\n\n"
            f"**Episode:** `{episode_index}`  \n"
            f"**Frames:** `{len(dataset)}`  \n"
            f"**FPS:** `{info['fps']}`  \n"
            f"**Dataset:** `{root}`  \n\n"
            "The plot contains only measured `state/*.pos` and commanded "
            "`action/*.pos` channels.",
            media_type="text/markdown",
        ),
        static=True,
    )

    for local_index in range(len(dataset)):
        frame = dataset[local_index]
        timestamp = float(_numpy(frame["timestamp"]).reshape(-1)[0])
        rr.set_time("frame_index", sequence=local_index)
        rr.set_time("timestamp", timestamp=timestamp)

        state = _numpy(frame["observation.state"]).reshape(-1)
        action = _numpy(frame["action"]).reshape(-1)
        measured = {
            joint: float(state[index])
            for joint, index in zip(
                channels.joint_names,
                channels.state_indices,
                strict=True,
            )
        }
        commanded = {
            joint: float(action[index])
            for joint, index in zip(
                channels.joint_names,
                channels.action_indices,
                strict=True,
            )
        }
        observation = {f"{joint}.pos": value for joint, value in measured.items()}
        command = {f"{joint}.pos": value for joint, value in commanded.items()}
        log_joint_positions(observation, command, channels.joint_names)
        mesh.log(measured, commanded)

        for key, name in zip(camera_keys, camera_names, strict=True):
            rr.log(
                f"cameras/{name}",
                rr.Image(to_hwc_uint8(frame[key]), color_model="RGB").compress(
                    jpeg_quality=65
                ),
            )

    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush()
    print(
        f"Loaded episode {episode_index}: {len(dataset)} frames, "
        f"{len(channels.joint_names)} measured/action position pairs, "
        f"{len(camera_keys)} camera stream(s)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=0)
    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        replay_episode(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episode_index=args.episode_index,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()
