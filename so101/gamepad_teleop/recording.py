"""Gamepad-only camera and LeRobotDataset recording support.

This module contains no robot or serial-port ownership.  Cameras and the
dataset are opened before the caller opens the servo bus, and all recording
transitions are edge-triggered by :class:`GamepadSample`.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

from .gamepad import GamepadSample


CAMERA_SPEC_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
CAMERA_ROTATIONS = (0, 90, 180, 270)
CAMERA_OUTPUT_SIZE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_]*)=(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)$"
)
CAMERA_WARMUP_SECONDS = 5
CAMERA_CONNECT_ATTEMPTS = 2


def recording_controls_markdown() -> str:
    """Return the compact gamepad recording help shown in Rerun."""

    return (
        "## Gamepad recording commands\n\n"
        "**A** Start episode · **Y** Save episode · **X / Back** Discard episode  \n"
        "**B** Return to base (motion stays in episode) · "
        "**Start / Menu** Finalize and exit"
    )


def recording_status_markdown(status: dict[str, Any]) -> str:
    """Render live dataset status and controls together in one Rerun panel."""

    return (
        "# Dataset recording\n\n"
        f"**State:** `{status['state']}`  \n"
        f"**Next episode:** `{status['episode_index']}`  \n"
        f"**Saved:** `{status['saved_episodes']}`  \n"
        f"**Current frames:** `{status['episode_frames']}`  \n"
        f"**Current duration:** `{status['episode_seconds']:.1f}s`  \n"
        f"**Last event:** `{status['last_event']}`  \n"
        f"**Dataset:** `{status['repo_id']}`  \n"
        f"**Task:** `{status['task']}`  \n"
        f"**Output:** `{status['dataset_root']}`\n\n"
        f"{recording_controls_markdown()}"
    )


@dataclass(frozen=True)
class CameraSpec:
    """One named camera source plus native and dataset-output settings."""

    name: str
    source: int | Path
    rotation_deg: int = 0
    fps: int | None = None
    output_width: int | None = None
    output_height: int | None = None

    def __post_init__(self) -> None:
        if not CAMERA_SPEC_PATTERN.fullmatch(self.name):
            raise ValueError(
                "camera name must start with a letter and contain only "
                "letters, numbers, and underscores"
            )
        if isinstance(self.source, int) and self.source < 0:
            raise ValueError("camera index must be non-negative")
        if self.rotation_deg not in CAMERA_ROTATIONS:
            raise ValueError(f"camera rotation must be one of {CAMERA_ROTATIONS}")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("camera output width and height must be specified together")
        if self.output_width is not None and (
            self.output_width <= 0 or self.output_height <= 0
        ):
            raise ValueError("camera output width and height must be positive")

    @property
    def description(self) -> str:
        fps = f"@{self.fps}" if self.fps is not None else ""
        output_size = (
            f" -> {self.output_width}x{self.output_height}"
            if self.output_width is not None
            else ""
        )
        return f"{self.name}={self.source}:{self.rotation_deg}{fps}{output_size}"


def parse_camera_spec(value: str) -> CameraSpec:
    """Parse ``NAME=INDEX[:ROTATION][@FPS]``."""

    if "=" not in value:
        raise ValueError("camera must use NAME=INDEX[:ROTATION][@FPS]")
    name, source_and_rotation = value.split("=", 1)
    name = name.strip()
    source_and_rotation = source_and_rotation.strip()
    if not source_and_rotation:
        raise ValueError("camera source cannot be empty")

    fps: int | None = None
    if "@" in source_and_rotation:
        source_and_rotation, fps_text = source_and_rotation.rsplit("@", 1)
        if not fps_text.strip().isdecimal() or int(fps_text) <= 0:
            raise ValueError("camera fps must be a positive integer")
        fps = int(fps_text)

    rotation = 0
    source_text = source_and_rotation
    if ":" in source_and_rotation:
        candidate_source, candidate_rotation = source_and_rotation.rsplit(":", 1)
        if candidate_rotation.strip() in {str(item) for item in CAMERA_ROTATIONS}:
            source_text = candidate_source.strip()
            rotation = int(candidate_rotation)
    if not source_text:
        raise ValueError("camera source cannot be empty")
    source: int | Path
    if source_text.isdecimal():
        source = int(source_text)
    else:
        source = Path(source_text).expanduser()
    return CameraSpec(
        name=name,
        source=source,
        rotation_deg=rotation,
        fps=fps,
    )


def parse_camera_specs(values: Iterable[str] | None) -> tuple[CameraSpec, ...]:
    """Parse repeatable camera arguments and reject ambiguous duplicates."""

    specs = tuple(parse_camera_spec(value) for value in (values or ("wrist=0:180",)))
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("camera names must be unique")
    sources = [str(spec.source) for spec in specs]
    if len(sources) != len(set(sources)):
        raise ValueError("each physical camera source may only be configured once")
    if not specs:
        raise ValueError("at least one camera is required for dataset recording")
    return specs


def apply_camera_output_sizes(
    specs: Iterable[CameraSpec],
    values: Iterable[str] | None,
) -> tuple[CameraSpec, ...]:
    """Apply repeatable ``NAME=WIDTHxHEIGHT`` output transforms to cameras.

    Frames are center-cropped to the requested aspect ratio before resizing,
    so non-square source images can become square without geometric stretching.
    """

    configured = tuple(specs)
    sizes: dict[str, tuple[int, int]] = {}
    for value in values or ():
        match = CAMERA_OUTPUT_SIZE_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError("camera output size must use NAME=WIDTHxHEIGHT")
        name = match.group("name")
        if name in sizes:
            raise ValueError(f"camera output size for {name!r} was specified twice")
        sizes[name] = (int(match.group("width")), int(match.group("height")))

    known_names = {spec.name for spec in configured}
    unknown_names = set(sizes) - known_names
    if unknown_names:
        unknown = ", ".join(sorted(unknown_names))
        raise ValueError(f"camera output size refers to unknown camera(s): {unknown}")

    return tuple(
        replace(
            spec,
            output_width=sizes[spec.name][0],
            output_height=sizes[spec.name][1],
        )
        if spec.name in sizes
        else spec
        for spec in configured
    )


def center_crop_and_resize(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Center-crop ``frame`` to the target aspect ratio, then resize it."""

    source_height, source_width = frame.shape[:2]
    if source_width == width and source_height == height:
        return frame

    if source_width * height > width * source_height:
        crop_width = max(1, source_height * width // height)
        left = (source_width - crop_width) // 2
        cropped = frame[:, left : left + crop_width]
    elif source_width * height < width * source_height:
        crop_height = max(1, source_width * height // width)
        top = (source_height - crop_height) // 2
        cropped = frame[top : top + crop_height, :]
    else:
        cropped = frame

    import cv2

    shrinking = width * height < cropped.shape[0] * cropped.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized)


class CameraRig:
    """Own any number of independently named LeRobot OpenCV cameras."""

    def __init__(
        self,
        specs: Iterable[CameraSpec],
        *,
        fps: int,
        width: int,
        height: int,
        camera_factory: Callable[[CameraSpec, int, int, int], Any] | None = None,
    ) -> None:
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("CameraRig requires at least one camera")
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("camera fps, width, and height must be positive")
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self._specs_by_name = {spec.name: spec for spec in self.specs}
        self._camera_factory = camera_factory or self._make_opencv_camera
        self._cameras: dict[str, Any] = {}
        self._latest: dict[str, np.ndarray] = {}

    @staticmethod
    def _make_opencv_camera(
        spec: CameraSpec,
        fps: int,
        width: int,
        height: int,
    ) -> Any:
        from lerobot.cameras import Cv2Rotation
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        rotation_value = -90 if spec.rotation_deg == 270 else spec.rotation_deg
        config = OpenCVCameraConfig(
            index_or_path=spec.source,
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation(rotation_value),
            # High-resolution AVFoundation cameras (notably Canon EOS models)
            # can take longer than LeRobot's one-second default to deliver the
            # first frame, even though the capture device opened successfully.
            warmup_s=CAMERA_WARMUP_SECONDS,
        )
        return OpenCVCamera(config)

    @property
    def is_connected(self) -> bool:
        return bool(self._cameras) and all(
            bool(camera.is_connected) for camera in self._cameras.values()
        )

    @property
    def frame_shapes(self) -> dict[str, tuple[int, int, int]]:
        if not self._latest:
            raise RuntimeError("camera frames are unavailable before connect()")
        return {
            name: tuple(int(item) for item in frame.shape)
            for name, frame in self._latest.items()
        }

    def connect(self) -> None:
        if self._cameras:
            raise RuntimeError("camera rig is already connected")
        try:
            for spec in self.specs:
                camera_fps = spec.fps or self.fps
                for attempt in range(1, CAMERA_CONNECT_ATTEMPTS + 1):
                    camera = self._camera_factory(
                        spec,
                        camera_fps,
                        self.width,
                        self.height,
                    )
                    try:
                        camera.connect()
                        break
                    except TimeoutError:
                        if attempt == CAMERA_CONNECT_ATTEMPTS:
                            raise
                        print(
                            f"WARNING: camera {spec.name!r} produced no startup "
                            "frame; reopening once"
                        )
                self._cameras[spec.name] = camera
            self._latest = self.read()
        except BaseException:
            self.disconnect()
            raise

    def read(self) -> dict[str, np.ndarray]:
        if not self._cameras:
            raise RuntimeError("camera rig is not connected")
        frames: dict[str, np.ndarray] = {}
        for name, camera in self._cameras.items():
            frame = np.asarray(camera.read_latest(max_age_ms=500))
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(
                    f"camera {name!r} returned {frame.shape}; expected RGB HxWx3"
                )
            if frame.dtype != np.uint8:
                raise RuntimeError(
                    f"camera {name!r} returned {frame.dtype}; expected uint8"
                )
            spec = self._specs_by_name[name]
            if spec.output_width is not None and spec.output_height is not None:
                frame = center_crop_and_resize(
                    frame,
                    spec.output_width,
                    spec.output_height,
                )
            frames[name] = frame.copy()
        self._latest = frames
        return frames

    def disconnect(self) -> None:
        for camera in reversed(tuple(self._cameras.values())):
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                print(f"WARNING: camera disconnect failed: {exc}")
        self._cameras.clear()


@dataclass(frozen=True)
class RecordingConfig:
    """Local LeRobotDataset recording configuration."""

    repo_id: str
    root: Path
    task: str
    camera_specs: tuple[CameraSpec, ...]
    fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    use_videos: bool = True
    resume: bool = False
    overwrite: bool = False
    num_episodes: int = 0
    max_episode_seconds: float = 0.0
    image_writer_threads: int = 4
    rerun_camera_hz: float = 5.0

    def __post_init__(self) -> None:
        if not self.repo_id.strip() or "/" not in self.repo_id:
            raise ValueError("repo_id must use NAMESPACE/DATASET_NAME")
        if not self.task.strip():
            raise ValueError("task cannot be empty")
        if not self.camera_specs:
            raise ValueError("at least one camera is required")
        if self.resume and self.overwrite:
            raise ValueError("resume/append and overwrite are mutually exclusive")
        if self.fps <= 0 or self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("recording fps and camera dimensions must be positive")
        if self.num_episodes < 0 or self.max_episode_seconds < 0.0:
            raise ValueError("episode limits cannot be negative")
        if self.image_writer_threads < 0 or self.rerun_camera_hz <= 0.0:
            raise ValueError("writer threads and Rerun camera rate are invalid")


def build_dataset_features(
    joint_names: Iterable[str],
    camera_shapes: dict[str, tuple[int, int, int]],
    *,
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    """Build a standard LeRobot state/action/image feature specification."""

    joints = tuple(joint_names)
    observation_features: dict[str, type | tuple[int, int, int]] = {
        f"{joint}.pos": float for joint in joints
    }
    observation_features.update(camera_shapes)
    action_features = {f"{joint}.pos": float for joint in joints}
    return {
        **hw_to_dataset_features(
            observation_features,
            "observation",
            use_video=use_videos,
        ),
        **hw_to_dataset_features(action_features, "action", use_video=use_videos),
    }


def build_observation_values(
    joint_names: Iterable[str],
    observation: dict[str, Any],
    camera_frames: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Keep only measured joint positions and camera frames in the dataset."""

    values: dict[str, Any] = dict(camera_frames)
    for joint in joint_names:
        values[f"{joint}.pos"] = float(observation[f"{joint}.pos"])
    return values


class GamepadDatasetRecorder:
    """Transactional gamepad episode state machine around LeRobotDataset."""

    WAITING = "waiting"
    RECORDING = "recording"
    FINALIZED = "finalized"

    def __init__(
        self,
        config: RecordingConfig,
        joint_names: Iterable[str],
        camera_shapes: dict[str, tuple[int, int, int]],
        *,
        dataset: Any | None = None,
    ) -> None:
        self.config = config
        self.joint_names = tuple(joint_names)
        self.features = build_dataset_features(
            self.joint_names,
            camera_shapes,
            use_videos=config.use_videos,
        )
        self.dataset = dataset
        self.is_open = False
        self.backup_path: Path | None = None
        self.state = self.WAITING
        self.episode_frames = 0
        self.episode_started_at: float | None = None
        self.stop_requested = False
        self.last_event = "recorder_created"
        self.event_sequence = 0

    @property
    def saved_episodes(self) -> int:
        if self.dataset is None:
            return 0
        return int(self.dataset.meta.total_episodes)

    def _event(self, message: str) -> None:
        self.last_event = message
        self.event_sequence += 1
        print(f"DATASET: {message}")

    def open(self) -> None:
        if self.dataset is not None:
            self.is_open = True
            self._event(f"ready at {self.config.root}")
            return
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot dataset support is missing. Install "
                "so101/gamepad_teleop/requirements.txt."
            ) from exc

        root = self.config.root.expanduser()
        if self.config.resume and not root.exists():
            raise FileNotFoundError(
                f"cannot append: dataset does not exist at {root}"
            )
        if not self.config.resume and root.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"dataset already exists at {root}; pass --append to add "
                    "episodes or --overwrite to archive it and start fresh"
                )
            self.backup_path = self._archive_existing_root(root)
            self._event(f"existing dataset archived to {self.backup_path}")

        if self.config.resume:
            self.dataset = LeRobotDataset.resume(
                repo_id=self.config.repo_id,
                root=self.config.root,
                video_backend="pyav",
                image_writer_threads=self.config.image_writer_threads,
            )
            self._validate_resume_features()
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                root=self.config.root,
                fps=self.config.fps,
                robot_type="so101_gamepad",
                features=self.features,
                use_videos=self.config.use_videos,
                image_writer_processes=0,
                image_writer_threads=self.config.image_writer_threads,
                video_backend="pyav",
                batch_encoding_size=1,
            )
        self.is_open = True
        self._event(f"ready at {self.config.root}")

    @staticmethod
    def _archive_existing_root(root: Path) -> Path:
        """Move an exact dataset root aside instead of destructively deleting it."""

        resolved = root.resolve()
        protected = {
            Path(resolved.anchor),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
        if resolved in protected:
            raise ValueError(f"refusing to overwrite protected path: {resolved}")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        backup = root.with_name(f"{root.name}.backup-{stamp}")
        root.rename(backup)
        return backup

    def _validate_resume_features(self) -> None:
        from lerobot.utils.constants import DEFAULT_FEATURES

        assert self.dataset is not None
        existing = self.dataset.features
        expected_keys = set(self.features)
        extra_keys = set(existing) - expected_keys
        if not expected_keys.issubset(existing) or not extra_keys.issubset(
            DEFAULT_FEATURES
        ):
            raise ValueError(
                "cannot resume: configured cameras/state fields do not match the dataset"
            )
        for key, expected in self.features.items():
            actual = existing[key]
            if (
                actual.get("dtype") != expected.get("dtype")
                or tuple(actual.get("shape", ())) != tuple(expected.get("shape", ()))
                or list(actual.get("names", ())) != list(expected.get("names", ()))
            ):
                raise ValueError(f"cannot resume: feature {key!r} is incompatible")

    def start_episode(self) -> bool:
        if self.state != self.WAITING or self.stop_requested:
            return False
        self.state = self.RECORDING
        self.episode_frames = 0
        self.episode_started_at = time.monotonic()
        self._event(f"episode {self.saved_episodes} recording started")
        return True

    def add_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        camera_frames: dict[str, np.ndarray],
    ) -> bool:
        if self.state != self.RECORDING:
            return False
        if self.dataset is None:
            raise RuntimeError("dataset recorder is not open")
        observation_values = build_observation_values(
            self.joint_names,
            observation,
            camera_frames,
        )
        frame = {
            **build_dataset_frame(
                self.features,
                observation_values,
                prefix="observation",
            ),
            **build_dataset_frame(self.features, action, prefix="action"),
            "task": self.config.task,
        }
        self.dataset.add_frame(frame)
        self.episode_frames += 1
        if (
            self.config.max_episode_seconds > 0.0
            and self.episode_started_at is not None
            and time.monotonic() - self.episode_started_at
            >= self.config.max_episode_seconds
        ):
            self.discard_episode("timeout")
        return True

    def save_episode(self) -> bool:
        if self.state != self.RECORDING:
            return False
        if self.episode_frames == 0:
            self._event("success ignored because the episode has no frames")
            return False
        assert self.dataset is not None
        frame_count = self.episode_frames
        # Sequential camera encoding avoids macOS multiprocessing/spawn issues
        # and keeps peak memory bounded when several camera streams are present.
        self.dataset.save_episode(parallel_encoding=False)
        self.state = self.WAITING
        self.episode_frames = 0
        self.episode_started_at = None
        self._event(
            f"episode {self.saved_episodes - 1} saved ({frame_count} frames)"
        )
        if (
            self.config.num_episodes > 0
            and self.saved_episodes >= self.config.num_episodes
        ):
            self.stop_requested = True
            self._event(f"requested {self.config.num_episodes} episodes completed")
        return True

    def discard_episode(self, reason: str) -> bool:
        if self.state != self.RECORDING:
            return False
        assert self.dataset is not None
        frame_count = self.episode_frames
        self.dataset.clear_episode_buffer(delete_images=True)
        self.state = self.WAITING
        self.episode_frames = 0
        self.episode_started_at = None
        self._event(f"episode discarded ({reason}, {frame_count} frames)")
        return True

    def handle_gamepad(self, sample: GamepadSample) -> bool:
        """Apply recording buttons and return whether the main loop should stop."""

        if sample.stop_recording:
            self.stop_requested = True
            self._event("finish requested by Start/Menu")
            return True
        if sample.rerecord:
            self.discard_episode("rerecord")
        elif sample.failure:
            self.discard_episode("failure")
        elif sample.success:
            self.save_episode()
        elif sample.start_episode:
            self.start_episode()
        return self.stop_requested

    def finalize(self) -> None:
        if self.state == self.FINALIZED:
            return
        if not self.is_open:
            if self.dataset is not None:
                self.dataset.finalize()
            self.state = self.FINALIZED
            return
        if self.state == self.RECORDING:
            self.discard_episode("shutdown before success")
        if self.dataset is not None:
            self.dataset.finalize()
        self.state = self.FINALIZED
        self._event(f"finalized with {self.saved_episodes} saved episodes")

    def status(self) -> dict[str, Any]:
        duration_s = (
            time.monotonic() - self.episode_started_at
            if self.episode_started_at is not None
            else 0.0
        )
        return {
            "state": self.state,
            "episode_index": self.saved_episodes,
            "saved_episodes": self.saved_episodes,
            "episode_frames": self.episode_frames,
            "episode_seconds": duration_s,
            "stop_requested": self.stop_requested,
            "last_event": self.last_event,
            "dataset_root": str(self.config.root),
            "repo_id": self.config.repo_id,
            "task": self.config.task,
            "backup_path": (
                str(self.backup_path) if self.backup_path is not None else None
            ),
        }


def build_recording_blueprint(
    camera_names: Iterable[str],
    *,
    status_view_name: str = "Dataset recording",
) -> Any:
    """Create the shared live-recording and dataset-replay layout."""

    import rerun.blueprint as rrb

    cameras = tuple(camera_names)
    camera_views = tuple(
        rrb.Spatial2DView(
            origin=f"cameras/{name}",
            contents=[f"cameras/{name}"],
            name=f"Camera: {name}",
        )
        for name in cameras
    )
    camera_columns = max(1, min(3, math.ceil(math.sqrt(len(camera_views)))))
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world",
                    contents=["world/**"],
                    name="SO-101",
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(
                        origin="recording/status",
                        name=status_view_name,
                    ),
                    rrb.TimeSeriesView(
                        origin="/",
                        contents=["state/**", "action/**"],
                        name="Measured state and commanded action (.pos only)",
                    ),
                ),
                column_shares=[1, 3],
            ),
            rrb.Grid(*camera_views, grid_columns=camera_columns, name="Camera streams"),
            row_shares=[1, 2],
        ),
        collapse_panels=True,
    )


def log_joint_positions(
    observation: dict[str, Any],
    action: dict[str, Any],
    joint_names: Iterable[str],
) -> None:
    """Log only measured and commanded ``.pos`` fields to Rerun plots."""

    import rerun as rr

    for joint in joint_names:
        key = f"{joint}.pos"
        rr.log(f"state/{key}", rr.Scalars(float(observation[key])))
        rr.log(f"action/{key}", rr.Scalars(float(action[key])))


def install_recording_layout(
    config: RecordingConfig,
    *,
    controller_name: str,
) -> None:
    """Install the custom layout after the mesh visualizer initializes."""

    import rerun as rr

    from lerobot.utils.visualization_utils import log_rerun_data

    blueprint = build_recording_blueprint(spec.name for spec in config.camera_specs)
    rr.send_blueprint(blueprint)
    log_rerun_data.blueprint = blueprint
    camera_lines = "\n".join(
        f"- `{spec.name}`: source `{spec.source}`, rotation `{spec.rotation_deg}°`"
        for spec in config.camera_specs
    )
    rr.log(
        "recording/status",
        rr.TextDocument(
            f"""# SO-101 gamepad dataset recording

**Controller:** {controller_name}<br>
**Dataset:** `{config.repo_id}`<br>
**Task:** {config.task}

{recording_controls_markdown()}

## Cameras
{camera_lines}
""",
            media_type="text/markdown",
        ),
    )


class RecordingRerunLogger:
    """Memory-bounded camera/status logger for a recording session."""

    def __init__(self, *, control_fps: int, camera_hz: float) -> None:
        self.camera_every_n_frames = max(1, round(control_fps / camera_hz))
        self.status_every_n_frames = max(1, control_fps)
        self._last_event_sequence = -1

    def log(
        self,
        recorder: GamepadDatasetRecorder,
        camera_frames: dict[str, np.ndarray],
        *,
        frame_index: int,
    ) -> None:
        import rerun as rr

        if frame_index % self.camera_every_n_frames == 0:
            for name, frame in camera_frames.items():
                rr.log(
                    f"cameras/{name}",
                    rr.Image(frame, color_model="RGB").compress(jpeg_quality=65),
                )
        if (
            frame_index % self.status_every_n_frames == 0
            or recorder.event_sequence != self._last_event_sequence
        ):
            status = recorder.status()
            rr.log(
                "recording/status",
                rr.TextDocument(
                    recording_status_markdown(status),
                    media_type="text/markdown",
                ),
            )
            rr.log("recording/episode_frames", rr.Scalars(status["episode_frames"]))
            rr.log("recording/saved_episodes", rr.Scalars(status["saved_episodes"]))
            if recorder.event_sequence != self._last_event_sequence:
                rr.log(
                    "recording/events",
                    rr.TextLog(status["last_event"], level=rr.TextLogLevel.INFO),
                )
                self._last_event_sequence = recorder.event_sequence
