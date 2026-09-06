"""Memory-bounded Rerun dashboard for learned SO-101 inference."""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from so101.teleop.urdf_model import URDFKinematicModel
from so101.teleop.visualization import configure_rerun_batching

from .inference_engine import InferenceStatus
from .policy_bundle import JOINT_NAMES, PolicyBundle
from .safety import TrackingStatus


@dataclass(frozen=True)
class RerunFrame:
    frame_index: int
    cameras: dict[str, np.ndarray]
    measured: np.ndarray
    command: np.ndarray
    raw_policy_action: np.ndarray | None
    engine: InferenceStatus
    tracking: TrackingStatus
    electrical: dict[str, dict[str, float | int]]


class PolicyRerunLogger:
    """Latest-only background logger so rendering cannot stall control."""

    def __init__(
        self,
        bundle: PolicyBundle,
        visual_model: URDFKinematicModel,
        *,
        control_hz: float,
        camera_hz: float = 5.0,
        mesh_hz: float = 10.0,
        memory_limit: str = "768MiB",
        server_memory_limit: str = "512MiB",
    ) -> None:
        self.bundle = bundle
        self.visual_model = visual_model
        self.camera_every = max(1, round(control_hz / camera_hz))
        self.mesh_every = max(1, round(control_hz / mesh_hz))
        self.status_every = max(1, round(control_hz / 2.0))
        self.memory_limit = memory_limit
        self.server_memory_limit = server_memory_limit
        self._queue: queue.Queue[RerunFrame | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._dropped = 0

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    def start(self, *, spawn: bool = True) -> None:
        if self._thread is not None:
            raise RuntimeError("Rerun logger is already started")
        import rerun as rr

        configure_rerun_batching()
        blueprint = self._blueprint()
        rr.init("so101_mini_pi0_inference", spawn=False, default_blueprint=blueprint)
        if spawn:
            rr.spawn(
                memory_limit=self.memory_limit,
                server_memory_limit=self.server_memory_limit,
                hide_welcome_screen=True,
                default_blueprint=blueprint,
            )
        rr.send_blueprint(blueprint)
        self._initialize_static_scene()
        self._thread = threading.Thread(
            target=self._worker, name="so101-rerun-logger", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Rerun logger did not stop in time")
        self._thread = None
        import rerun as rr

        recording = rr.get_global_data_recording()
        if recording is not None:
            recording.flush()

    def submit(
        self,
        *,
        frame_index: int,
        cameras: Mapping[str, np.ndarray],
        measured: np.ndarray,
        command: np.ndarray,
        raw_policy_action: np.ndarray | None,
        engine: InferenceStatus,
        tracking: TrackingStatus,
        electrical: Mapping[str, Mapping[str, float | int]] | None = None,
    ) -> None:
        frame = RerunFrame(
            frame_index=int(frame_index),
            cameras={name: np.asarray(value).copy() for name, value in cameras.items()},
            measured=np.asarray(measured, dtype=np.float32).copy(),
            command=np.asarray(command, dtype=np.float32).copy(),
            raw_policy_action=(
                None
                if raw_policy_action is None
                else np.asarray(raw_policy_action, dtype=np.float32).copy()
            ),
            engine=engine,
            tracking=tracking,
            electrical={
                name: dict(values) for name, values in (electrical or {}).items()
            },
        )
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)

    def _blueprint(self) -> Any:
        import rerun.blueprint as rrb

        camera_views = tuple(
            rrb.Spatial2DView(
                origin=f"cameras/{name}",
                contents=[f"cameras/{name}"],
                name=f"Camera: {name}",
            )
            for name in self.bundle.camera_names
        )
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="world", contents=["world/**"], name="SO-101"
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="inference/status", name="Inference status"
                        ),
                        rrb.TimeSeriesView(
                            origin="/",
                            contents=["timing/**", "queue/**", "tracking/max_error"],
                            name="RTC timing and queue",
                        ),
                        rrb.TimeSeriesView(
                            origin="/",
                            contents=["state/**", "action/**", "policy_raw/**"],
                            name="Joint state and policy actions",
                        ),
                    ),
                    column_shares=[1, 3],
                ),
                rrb.Horizontal(*camera_views, name="Camera streams"),
                row_shares=[2, 1],
            ),
            collapse_panels=True,
        )

    def _initialize_static_scene(self) -> None:
        import rerun as rr

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        if not self.visual_model.visuals:
            raise ValueError("Official SO-101 visual model contains no mesh geometry")
        for index, visual in enumerate(self.visual_model.visuals):
            transform = visual.origin_transform
            for stream, color in (
                ("measured", list(visual.rgba)),
                ("commanded", [255, 176, 32, 72]),
            ):
                entity = f"world/robot_mesh/{stream}/{visual.link}/visual_{index}"
                rr.log(
                    entity,
                    rr.Transform3D(
                        translation=transform[:3, 3], mat3x3=transform[:3, :3]
                    ),
                    static=True,
                )
                rr.log(
                    entity,
                    rr.Asset3D(path=visual.mesh_path, albedo_factor=color),
                    static=True,
                )

    def _worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            self._log(frame)

    def _log(self, frame: RerunFrame) -> None:
        import rerun as rr

        rr.set_time("control_frame", sequence=frame.frame_index)
        rr.log("timing/inference_ms", rr.Scalars(frame.engine.last_latency_ms))
        rr.log("queue/size", rr.Scalars(frame.engine.queue_size))
        rr.log("queue/underflows", rr.Scalars(frame.engine.underflow_count))
        rr.log("queue/rejected_chunks", rr.Scalars(frame.engine.rejected_chunks))
        rr.log("queue/rtc_fallbacks", rr.Scalars(frame.engine.rtc_fallbacks))
        rr.log("tracking/max_error", rr.Scalars(frame.tracking.worst_error))
        for index, name in enumerate(JOINT_NAMES):
            rr.log(f"state/{name}", rr.Scalars(float(frame.measured[index])))
            rr.log(f"action/safe/{name}", rr.Scalars(float(frame.command[index])))
            rr.log(f"tracking/{name}", rr.Scalars(float(frame.tracking.errors[name])))
            if frame.raw_policy_action is not None:
                rr.log(
                    f"policy_raw/{name}",
                    rr.Scalars(float(frame.raw_policy_action[index])),
                )
        for joint, values in frame.electrical.items():
            for key in ("current_ma", "voltage_v", "load_percent", "temperature_c"):
                if key in values:
                    rr.log(f"electrical/{joint}/{key}", rr.Scalars(float(values[key])))

        if frame.frame_index % self.camera_every == 0:
            for name, image in frame.cameras.items():
                rr.log(
                    f"cameras/{name}",
                    rr.Image(image, color_model="RGB").compress(jpeg_quality=65),
                )
        if frame.frame_index % self.mesh_every == 0:
            for stream, values in (
                ("measured", frame.measured),
                ("commanded", frame.command),
            ):
                positions = {
                    name: float(values[index]) for index, name in enumerate(JOINT_NAMES)
                }
                for link, transform in self.visual_model.lerobot_link_transforms(
                    positions
                ).items():
                    rr.log(
                        f"world/robot_mesh/{stream}/{link}",
                        rr.Transform3D(
                            translation=transform[:3, 3], mat3x3=transform[:3, :3]
                        ),
                    )

        status = frame.engine
        if frame.frame_index % self.status_every == 0 or status.fault:
            rr.log(
                "inference/status",
                rr.TextDocument(
                    "# mini-pi0 inference\n\n"
                    f"**Phase:** `{status.phase}`  \n"
                    f"**Paused:** `{status.paused}`  \n"
                    f"**Fault:** `{status.fault or 'none'}`  \n"
                    f"**Checkpoint:** `{self.bundle.checkpoint_path.name}`  \n"
                    f"**Parameters:** `{self.bundle.parameter_count:,}`  \n"
                    f"**Device / precision:** `{self.bundle.device}` / `{self.bundle.precision_name}`  \n"
                    f"**RTC queue:** `{status.queue_size}` actions  \n"
                    f"**Inference:** `{status.last_latency_ms:.1f} ms`  \n"
                    f"**Delay predicted / actual:** `{status.last_predicted_delay}` / `{status.last_real_delay}`  \n"
                    f"**Rejected / stale:** `{status.rejected_chunks}` / `{status.stale_results_dropped}`  \n"
                    f"**Unguided recoveries:** `{status.rtc_fallbacks}`  \n"
                    f"**Last rejection:** `{status.last_rejection_reason or 'none'}`  \n"
                    f"**Rerun frames dropped:** `{self._dropped}`",
                    media_type="text/markdown",
                ),
            )
