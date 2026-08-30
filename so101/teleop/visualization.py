"""Shared SO-101 Cartesian/URDF snapshots with optional Rerun visualization."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .urdf_model import URDFKinematicModel


RERUN_FLUSH_NUM_BYTES = 1_048_576
RERUN_FLUSH_TICK_S = 0.1


def configure_rerun_batching() -> None:
    """Use efficient low-latency batches instead of LeRobot's 8 KiB default.

    Rerun stores each flushed batch as a chunk. The very small default selected
    by LeRobot is useful for sparse camera streams, but creates excessive chunk
    overhead for our many robot-link transforms and scalar channels. Respect an
    explicit user override while choosing the Rerun SDK's normal 1 MiB threshold
    and a 100 ms live-view latency for this teleoperation stack.
    """
    os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", str(RERUN_FLUSH_NUM_BYTES))
    os.environ.setdefault("RERUN_FLUSH_TICK_SECS", str(RERUN_FLUSH_TICK_S))


@dataclass(frozen=True)
class CartesianSnapshot:
    actual_position_m: list[float]
    target_position_m: list[float]
    error_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class URDFRenderSnapshot:
    name: str
    root_link: str
    edges: list[dict[str, str]]
    actual_links_m: dict[str, list[float]]
    target_links_m: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ordered_joint_positions(values: dict[str, Any], joint_names: list[str]) -> np.ndarray:
    """Read normalized joint positions from observation or action dictionaries."""
    positions = []
    for joint in joint_names:
        key = f"{joint}.pos"
        if key not in values:
            raise ValueError(f"Joint data is missing: {key}")
        positions.append(float(values[key]))
    return np.asarray(positions, dtype=float)


def calculate_cartesian_snapshot(
    kinematics: Any,
    joint_names: list[str],
    observation: dict[str, Any],
    action: dict[str, Any],
) -> tuple[CartesianSnapshot, np.ndarray, np.ndarray]:
    """Compute measured and commanded end-effector transforms using FK."""
    actual_transform = np.asarray(
        kinematics.forward_kinematics(ordered_joint_positions(observation, joint_names)),
        dtype=float,
    )
    target_transform = np.asarray(
        kinematics.forward_kinematics(ordered_joint_positions(action, joint_names)),
        dtype=float,
    )
    actual_position = actual_transform[:3, 3]
    target_position = target_transform[:3, 3]
    snapshot = CartesianSnapshot(
        actual_position_m=actual_position.tolist(),
        target_position_m=target_position.tolist(),
        error_m=float(np.linalg.norm(target_position - actual_position)),
    )
    return snapshot, actual_transform, target_transform


class EndEffector3DVisualizer:
    """Render actual/target poses and a measured trajectory in Rerun."""

    ACTUAL_COLOR = [32, 220, 120]
    TARGET_COLOR = [255, 176, 32]
    ERROR_COLOR = [255, 80, 80]
    AXIS_COLORS = [[255, 64, 64], [64, 255, 64], [64, 128, 255]]

    def __init__(
        self,
        kinematics: Any,
        joint_names: list[str],
        urdf_model: URDFKinematicModel,
        trail_length: int = 900,
        *,
        rerun_enabled: bool = False,
        visual_urdf_model: URDFKinematicModel | None = None,
        show_skeleton: bool = True,
        show_trail: bool = True,
        rerun_log_every_n_frames: int = 1,
    ) -> None:
        if trail_length < 2:
            raise ValueError("3D trail length must be at least 2")
        if rerun_log_every_n_frames < 1:
            raise ValueError("Rerun frame decimation must be at least 1")
        self.kinematics = kinematics
        self.joint_names = list(joint_names)
        self.urdf_model = urdf_model
        self.rerun_enabled = bool(rerun_enabled)
        self.visual_urdf_model = visual_urdf_model
        self.show_skeleton = bool(show_skeleton)
        self.show_trail = bool(show_trail)
        self.rerun_log_every_n_frames = int(rerun_log_every_n_frames)
        self._rerun_frame_index = 0
        self._trail: deque[list[float]] = deque(maxlen=trail_length)
        self._initialized = False

    def initialize(self) -> None:
        """Install a Rerun layout containing the 3D view and telemetry plots."""
        if not self.rerun_enabled:
            self._initialized = True
            return

        import rerun as rr
        import rerun.blueprint as rrb

        from lerobot.utils.visualization_utils import log_rerun_data

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world",
                    contents=["world/**"],
                    name="SO-101 end effector",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin="/",
                        contents=["metrics/**"],
                        name="Cartesian tracking",
                    ),
                    rrb.TimeSeriesView(
                        origin="/",
                        contents=["observation/**", "action/**"],
                        name="Motor and input data",
                    ),
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        # Prevent LeRobot's first scalar log from replacing this custom layout.
        log_rerun_data.blueprint = blueprint
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "world/base",
            rr.Points3D([[0.0, 0.0, 0.0]], radii=[0.008], colors=[[180, 180, 180]]),
            static=True,
        )
        if self.visual_urdf_model is not None:
            if not self.visual_urdf_model.visuals:
                raise ValueError("The Rerun visual URDF contains no mesh geometry")
            for index, visual in enumerate(self.visual_urdf_model.visuals):
                visual_transform = visual.origin_transform
                for stream, color in (
                    ("measured", list(visual.rgba)),
                    ("commanded", [255, 176, 32, 72]),
                ):
                    entity = (
                        f"world/robot_mesh/{stream}/{visual.link}/visual_{index}"
                    )
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
        self._initialized = True

    def _log_visual_mesh_poses(
        self,
        actual_joint_positions: dict[str, float],
        target_joint_positions: dict[str, float],
    ) -> None:
        if self.visual_urdf_model is None:
            return

        import rerun as rr

        for stream, joints in (
            ("measured", actual_joint_positions),
            ("commanded", target_joint_positions),
        ):
            transforms = self.visual_urdf_model.lerobot_link_transforms(joints)
            for link, transform in transforms.items():
                rr.log(
                    f"world/robot_mesh/{stream}/{link}",
                    rr.Transform3D(
                        translation=transform[:3, 3],
                        mat3x3=transform[:3, :3],
                    ),
                )

    def log(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> tuple[CartesianSnapshot, URDFRenderSnapshot]:
        """Log Cartesian and full URDF state for Rerun, the UI, and incident logs."""
        if not self._initialized:
            raise RuntimeError("End-effector visualizer has not been initialized")

        snapshot, actual_transform, _target_transform = calculate_cartesian_snapshot(
            self.kinematics, self.joint_names, observation, action
        )
        actual = np.asarray(snapshot.actual_position_m, dtype=float)
        target = np.asarray(snapshot.target_position_m, dtype=float)
        self._trail.append(snapshot.actual_position_m)

        actual_joint_positions = {
            joint: float(observation[f"{joint}.pos"]) for joint in self.joint_names
        }
        target_joint_positions = {
            joint: float(action[f"{joint}.pos"]) for joint in self.joint_names
        }
        actual_links = self.urdf_model.lerobot_link_positions(actual_joint_positions)
        target_links = self.urdf_model.lerobot_link_positions(target_joint_positions)
        urdf_snapshot = URDFRenderSnapshot(
            name=self.urdf_model.robot_name,
            root_link=self.urdf_model.root_link,
            edges=self.urdf_model.edges,
            actual_links_m=actual_links,
            target_links_m=target_links,
        )

        if not self.rerun_enabled:
            return snapshot, urdf_snapshot

        rerun_frame_index = self._rerun_frame_index
        self._rerun_frame_index += 1
        if rerun_frame_index % self.rerun_log_every_n_frames != 0:
            return snapshot, urdf_snapshot

        import rerun as rr

        self._log_visual_mesh_poses(actual_joint_positions, target_joint_positions)

        actual_segments = [
            [actual_links[edge["parent"]], actual_links[edge["child"]]]
            for edge in self.urdf_model.edges
        ]
        target_segments = [
            [target_links[edge["parent"]], target_links[edge["child"]]]
            for edge in self.urdf_model.edges
        ]
        if self.show_skeleton:
            rr.log(
                "world/robot/actual",
                rr.LineStrips3D(
                    actual_segments,
                    radii=[0.006] * len(actual_segments),
                    colors=[self.ACTUAL_COLOR] * len(actual_segments),
                ),
            )
            rr.log(
                "world/robot/target",
                rr.LineStrips3D(
                    target_segments,
                    radii=[0.003] * len(target_segments),
                    colors=[[255, 176, 32, 130]] * len(target_segments),
                ),
            )
            rr.log(
                "world/robot/joints",
                rr.Points3D(
                    list(actual_links.values()),
                    radii=[0.007] * len(actual_links),
                    colors=[self.ACTUAL_COLOR] * len(actual_links),
                    labels=list(actual_links),
                ),
            )

        rr.log(
            "world/end_effector/actual",
            rr.Points3D(
                [actual], radii=[0.012], colors=[self.ACTUAL_COLOR], labels=["actual"]
            ),
        )
        rr.log(
            "world/end_effector/target",
            rr.Points3D(
                [target], radii=[0.010], colors=[self.TARGET_COLOR], labels=["commanded"]
            ),
        )
        rr.log(
            "world/end_effector/error",
            rr.LineStrips3D(
                [[actual, target]], radii=[0.002], colors=[self.ERROR_COLOR]
            ),
        )
        if self.show_trail and len(self._trail) >= 2:
            rr.log(
                "world/end_effector/actual_trail",
                rr.LineStrips3D(
                    [list(self._trail)], radii=[0.0015], colors=[self.ACTUAL_COLOR]
                ),
            )

        axis_vectors = actual_transform[:3, :3].T * 0.04
        axis_origins = np.repeat(actual[None, :], 3, axis=0)
        rr.log(
            "world/end_effector/actual_axes",
            rr.Arrows3D(
                origins=axis_origins,
                vectors=axis_vectors,
                radii=[0.0015, 0.0015, 0.0015],
                colors=self.AXIS_COLORS,
            ),
        )

        rr.log("metrics/cartesian_error_mm", rr.Scalars(snapshot.error_m * 1000.0))
        for index, axis in enumerate("xyz"):
            rr.log(
                f"metrics/actual_{axis}_m",
                rr.Scalars(snapshot.actual_position_m[index]),
            )
            rr.log(
                f"metrics/target_{axis}_m",
                rr.Scalars(snapshot.target_position_m[index]),
            )
        return snapshot, urdf_snapshot
