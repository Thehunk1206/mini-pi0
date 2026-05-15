"""Repo-local PegInsertionSide variants for mini-pi0 experiments.

This module registers a PegInsertionSide environment with additional fixed
sensors aimed at the insertion region. The task dynamics remain the ManiSkill
task; only sensor geometry is changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@dataclass(frozen=True)
class InsertionCameraPose:
    """Fixed camera geometry for observing peg-hole insertion."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float = 48.0


DEFAULT_INSERTION_CAMERAS: tuple[InsertionCameraPose, ...] = (
    InsertionCameraPose(
        name="hole_left_camera",
        eye=(-0.22, -0.20, 0.145),
        target=(0.0, 0.30, 0.095),
        fov_deg=58.0,
    ),
    InsertionCameraPose(
        name="hole_right_camera",
        eye=(0.22, -0.20, 0.145),
        target=(0.0, 0.30, 0.095),
        fov_deg=58.0,
    ),
)


@register_env("MiniPi0PegInsertionSide-v1", max_episode_steps=1000, override=True)
class MiniPi0PegInsertionSideEnv(PegInsertionSideEnv):
    """PegInsertionSide with close side cameras facing the insertion region.

    Args:
        sensor_width: Width used for task-level cameras.
        sensor_height: Height used for task-level cameras.
        insertion_camera_poses: Optional fixed camera geometry overrides. The
            default pair views the hole from symmetric oblique approach-side
            angles so one view remains useful when the arm occludes the other.
        *args: Forwarded to ManiSkill's PegInsertionSide environment.
        **kwargs: Forwarded to ManiSkill's PegInsertionSide environment.
    """

    def __init__(
        self,
        *args: object,
        sensor_width: int = 224,
        sensor_height: int = 224,
        insertion_camera_poses: tuple[InsertionCameraPose, ...] | None = None,
        **kwargs: object,
    ) -> None:
        self.sensor_width = int(sensor_width)
        self.sensor_height = int(sensor_height)
        self.insertion_camera_poses = insertion_camera_poses or DEFAULT_INSERTION_CAMERAS
        self._set_default_sensor_size(kwargs)
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        configs = list(super()._default_sensor_configs)
        for cfg in configs:
            cfg.width = self.sensor_width
            cfg.height = self.sensor_height
            cfg.__post_init__()

        for pose_cfg in self.insertion_camera_poses:
            hole_pose = sapien_utils.look_at(eye=pose_cfg.eye, target=pose_cfg.target)
            configs.append(
                CameraConfig(
                    pose_cfg.name,
                    hole_pose,
                    self.sensor_width,
                    self.sensor_height,
                    np.deg2rad(pose_cfg.fov_deg),
                    0.01,
                    100,
                    shader_pack="minimal",
                )
            )
        return configs

    def _set_default_sensor_size(self, kwargs: dict[str, object]) -> None:
        sensor_configs = dict(kwargs.get("sensor_configs") or {})
        sensor_configs.setdefault("width", self.sensor_width)
        sensor_configs.setdefault("height", self.sensor_height)
        kwargs["sensor_configs"] = sensor_configs
