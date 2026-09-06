"""Typed configuration for SO-101 learned-policy deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RTCInferenceConfig:
    """Real-Time Chunking and flow-solver controls."""

    enabled: bool = True
    flow_steps: int = 8
    solver: str = "euler"
    execution_horizon: int = 10
    replan_interval: int = 6
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: str = "EXP"
    fixed_noise: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.flow_steps <= 0:
            raise ValueError("flow_steps must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if self.enabled and self.solver != "euler":
            raise ValueError("guided RTC currently requires the Euler solver")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if self.replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    """Command and following-error safety limits for initial deployment."""

    arm_velocity_deg_s: tuple[float, ...] = (60.0, 60.0, 60.0, 90.0, 120.0)
    gripper_velocity_percent_s: float = 100.0
    following_warning_deg: float = 10.0
    following_fault_deg: float = 15.0
    following_fault_cycles: int = 3
    gripper_fault_percent: float = 10.0
    camera_stale_s: float = 0.25
    underflow_fault_cycles: int = 6
    boundary_saturation_deg: float = 2.0
    gripper_boundary_saturation_percent: float = 2.0

    def __post_init__(self) -> None:
        if len(self.arm_velocity_deg_s) != 5 or any(
            value <= 0 for value in self.arm_velocity_deg_s
        ):
            raise ValueError("arm_velocity_deg_s must contain five positive values")
        if self.gripper_velocity_percent_s <= 0:
            raise ValueError("gripper_velocity_percent_s must be positive")
        if not 0 < self.following_warning_deg < self.following_fault_deg:
            raise ValueError("following-error warning must be below fault threshold")
        if self.following_fault_cycles <= 0 or self.underflow_fault_cycles <= 0:
            raise ValueError("fault cycle counts must be positive")
        if self.camera_stale_s <= 0:
            raise ValueError("camera_stale_s must be positive")
        if (
            self.boundary_saturation_deg < 0
            or self.gripper_boundary_saturation_percent < 0
        ):
            raise ValueError("boundary saturation tolerances cannot be negative")


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level runtime settings shared by dry-run and hardware execution."""

    checkpoint: Path
    device: str = "auto"
    precision: str = "auto"
    control_hz: float = 30.0
    rtc: RTCInferenceConfig = field(default_factory=RTCInferenceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
