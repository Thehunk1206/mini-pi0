"""Calibration and following-error gates for learned joint commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .config import SafetyConfig
from .policy_bundle import JOINT_NAMES


class UnsafePolicyChunk(ValueError):
    """Raised when an entire generated chunk must be rejected."""


@dataclass(frozen=True)
class TrackingStatus:
    warning: bool
    fault: bool
    worst_joint: str | None
    worst_error: float
    errors: dict[str, float]


class PolicySafetyGate:
    """Validate full chunks, rate-limit them, and monitor measured tracking."""

    def __init__(
        self,
        joint_limits: Mapping[str, tuple[float, float]],
        *,
        control_hz: float,
        config: SafetyConfig,
    ) -> None:
        if tuple(joint_limits) != JOINT_NAMES:
            raise ValueError(
                f"Joint limits must preserve order {JOINT_NAMES}, got {tuple(joint_limits)}"
            )
        self.joint_limits = {
            name: tuple(map(float, joint_limits[name])) for name in JOINT_NAMES
        }
        self.control_hz = float(control_hz)
        self.config = config
        self._lock = Lock()
        self._last_command: np.ndarray | None = None
        self._fault_streak = 0

    def reset(self, measured: np.ndarray) -> None:
        values = self._validate_vector(measured, "measured state")
        with self._lock:
            self._last_command = values.copy()
            self._fault_streak = 0

    def process_chunk(self, chunk: np.ndarray, measured: np.ndarray) -> np.ndarray:
        """Atomically validate raw targets, then apply bounded per-cycle slew."""

        values = self.validate_chunk(chunk)
        measured_values = self._validate_vector(measured, "measured state")
        lower = np.asarray(
            [self.joint_limits[name][0] for name in JOINT_NAMES], dtype=np.float32
        )
        upper = np.asarray(
            [self.joint_limits[name][1] for name in JOINT_NAMES], dtype=np.float32
        )

        maximum_delta = (
            np.asarray(
                (
                    *self.config.arm_velocity_deg_s,
                    self.config.gripper_velocity_percent_s,
                ),
                dtype=np.float32,
            )
            / self.control_hz
        )
        with self._lock:
            previous = (
                self._last_command.copy()
                if self._last_command is not None
                else measured_values.copy()
            )
        safe = np.empty_like(values)
        for index, target in enumerate(values):
            previous = previous + np.clip(
                target - previous, -maximum_delta, maximum_delta
            )
            previous = np.clip(previous, lower, upper)
            safe[index] = previous
        return safe

    def validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Validate shape, finiteness, and every calibrated target atomically."""

        values = np.asarray(chunk, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(JOINT_NAMES)
            or not np.isfinite(values).all()
        ):
            raise UnsafePolicyChunk(
                f"Policy chunk must be finite [T, 6], got {values.shape}"
            )
        lower = np.asarray(
            [self.joint_limits[name][0] for name in JOINT_NAMES], dtype=np.float32
        )
        upper = np.asarray(
            [self.joint_limits[name][1] for name in JOINT_NAMES], dtype=np.float32
        )
        tolerance = np.asarray(
            (
                *([self.config.boundary_saturation_deg] * 5),
                self.config.gripper_boundary_saturation_percent,
            ),
            dtype=np.float32,
        )
        outside = np.argwhere(
            (values < (lower - tolerance)[None, :])
            | (values > (upper + tolerance)[None, :])
        )
        if outside.size:
            step, joint_index = (int(item) for item in outside[0])
            name = JOINT_NAMES[joint_index]
            raise UnsafePolicyChunk(
                f"Policy chunk target {name}={values[step, joint_index]:.2f} at step {step} "
                f"is outside calibrated range [{lower[joint_index]:.2f}, {upper[joint_index]:.2f}]"
            )
        return np.clip(values, lower[None, :], upper[None, :])

    def record_command(self, command: np.ndarray) -> None:
        values = self._validate_vector(command, "command")
        with self._lock:
            self._last_command = values.copy()

    def evaluate_tracking(self, measured: np.ndarray) -> TrackingStatus:
        values = self._validate_vector(measured, "measured state")
        with self._lock:
            if self._last_command is None:
                self._last_command = values.copy()
            errors_array = np.abs(self._last_command - values)
            arm_worst = float(np.max(errors_array[:5]))
            gripper_error = float(errors_array[5])
            over_fault = (
                arm_worst > self.config.following_fault_deg
                or gripper_error > self.config.gripper_fault_percent
            )
            self._fault_streak = self._fault_streak + 1 if over_fault else 0
            fault = self._fault_streak >= self.config.following_fault_cycles
            warning = (
                arm_worst > self.config.following_warning_deg
                or gripper_error > self.config.gripper_fault_percent
            )
        worst_index = int(np.argmax(errors_array)) if len(errors_array) else -1
        return TrackingStatus(
            warning=warning,
            fault=fault,
            worst_joint=JOINT_NAMES[worst_index] if worst_index >= 0 else None,
            worst_error=float(errors_array[worst_index]) if worst_index >= 0 else 0.0,
            errors={
                name: float(errors_array[index])
                for index, name in enumerate(JOINT_NAMES)
            },
        )

    @staticmethod
    def _validate_vector(value: np.ndarray, label: str) -> np.ndarray:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if values.shape != (len(JOINT_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"{label} must contain six finite values, got {values}")
        return values
