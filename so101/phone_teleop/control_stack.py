"""Live phone filtering, Ruckig OTG, and tracking-safety state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization

from .filtering import (
    DEFAULT_PHONE_FILTER_SETTINGS,
    OneEuroXYZFilter,
    PhoneFilterSample,
    validated_phone_filter_settings,
)


ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME = "gripper"
PROFILE_LIMIT_SCALES = {
    # Smooth deliberately reduces jerk more than velocity and acceleration.
    "Smooth": {
        "arm_velocity": 2.0 / 3.0,
        "arm_acceleration": 2.0 / 3.0,
        "arm_jerk": 0.4,
        "gripper_velocity": 0.6,
        "gripper_acceleration": 0.6,
        "gripper_jerk": 0.48,
    },
    "Safe": dict.fromkeys(
        (
            "arm_velocity", "arm_acceleration", "arm_jerk",
            "gripper_velocity", "gripper_acceleration", "gripper_jerk",
        ),
        1.0,
    ),
    "Balanced": dict.fromkeys(
        (
            "arm_velocity", "arm_acceleration", "arm_jerk",
            "gripper_velocity", "gripper_acceleration", "gripper_jerk",
        ),
        1.5,
    ),
    "Responsive": dict.fromkeys(
        (
            "arm_velocity", "arm_acceleration", "arm_jerk",
            "gripper_velocity", "gripper_acceleration", "gripper_jerk",
        ),
        2.5,
    ),
}
PROFILE_NAMES = tuple(PROFILE_LIMIT_SCALES)
TARGET_VELOCITY_TIME_CONSTANT_S = 0.12
TARGET_VELOCITY_LIMIT_FRACTION = 0.8
TARGET_VELOCITY_LIMIT_MARGIN = np.array([10.0, 10.0, 10.0, 10.0, 10.0])


@dataclass(frozen=True)
class CommissioningLimits:
    arm_velocity_deg_s: tuple[float, ...] = (30.0, 30.0, 30.0, 45.0, 60.0)
    arm_acceleration_deg_s2: tuple[float, ...] = (90.0, 90.0, 90.0, 135.0, 180.0)
    arm_jerk_deg_s3: tuple[float, ...] = (450.0, 450.0, 450.0, 675.0, 900.0)
    gripper_velocity_percent_s: float = 25.0
    gripper_acceleration_percent_s2: float = 75.0
    gripper_jerk_percent_s3: float = 375.0

    def scaled(self, profile: str) -> dict[str, list[float]]:
        try:
            scales = PROFILE_LIMIT_SCALES[profile]
        except KeyError as exc:
            raise ValueError(f"Unknown motion profile: {profile}") from exc
        return {
            "arm_velocity": [
                value * scales["arm_velocity"] for value in self.arm_velocity_deg_s
            ],
            "arm_acceleration": [
                value * scales["arm_acceleration"]
                for value in self.arm_acceleration_deg_s2
            ],
            "arm_jerk": [
                value * scales["arm_jerk"] for value in self.arm_jerk_deg_s3
            ],
            "gripper_velocity": [
                self.gripper_velocity_percent_s * scales["gripper_velocity"]
            ],
            "gripper_acceleration": [
                self.gripper_acceleration_percent_s2 * scales["gripper_acceleration"]
            ],
            "gripper_jerk": [
                self.gripper_jerk_percent_s3 * scales["gripper_jerk"]
            ],
        }


DEFAULT_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (math.degrees(-1.69), math.degrees(1.69)),
    "wrist_flex": (math.degrees(-1.65806), math.degrees(1.65806)),
    "wrist_roll": (math.degrees(-2.74385), math.degrees(2.84121)),
    "gripper": (0.0, 100.0),
}


class HoldActiveError(RuntimeError):
    """Raised when motion settings are changed while the clutch is active."""


class _RuckigGroup:
    def __init__(self, degrees_of_freedom: int, cycle_s: float) -> None:
        self.degrees_of_freedom = degrees_of_freedom
        self.otg = Ruckig(degrees_of_freedom, cycle_s)
        self.input = InputParameter(degrees_of_freedom)
        self.output = OutputParameter(degrees_of_freedom)
        self.input.synchronization = Synchronization.Time
        self.initialized = False
        self.result = Result.Finished

    def reset(self, position: Any) -> None:
        values = np.asarray(position, dtype=float)
        if values.shape != (self.degrees_of_freedom,):
            raise ValueError("Ruckig reset position has the wrong shape")
        self.input.current_position = values.tolist()
        self.input.current_velocity = [0.0] * self.degrees_of_freedom
        self.input.current_acceleration = [0.0] * self.degrees_of_freedom
        self.input.target_position = values.tolist()
        self.input.target_velocity = [0.0] * self.degrees_of_freedom
        self.input.target_acceleration = [0.0] * self.degrees_of_freedom
        self.output = OutputParameter(self.degrees_of_freedom)
        self.initialized = True
        self.result = Result.Finished

    def step(
        self,
        target: Any,
        max_velocity: Any,
        max_acceleration: Any,
        max_jerk: Any,
        *,
        target_velocity: Any | None = None,
        target_acceleration: Any | None = None,
    ) -> Result:
        if not self.initialized:
            raise RuntimeError("Ruckig group must be initialized from measured state")
        self.input.target_position = np.asarray(target, dtype=float).tolist()
        self.input.target_velocity = (
            [0.0] * self.degrees_of_freedom
            if target_velocity is None
            else np.asarray(target_velocity, dtype=float).tolist()
        )
        self.input.target_acceleration = (
            [0.0] * self.degrees_of_freedom
            if target_acceleration is None
            else np.asarray(target_acceleration, dtype=float).tolist()
        )
        self.input.max_velocity = np.asarray(max_velocity, dtype=float).tolist()
        self.input.max_acceleration = np.asarray(max_acceleration, dtype=float).tolist()
        self.input.max_jerk = np.asarray(max_jerk, dtype=float).tolist()
        self.result = self.otg.update(self.input, self.output)
        if self.result.value < 0:
            raise RuntimeError(f"Ruckig update failed: {self.result.name}")
        self.output.pass_to_input(self.input)
        return self.result

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.input.current_position, dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self.input.current_velocity, dtype=float)

    @property
    def acceleration(self) -> np.ndarray:
        return np.asarray(self.input.current_acceleration, dtype=float)

    @property
    def jerk(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros(self.degrees_of_freedom)
        return np.asarray(self.output.new_jerk, dtype=float)


class PhoneControlStack:
    """Own all live filtering, OTG, profile, and tracking-fault state."""

    def __init__(
        self,
        joint_names: list[str] | tuple[str, ...],
        *,
        cycle_s: float = 1.0 / 30.0,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        limits: CommissioningLimits | None = None,
        phone_filter_settings: dict[str, float] | None = None,
        target_velocity_time_constant_s: float = TARGET_VELOCITY_TIME_CONSTANT_S,
        target_velocity_limit_fraction: float = TARGET_VELOCITY_LIMIT_FRACTION,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if tuple(name for name in self.joint_names if name != GRIPPER_JOINT_NAME) != ARM_JOINT_NAMES:
            raise ValueError(f"Expected SO-101 joint order {(*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME)}")
        if GRIPPER_JOINT_NAME not in self.joint_names:
            raise ValueError("SO-101 control stack requires a gripper joint")
        self.arm_joint_names = ARM_JOINT_NAMES
        self.cycle_s = float(cycle_s)
        self.joint_limits = dict(DEFAULT_JOINT_LIMITS_DEG if joint_limits is None else joint_limits)
        if set(self.joint_limits) != set(self.joint_names):
            raise ValueError("Joint limits must cover every SO-101 joint exactly")
        if target_velocity_time_constant_s <= 0.0:
            raise ValueError("Target-velocity time constant must be positive")
        if not 0.0 < target_velocity_limit_fraction <= 1.0:
            raise ValueError("Target-velocity limit fraction must be in (0, 1]")
        self.commissioning_limits = limits or CommissioningLimits()
        self.profile = "Safe"
        self.phone_filter_settings = validated_phone_filter_settings(
            phone_filter_settings or {}, base=DEFAULT_PHONE_FILTER_SETTINGS
        )
        self.phone_filter = OneEuroXYZFilter(**self.phone_filter_settings)
        self.target_velocity_time_constant_s = float(target_velocity_time_constant_s)
        self.target_velocity_limit_fraction = float(target_velocity_limit_fraction)
        self._arm = _RuckigGroup(5, self.cycle_s)
        self._previous_hold = False
        self.hold_active = False
        self.paused_fault = False
        self._arm_fault_cycles = 0
        self._last_valid_target: np.ndarray | None = None
        self._arm_stop_target: np.ndarray | None = None
        self._gripper_command: float | None = None
        self._last_filter_sample: PhoneFilterSample | None = None
        self._previous_velocity_target: np.ndarray | None = None
        self._filtered_target_velocity = np.zeros(len(self.arm_joint_names), dtype=float)
        self.latest: dict[str, Any] = self._empty_snapshot()

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "phone_filter": {},
            "raw_ik_joint_target": {},
            "ruckig": {
                "position": {}, "velocity": {}, "acceleration": {}, "jerk": {},
                "target_velocity": {},
                "result": "Finished", "status": "uninitialized",
            },
            "active_profile": self.profile,
            "filter_settings": dict(self.phone_filter_settings),
            "constraints": self.commissioning_limits.scaled(self.profile),
            "clutch": {
                "active": False,
                "previous_active": False,
                "gripper_active": False,
                "gripper_direction": 0,
            },
            "gripper_direct": {
                "position": None,
                "requested_position": None,
                "direction": 0,
                "status": "uninitialized",
            },
            "tracking": {"errors": {}, "warning": False, "fault": False, "paused": False},
            "target_valid": False,
            "target_rejection": None,
        }

    def reset(self, measured: dict[str, float] | None = None, *, reason: str = "manual") -> None:
        self.phone_filter.reset()
        self._last_filter_sample = None
        self._previous_hold = False
        self.hold_active = False
        self.paused_fault = False
        self._arm_fault_cycles = 0
        self._last_valid_target = None
        self._arm_stop_target = None
        self._gripper_command = None
        self._previous_velocity_target = None
        self._filtered_target_velocity.fill(0.0)
        if measured is not None:
            vector = self._measured_vector(measured)
            self._arm.reset(vector[:5])
            self._gripper_command = float(vector[5])
            self._last_valid_target = vector.copy()
        self.latest = self._empty_snapshot()
        self.latest["ruckig"]["status"] = f"reset:{reason}"

    def set_profile(self, profile: str) -> None:
        if self.hold_active:
            raise HoldActiveError("Release Hold before changing the motion profile")
        if profile not in PROFILE_LIMIT_SCALES:
            raise ValueError(f"Unknown motion profile: {profile}")
        self.profile = profile

    def set_filter_settings(self, settings: dict[str, Any]) -> None:
        if self.hold_active:
            raise HoldActiveError("Release Hold before changing phone filtering")
        validated = validated_phone_filter_settings(
            settings, base=self.phone_filter_settings
        )
        self.phone_filter_settings = validated
        self.phone_filter = OneEuroXYZFilter(**validated)
        self._last_filter_sample = None

    def prepare_phone_action(self, phone_action: dict[str, Any], timestamp_s: float) -> dict[str, Any]:
        """Filter calibrated ``phone.pos`` before LeRobot's Android mapping."""
        prepared = dict(phone_action)
        if "phone.pos" not in prepared:
            raise ValueError("Phone action is missing calibrated phone.pos")
        enabled = bool(prepared.get("phone.enabled", False))
        if enabled and not self._previous_hold:
            self.phone_filter.reset()
        sample = self.phone_filter.update(prepared["phone.pos"], timestamp_s)
        self._last_filter_sample = sample
        prepared["phone.pos"] = np.asarray(sample.deadband_position_m, dtype=float)
        return prepared

    def _measured_vector(self, measured: dict[str, float]) -> np.ndarray:
        try:
            values = np.asarray([float(measured[name]) for name in self.joint_names], dtype=float)
        except KeyError as exc:
            raise ValueError(f"Measured joints are missing {exc.args[0]}") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("Measured joint positions must be finite")
        return values

    def _target_vector(self, target: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
        try:
            values = np.asarray(
                [float(target[f"{name}.pos"]) for name in self.joint_names], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            return None, f"missing or invalid joint target: {exc}"
        if not np.all(np.isfinite(values)):
            return None, "joint target contains a non-finite value"
        for name, value in zip(self.joint_names, values, strict=True):
            lower, upper = self.joint_limits[name]
            if value < lower or value > upper:
                return None, f"{name} target {value:.3f} is outside [{lower:.3f}, {upper:.3f}]"
        return values, None

    def _command_vector(self) -> np.ndarray:
        gripper = 0.0 if self._gripper_command is None else self._gripper_command
        return np.r_[self._arm.position, gripper]

    def _tracking_state(
        self,
        measured: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        errors = np.abs(self._command_vector() - measured)
        arm_peak = float(np.max(errors[:5]))
        warning = arm_peak > 10.0
        if arm_peak > 15.0:
            self._arm_fault_cycles += 1
        else:
            self._arm_fault_cycles = 0
        arm_fault = self._arm_fault_cycles >= 3
        return errors, warning, arm_fault

    def _quick_stop_target(self) -> np.ndarray:
        limits = self.commissioning_limits.scaled(self.profile)
        position = self._arm.position
        velocity = self._arm.velocity
        acceleration_limits = np.asarray(limits["arm_acceleration"], dtype=float)
        target = position + np.sign(velocity) * velocity**2 / (2.0 * acceleration_limits)
        for index, name in enumerate(self.arm_joint_names):
            lower, upper = self.joint_limits[name]
            # A measured pose can already be outside a newly tightened model
            # envelope (for example, an old saved base pose). Releasing Hold
            # must never create motion merely to enforce that envelope.
            if position[index] < lower or position[index] > upper:
                target[index] = position[index]
            else:
                target[index] = float(np.clip(target[index], lower, upper))
        return target

    def _moving_target_velocity(
        self,
        target: np.ndarray,
        active_limits: dict[str, list[float]],
        *,
        reset: bool,
    ) -> np.ndarray:
        """Estimate a bounded velocity for a continuously moving IK target."""
        if reset or self._previous_velocity_target is None:
            self._filtered_target_velocity.fill(0.0)
        else:
            raw_velocity = (target - self._previous_velocity_target) / self.cycle_s
            max_velocity = np.asarray(active_limits["arm_velocity"], dtype=float)
            bounded_velocity = np.clip(
                raw_velocity,
                -max_velocity * self.target_velocity_limit_fraction,
                max_velocity * self.target_velocity_limit_fraction,
            )
            alpha = 1.0 - math.exp(
                -self.cycle_s / self.target_velocity_time_constant_s
            )
            self._filtered_target_velocity += alpha * (
                bounded_velocity - self._filtered_target_velocity
            )

        # Do not request an outward terminal velocity close to a joint limit.
        for index, name in enumerate(self.arm_joint_names):
            lower, upper = self.joint_limits[name]
            margin = TARGET_VELOCITY_LIMIT_MARGIN[index]
            if target[index] >= upper - margin and self._filtered_target_velocity[index] > 0:
                self._filtered_target_velocity[index] = 0.0
            if target[index] <= lower + margin and self._filtered_target_velocity[index] < 0:
                self._filtered_target_velocity[index] = 0.0

        self._previous_velocity_target = target.copy()
        return self._filtered_target_velocity.copy()

    @staticmethod
    def _result_name(result: Result) -> str:
        return result.name if hasattr(result, "name") else str(result)

    def step(
        self,
        measured: dict[str, float],
        raw_joint_target: dict[str, Any],
        *,
        hold_active: bool,
        gripper_active: bool = False,
        gripper_direction: int = 0,
    ) -> dict[str, float]:
        measured_vector = self._measured_vector(measured)
        if not self._arm.initialized:
            self.reset(measured, reason="measured_initialization")

        previous_hold = self._previous_hold
        self.hold_active = bool(hold_active)
        gripper_direction = int(np.sign(gripper_direction))
        gripper_button_active = bool(gripper_active) or gripper_direction != 0

        target, rejection = self._target_vector(raw_joint_target)
        target_valid = target is not None
        if target is not None:
            raw_target_payload = dict(zip(self.joint_names, target.tolist(), strict=True))
        else:
            raw_target_payload = {
                name: raw_joint_target.get(f"{name}.pos") for name in self.joint_names
            }

        requested_gripper: float | None = None
        try:
            candidate = float(raw_joint_target[f"{GRIPPER_JOINT_NAME}.pos"])
            lower, upper = self.joint_limits[GRIPPER_JOINT_NAME]
            if math.isfinite(candidate) and lower <= candidate <= upper:
                requested_gripper = candidate
        except (KeyError, TypeError, ValueError):
            pass

        # The gripper intentionally bypasses Ruckig. LeRobot converts A/B into
        # a bounded position target, which is sent directly to the servo. With
        # neither button pressed, retain the last commanded position.
        if gripper_button_active and requested_gripper is not None:
            self._gripper_command = requested_gripper
            gripper_status = "direct_button"
        elif gripper_button_active:
            gripper_status = "invalid_target_hold"
        else:
            gripper_status = "direct_hold"
        if self._gripper_command is None:
            self._gripper_command = float(measured_vector[5])

        errors, warning, arm_fault = self._tracking_state(measured_vector)
        status = "tracking"

        if arm_fault:
            self.paused_fault = True
            self.phone_filter.reset()
            self._arm.reset(measured_vector[:5])
            self._last_valid_target = measured_vector.copy()
            self._arm_stop_target = measured_vector[:5].copy()
            status = "paused_tracking_fault"

        if self.paused_fault:
            if not self.hold_active:
                self.paused_fault = False
                self._arm_fault_cycles = 0
                self._arm.reset(measured_vector[:5])
                self._last_valid_target = measured_vector.copy()
                self._arm_stop_target = measured_vector[:5].copy()
                status = "fault_recovered_after_release"
            else:
                status = "paused_tracking_fault"

        active_limits = self.commissioning_limits.scaled(self.profile)
        if not self.paused_fault:
            if self.hold_active:
                if target is not None:
                    self._last_valid_target = target.copy()
                target_to_use = (
                    self._last_valid_target[:5].copy()
                    if self._last_valid_target is not None
                    else measured_vector[:5].copy()
                )
                self._arm_stop_target = None
                target_velocity = self._moving_target_velocity(
                    target_to_use,
                    active_limits,
                    reset=not previous_hold,
                )
                status = "tracking" if target_valid else "holding_last_valid_target"
            else:
                if previous_hold or self._arm_stop_target is None:
                    self._arm_stop_target = self._quick_stop_target()
                target_to_use = self._arm_stop_target.copy()
                target_velocity = np.zeros(len(self.arm_joint_names), dtype=float)
                self._previous_velocity_target = None
                self._filtered_target_velocity.fill(0.0)
                status = "quick_stop" if previous_hold else "released_hold"

            arm_result = self._arm.step(
                target_to_use,
                active_limits["arm_velocity"],
                active_limits["arm_acceleration"],
                active_limits["arm_jerk"],
                target_velocity=target_velocity,
            )
        else:
            arm_result = Result.Finished
            target_velocity = np.zeros(len(self.arm_joint_names), dtype=float)

        command = self._command_vector()
        self.latest = {
            "phone_filter": self._last_filter_sample.to_dict() if self._last_filter_sample else {},
            "raw_ik_joint_target": raw_target_payload,
            "ruckig": {
                "position": dict(
                    zip(self.arm_joint_names, self._arm.position.tolist(), strict=True)
                ),
                "velocity": dict(
                    zip(self.arm_joint_names, self._arm.velocity.tolist(), strict=True)
                ),
                "acceleration": dict(
                    zip(self.arm_joint_names, self._arm.acceleration.tolist(), strict=True)
                ),
                "jerk": dict(
                    zip(self.arm_joint_names, self._arm.jerk.tolist(), strict=True)
                ),
                "target_velocity": dict(
                    zip(self.arm_joint_names, target_velocity.tolist(), strict=True)
                ),
                "result": f"arm={self._result_name(arm_result)}",
                "status": status,
            },
            "gripper_direct": {
                "position": self._gripper_command,
                "requested_position": requested_gripper,
                "direction": gripper_direction,
                "status": gripper_status,
            },
            "active_profile": self.profile,
            "filter_settings": dict(self.phone_filter_settings),
            "constraints": active_limits,
            "clutch": {
                "active": self.hold_active,
                "previous_active": previous_hold,
                "gripper_active": gripper_button_active,
                "gripper_direction": gripper_direction,
            },
            "tracking": {
                "errors": dict(zip(self.joint_names, errors.tolist(), strict=True)),
                "warning": warning,
                "fault": arm_fault,
                "arm_fault": arm_fault,
                "arm_fault_cycles": self._arm_fault_cycles,
                "paused": self.paused_fault,
            },
            "target_valid": target_valid,
            "target_rejection": rejection,
        }
        self._previous_hold = self.hold_active
        return {f"{name}.pos": float(value) for name, value in zip(self.joint_names, command, strict=True)}
