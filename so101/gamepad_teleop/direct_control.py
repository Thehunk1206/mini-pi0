"""Validated direct joint commands for gamepad teleoperation.

This deliberately contains no online trajectory generator. Stick velocities
are integrated upstream into small target changes; valid IK/direct-joint
targets are then sent to the servos as position commands, following the
reference joystick controller's atomic update model.
"""

from __future__ import annotations

import math
from typing import Any


class DirectGamepadControl:
    """Validate and atomically accept direct gamepad joint targets."""

    def __init__(
        self,
        joint_names: list[str],
        *,
        joint_limits: dict[str, tuple[float, float]],
        max_target_step_deg: tuple[float, ...],
        joint_limit_feedback_margin_deg: float = 5.0,
        elbow_singularity_deg: float = 0.0,
    ) -> None:
        self.joint_names = list(joint_names)
        self.arm_joint_names = [
            name for name in self.joint_names if name != "gripper"
        ]
        self.joint_limits = dict(joint_limits)
        if len(max_target_step_deg) != len(self.arm_joint_names):
            raise ValueError("one direct target-step limit is required per arm joint")
        self.max_target_step_deg = tuple(
            float(value) for value in max_target_step_deg
        )
        self.joint_limit_feedback_margin_deg = float(
            joint_limit_feedback_margin_deg
        )
        self.elbow_singularity_deg = float(elbow_singularity_deg)
        if not math.isfinite(self.elbow_singularity_deg):
            raise ValueError("elbow singularity must be finite")
        if self.joint_limit_feedback_margin_deg < 0.0:
            raise ValueError("joint-limit feedback margin must be non-negative")
        self._last_valid_target: dict[str, float] = {}
        self.latest: dict[str, Any] = {}

    def reset(self, measured_positions: dict[str, float], *, reason: str) -> None:
        initial_target: dict[str, float] = {}
        for name in self.joint_names:
            value = float(measured_positions[name])
            lower, upper = self.joint_limits[name]
            if not math.isfinite(value) or value < lower - 1e-3 or value > upper + 1e-3:
                raise ValueError(
                    f"measured {name} position {value:.2f} is outside calibrated "
                    f"range [{lower:.2f}, {upper:.2f}]"
                )
            initial_target[name] = min(max(value, lower), upper)
        self._last_valid_target = initial_target
        self.latest = {
            "mode": "direct",
            "status": "direct_ready",
            "reason": reason,
            "target_valid": True,
            "target_rejection": None,
            "safety_event": None,
            "raw_ik_joint_target": dict(self._last_valid_target),
        }

    def _validate_values(
        self, requested_action: dict[str, float]
    ) -> tuple[dict[str, float], str | None, str | None]:
        target: dict[str, float] = {}
        for name in self.joint_names:
            key = f"{name}.pos"
            if key not in requested_action:
                return {}, f"missing target {key}", "invalid_target"
            value = float(requested_action[key])
            if not math.isfinite(value):
                return {}, f"non-finite target for {name}", "invalid_target"
            lower, upper = self.joint_limits[name]
            tolerance = 1e-3
            if value < lower - tolerance or value > upper + tolerance:
                return (
                    {},
                    f"{name} target {value:.2f} outside [{lower:.2f}, {upper:.2f}]",
                    "joint_limit",
                )
            target[name] = min(max(value, lower), upper)

        return target, None, None

    def _validate_steps(
        self, target: dict[str, float]
    ) -> tuple[str | None, str | None]:
        for name, maximum in zip(
            self.arm_joint_names, self.max_target_step_deg, strict=True
        ):
            previous = self._last_valid_target.get(name, target[name])
            step = abs(target[name] - previous)
            if step > maximum:
                return (
                    f"{name} IK jump {step:.2f} deg exceeds {maximum:.2f} deg/cycle",
                    "ik_jump",
                )
        return None, None

    def _arm_joints_moving_into_limit(self, target: dict[str, float]) -> list[str]:
        """Return only joints actively moving farther into an endpoint band."""

        margin = self.joint_limit_feedback_margin_deg
        result: list[str] = []
        for name in self.arm_joint_names:
            previous = self._last_valid_target.get(name, target[name])
            delta = target[name] - previous
            lower, upper = self.joint_limits[name]
            moving_into_lower = delta < -1e-9 and target[name] <= lower + margin
            moving_into_upper = delta > 1e-9 and target[name] >= upper - margin
            if moving_into_lower or moving_into_upper:
                result.append(name)
        return result

    def step(
        self,
        measured_positions: dict[str, float],
        requested_action: dict[str, float],
        *,
        arm_input_active: bool = True,
    ) -> dict[str, float]:
        if set(self._last_valid_target) != set(self.joint_names):
            raise RuntimeError("direct gamepad control must be reset from measured state")
        target, rejection, safety_event = self._validate_values(requested_action)
        command = {
            name: self._last_valid_target[name] for name in self.joint_names
        }

        extension_limited = False
        if rejection is None and arm_input_active:
            previous_elbow = self._last_valid_target.get(
                "elbow_flex", target.get("elbow_flex", 0.0)
            )
            requested_elbow = target.get("elbow_flex", previous_elbow)
            extension_limited = (
                (previous_elbow - self.elbow_singularity_deg)
                * (requested_elbow - self.elbow_singularity_deg)
                < 0.0
            )
            if extension_limited:
                safety_event = "workspace_limit"

                # Shoulder lift and elbow are the position-IK pair. Hold only
                # those two joints at their last safe solution while allowing
                # independent pan and wrist commands to proceed. The caller
                # rolls back this cycle's Cartesian increment so the blocked
                # IK target cannot accumulate into a later jump rejection.
                for name in ("shoulder_lift", "elbow_flex"):
                    target[name] = self._last_valid_target[name]

        if rejection is None:
            rejection, step_safety_event = self._validate_steps(target)
            if step_safety_event is not None:
                safety_event = step_safety_event

        joint_limit_joints: list[str] = []
        if rejection is None and arm_input_active:
            # Evaluate direction before accepting the new target. A joint that
            # merely sits near an endpoint (including a saved base pose) must
            # not vibrate while an unrelated axis is commanded.
            joint_limit_joints = self._arm_joints_moving_into_limit(target)
            for name in self.arm_joint_names:
                command[name] = target[name]
                self._last_valid_target[name] = target[name]
        elif rejection is None:
            for name in self.arm_joint_names:
                command[name] = self._last_valid_target.get(
                    name, float(measured_positions[name])
                )

        # Invalid targets are discarded atomically, matching the reference
        # joystick controller. Otherwise LT/RT updates the gripper directly.
        if rejection is None:
            command["gripper"] = target["gripper"]
            self._last_valid_target["gripper"] = target["gripper"]

        if joint_limit_joints:
            safety_event = "joint_limit"

        status = "direct_tracking"
        if rejection is not None:
            status = "target_rejected"
        elif extension_limited:
            status = "extension_limited"
        self.latest = {
            "mode": "direct",
            "status": status,
            "target_valid": rejection is None,
            "target_rejection": rejection,
            "safety_event": safety_event,
            "raw_ik_joint_target": {
                name: requested_action.get(f"{name}.pos") for name in self.joint_names
            },
            "joint_limit_joints": joint_limit_joints,
            "extension_limited": extension_limited,
            "elbow_singularity_deg": self.elbow_singularity_deg,
            "extension_held_joints": (
                ["shoulder_lift", "elbow_flex"] if extension_limited else []
            ),
        }
        return {f"{name}.pos": value for name, value in command.items()}
