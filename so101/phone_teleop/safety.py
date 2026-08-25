"""Conservative phone-control safety helpers for the SO-101."""

from __future__ import annotations

from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import RobotActionProcessorStep


ROTATION_TARGET_KEYS = ("target_wx", "target_wy", "target_wz")


class TranslationOnlyPhoneControl(RobotActionProcessorStep):
    """Ignore phone rotation while preserving XYZ translation and gripper input.

    ``EEReferenceAndDelta`` receives zero rotation deltas, so it keeps the
    orientation measured when Hold to move is engaged instead of following
    subsequent phone rotation.
    """

    def action(self, action: RobotAction) -> RobotAction:
        missing = [key for key in ROTATION_TARGET_KEYS if key not in action]
        if missing:
            raise ValueError(
                "Phone action is missing orientation component(s): "
                + ", ".join(missing)
            )
        for key in ROTATION_TARGET_KEYS:
            action[key] = 0.0
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def configure_servo_acceleration(
    bus: Any,
    motor_names: list[str],
    acceleration: int,
    *,
    num_retry: int = 2,
) -> dict[str, int]:
    """Set and verify the volatile STS3215 acceleration register."""
    if not 1 <= acceleration <= 254:
        raise ValueError("Servo acceleration must be between 1 and 254")
    if not motor_names:
        raise ValueError("At least one motor is required")

    values = dict.fromkeys(motor_names, acceleration)
    bus.sync_write("Acceleration", values, normalize=False, num_retry=num_retry)
    readback = {
        motor: int(value)
        for motor, value in bus.sync_read(
            "Acceleration", normalize=False, num_retry=num_retry
        ).items()
    }
    mismatched = {
        motor: value for motor, value in readback.items() if value != acceleration
    }
    if set(readback) != set(motor_names) or mismatched:
        raise RuntimeError(
            "Could not verify servo acceleration: "
            f"expected={acceleration}, readback={readback}"
        )
    return readback
