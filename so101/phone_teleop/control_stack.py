"""Phone-specific filtering layered on the shared SO-101 motion stack."""

from __future__ import annotations

from typing import Any

import numpy as np

from so101.teleop.control_stack import (
    ARM_JOINT_NAMES,
    DEFAULT_JOINT_LIMITS_DEG,
    GRIPPER_JOINT_NAME,
    PROFILE_LIMIT_SCALES,
    PROFILE_NAMES,
    CommissioningLimits,
    HoldActiveError,
    SO101ControlStack,
)

from .filtering import (
    DEFAULT_PHONE_FILTER_SETTINGS,
    OneEuroXYZFilter,
    PhoneFilterSample,
    validated_phone_filter_settings,
)


class PhoneControlStack(SO101ControlStack):
    """Add calibrated-phone One-Euro filtering to the shared motion stack."""

    def __init__(
        self,
        joint_names: list[str] | tuple[str, ...],
        *,
        phone_filter_settings: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(joint_names, **kwargs)
        self.phone_filter_settings = validated_phone_filter_settings(
            phone_filter_settings or {}, base=DEFAULT_PHONE_FILTER_SETTINGS
        )
        self.phone_filter = OneEuroXYZFilter(**self.phone_filter_settings)
        self._last_filter_sample: PhoneFilterSample | None = None
        self._publish_phone_state()

    def _publish_phone_state(self) -> None:
        self.latest["phone_filter"] = (
            self._last_filter_sample.to_dict() if self._last_filter_sample else {}
        )
        self.latest["filter_settings"] = dict(self.phone_filter_settings)

    def reset(self, measured: dict[str, float] | None = None, *, reason: str = "manual") -> None:
        self.phone_filter.reset()
        self._last_filter_sample = None
        super().reset(measured, reason=reason)
        self._publish_phone_state()

    def set_filter_settings(self, settings: dict[str, Any]) -> None:
        if self.hold_active:
            raise HoldActiveError("Release Hold before changing phone filtering")
        validated = validated_phone_filter_settings(
            settings, base=self.phone_filter_settings
        )
        self.phone_filter_settings = validated
        self.phone_filter = OneEuroXYZFilter(**validated)
        self._last_filter_sample = None
        self._publish_phone_state()

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

    def step(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        command = super().step(*args, **kwargs)
        if self.latest.get("tracking", {}).get("fault"):
            self.phone_filter.reset()
        self._publish_phone_state()
        return command


__all__ = [
    "ARM_JOINT_NAMES",
    "DEFAULT_JOINT_LIMITS_DEG",
    "GRIPPER_JOINT_NAME",
    "PROFILE_LIMIT_SCALES",
    "PROFILE_NAMES",
    "CommissioningLimits",
    "HoldActiveError",
    "PhoneControlStack",
    "SO101ControlStack",
]
