"""Read shared LeRobot/STS3215 calibration without opening the servo bus."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STS3215_RESOLUTION = 4096


@dataclass(frozen=True)
class MotorCalibrationInfo:
    name: str
    motor_id: int
    drive_mode: int
    homing_offset_counts: int
    range_min_counts: int
    range_max_counts: int
    normalized_min: float
    normalized_max: float
    normalized_unit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_motor_calibration(path: Path, joint_names: list[str] | tuple[str, ...]) -> dict[str, MotorCalibrationInfo]:
    """Convert raw calibration endpoints to the values LeRobot exposes.

    Arm joints configured with ``use_degrees=True`` use the midpoint of their
    recorded raw range as zero and 4095 counts per 360 degrees. The gripper is
    normalized linearly to 0..100. These recorded ranges describe this arm's
    calibrated servo travel. Callers may either use that physical envelope
    directly or intersect it with model limits when the coordinate frames are
    known to share the same zero.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != set(joint_names):
        raise ValueError(f"Calibration must contain exactly these joints: {list(joint_names)}")
    parsed: dict[str, MotorCalibrationInfo] = {}
    for name in joint_names:
        row = payload[name]
        required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"Calibration entry is incomplete: {name}")
        minimum = int(row["range_min"])
        maximum = int(row["range_max"])
        if not 0 <= minimum < maximum < STS3215_RESOLUTION:
            raise ValueError(f"Calibration range is invalid for {name}: {minimum}..{maximum}")
        if name == "gripper":
            normalized_min, normalized_max, unit = 0.0, 100.0, "percent"
        else:
            half_range_degrees = (maximum - minimum) * 180.0 / (STS3215_RESOLUTION - 1)
            normalized_min, normalized_max, unit = -half_range_degrees, half_range_degrees, "degrees"
        parsed[name] = MotorCalibrationInfo(
            name=name,
            motor_id=int(row["id"]),
            drive_mode=int(row["drive_mode"]),
            homing_offset_counts=int(row["homing_offset"]),
            range_min_counts=minimum,
            range_max_counts=maximum,
            normalized_min=normalized_min,
            normalized_max=normalized_max,
            normalized_unit=unit,
        )
    ids = [info.motor_id for info in parsed.values()]
    if len(set(ids)) != len(ids):
        raise ValueError("Calibration motor IDs must be unique")
    return parsed


def effective_joint_limits(
    urdf_limits_deg: dict[str, tuple[float, float]],
    calibration: dict[str, MotorCalibrationInfo],
) -> dict[str, tuple[float, float]]:
    """Intersect calibrated servo travel with the authoritative URDF limits."""
    if set(urdf_limits_deg) - {"gripper"} != set(calibration) - {"gripper"}:
        raise ValueError("URDF and calibration arm joints differ")
    limits: dict[str, tuple[float, float]] = {}
    for name, info in calibration.items():
        if name == "gripper":
            limits[name] = (0.0, 100.0)
            continue
        urdf_min, urdf_max = urdf_limits_deg[name]
        lower = max(float(urdf_min), info.normalized_min)
        upper = min(float(urdf_max), info.normalized_max)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"URDF and calibration have no usable overlap for {name}")
        limits[name] = (lower, upper)
    return limits


def calibrated_joint_limits(
    calibration: dict[str, MotorCalibrationInfo],
) -> dict[str, tuple[float, float]]:
    """Return limits in the normalized coordinates exposed by this motor bus.

    LeRobot's degree normalization is derived from each recorded servo range.
    Those coordinates can be offset from the CAD model's nominal zero, so the
    physical direct-control path must use the calibrated envelope when deciding
    whether a command would drive a servo past its recorded endpoints.
    """
    return {
        name: (
            (0.0, 100.0)
            if name == "gripper"
            else (float(info.normalized_min), float(info.normalized_max))
        )
        for name, info in calibration.items()
    }


def positions_outside_limits(
    positions: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    outside: dict[str, dict[str, float]] = {}
    for name, (lower, upper) in limits.items():
        value = float(positions[name])
        if value < lower or value > upper:
            outside[name] = {"position": value, "minimum": lower, "maximum": upper}
    return outside
