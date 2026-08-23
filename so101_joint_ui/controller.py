from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


LOGGER = logging.getLogger(__name__)

BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
JOINTS = (*BODY_JOINTS, "gripper")
ENCODER_MAX = 4095
CURRENT_MA_PER_COUNT = 6.5
LOAD_PERCENT_PER_COUNT = 0.1
BASE_JOINT_TOLERANCE_DEG = 1.0
BASE_GRIPPER_TOLERANCE_PERCENT = 2.0


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float
    unit: str


def load_calibration_file(path: Path) -> dict[str, MotorCalibration]:
    """Load and validate a LeRobot SO-101 calibration JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with path.open() as handle:
        raw = json.load(handle)

    missing = [joint for joint in JOINTS if joint not in raw]
    if missing:
        raise ValueError(f"Calibration is missing joints: {', '.join(missing)}")

    calibration: dict[str, MotorCalibration] = {}
    expected_ids = dict(zip(JOINTS, range(1, 7), strict=True))
    for joint in JOINTS:
        entry = raw[joint]
        calibration[joint] = MotorCalibration(
            id=int(entry["id"]),
            drive_mode=int(entry["drive_mode"]),
            homing_offset=int(entry["homing_offset"]),
            range_min=int(entry["range_min"]),
            range_max=int(entry["range_max"]),
        )
        cal = calibration[joint]
        if cal.id != expected_ids[joint]:
            raise ValueError(
                f"Calibration ID mismatch for {joint}: expected {expected_ids[joint]}, got {cal.id}"
            )
        if not 0 <= cal.range_min < cal.range_max <= ENCODER_MAX:
            raise ValueError(
                f"Invalid calibrated range for {joint}: {cal.range_min}..{cal.range_max}"
            )

    return calibration


def calibrated_joint_limits(
    calibration: dict[str, MotorCalibration],
) -> dict[str, JointLimit]:
    """Return UI limits in the same normalized units used by SO101Follower."""
    limits: dict[str, JointLimit] = {}
    for joint in BODY_JOINTS:
        cal = calibration[joint]
        midpoint = (cal.range_min + cal.range_max) / 2
        limits[joint] = JointLimit(
            minimum=(cal.range_min - midpoint) * 360 / ENCODER_MAX,
            maximum=(cal.range_max - midpoint) * 360 / ENCODER_MAX,
            unit="°",
        )
    limits["gripper"] = JointLimit(minimum=0.0, maximum=100.0, unit="%")
    return limits


def bounded_step(current: float, target: float, maximum_step: float) -> float:
    """Advance toward target by no more than maximum_step."""
    delta = target - current
    if abs(delta) <= maximum_step:
        return target
    return current + maximum_step * (1 if delta > 0 else -1)


def current_raw_to_ma(value: int | float) -> float:
    """Convert an STS3215 Present_Current register value to milliamps."""
    return abs(float(value)) * CURRENT_MA_PER_COUNT


def load_raw_to_percent(value: int | float) -> float:
    """Convert signed STS3215 Present_Load effort to percent of output."""
    return float(value) * LOAD_PERCENT_PER_COUNT


def default_base_position_file(robot_id: str) -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "base_positions"
        / "robots"
        / "so_follower"
        / f"{robot_id}.json"
    )


def validate_base_position(
    positions: dict[str, Any], limits: dict[str, JointLimit]
) -> dict[str, float]:
    """Validate a complete base pose in the UI's normalized motor units."""
    missing = [joint for joint in JOINTS if joint not in positions]
    extra = [joint for joint in positions if joint not in JOINTS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ValueError("Invalid base position (" + "; ".join(details) + ")")

    validated: dict[str, float] = {}
    for joint in JOINTS:
        value = float(positions[joint])
        if not math.isfinite(value):
            raise ValueError(f"Invalid base position for {joint}: {value}")
        limit = limits[joint]
        if not limit.minimum <= value <= limit.maximum:
            raise ValueError(
                f"Base position for {joint} is outside calibrated limits: "
                f"{value} not in {limit.minimum}..{limit.maximum}"
            )
        validated[joint] = value
    return validated


def load_base_position_file(
    path: Path, limits: dict[str, JointLimit]
) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    positions = payload.get("positions") if isinstance(payload, dict) else None
    if not isinstance(positions, dict):
        raise ValueError(f"Base-position file has no positions object: {path}")
    return validate_base_position(positions, limits)


def save_base_position_file(
    path: Path, robot_id: str, positions: dict[str, float]
) -> None:
    """Atomically persist a captured base pose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot_id": robot_id,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "positions": positions,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def positions_reached(
    positions: dict[str, float],
    targets: dict[str, float],
    *,
    joint_tolerance_deg: float = BASE_JOINT_TOLERANCE_DEG,
    gripper_tolerance_percent: float = BASE_GRIPPER_TOLERANCE_PERCENT,
) -> bool:
    return all(
        abs(positions[joint] - targets[joint])
        <= (gripper_tolerance_percent if joint == "gripper" else joint_tolerance_deg)
        for joint in JOINTS
    )


def advance_commanded_positions(
    commanded: dict[str, float],
    targets: dict[str, float],
    max_joint_step: float,
    max_gripper_step: float,
) -> dict[str, float]:
    """Advance stored command setpoints independently of measured positions.

    A commanded trajectory must accumulate even when a loaded joint initially
    stays still. Re-anchoring each step to measured position can leave too
    little position error for the servo to overcome static friction.
    """
    return {
        joint: bounded_step(
            commanded[joint],
            targets[joint],
            max_gripper_step if joint == "gripper" else max_joint_step,
        )
        for joint in JOINTS
    }


class RobotController:
    """Thread-safe SO-101 controller with a bounded-rate target loop."""

    def __init__(
        self,
        calibration_file: Path,
        default_serial_port: str = "",
        *,
        base_position_file: Path | None = None,
        control_hz: float = 20.0,
        max_joint_speed_deg_s: float = 45.0,
        max_gripper_speed_percent_s: float = 35.0,
        max_following_error_deg: float = 15.0,
        max_gripper_following_error_percent: float = 20.0,
        max_temperature_c: int = 55,
    ) -> None:
        self.calibration_file = calibration_file.expanduser().resolve()
        self.robot_id = self.calibration_file.stem
        self.calibration = load_calibration_file(self.calibration_file)
        self.limits = calibrated_joint_limits(self.calibration)
        self.base_position_file = (
            base_position_file.expanduser().resolve()
            if base_position_file is not None
            else default_base_position_file(self.robot_id)
        )
        self.default_serial_port = default_serial_port
        self.control_hz = control_hz
        self.max_joint_step = max_joint_speed_deg_s / control_hz
        self.max_gripper_step = max_gripper_speed_percent_s / control_hz
        self.max_following_error_deg = max_following_error_deg
        self.max_gripper_following_error_percent = max_gripper_following_error_percent
        self.max_temperature_c = max_temperature_c

        self._lock = threading.RLock()
        self._robot: SO101Follower | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._serial_port = default_serial_port
        self._positions = {joint: 0.0 for joint in JOINTS}
        self._targets = dict(self._positions)
        self._commanded_positions = dict(self._positions)
        self._following_errors = dict(self._positions)
        self._temperatures: dict[str, int] = {}
        self._voltages: dict[str, int] = {}
        self._currents_raw: dict[str, int] = {}
        self._currents_ma: dict[str, float] = {}
        self._loads_raw: dict[str, int] = {}
        self._loads_percent: dict[str, float] = {}
        self._torque_enabled = False
        self._last_error: str | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=80)
        self._telemetry_log: deque[dict[str, Any]] = deque(maxlen=120)
        self._telemetry_sequence = 0
        self._base_position: dict[str, float] | None = None
        self._returning_to_base = False
        self._event("info", f"Loaded calibration for '{self.robot_id}'")
        try:
            self._base_position = load_base_position_file(self.base_position_file, self.limits)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._event("error", f"Could not load base position: {exc}")
        else:
            if self._base_position is not None:
                self._event("info", f"Loaded base position from {self.base_position_file}")

    @property
    def is_connected(self) -> bool:
        robot = self._robot
        return bool(robot and robot.bus.is_connected)

    def _event(self, level: str, message: str) -> None:
        self._events.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        getattr(LOGGER, level if level in {"info", "warning", "error"} else "info")(message)

    def _validate_raw_positions(self, positions: dict[str, int]) -> None:
        outside = []
        for joint, position in positions.items():
            cal = self.calibration[joint]
            if not cal.range_min <= position <= cal.range_max:
                outside.append(f"{joint}={position} outside {cal.range_min}..{cal.range_max}")
        if outside:
            raise RuntimeError("Unsafe motor position: " + "; ".join(outside))

    def connect(self, serial_port: str | None = None) -> dict[str, Any]:
        serial_port = (serial_port or self.default_serial_port).strip()
        if not serial_port:
            raise ValueError("A serial port is required")

        with self._lock:
            if self.is_connected:
                return self.snapshot()

            config = SO101FollowerConfig(
                port=serial_port,
                id=self.robot_id,
                calibration_dir=self.calibration_file.parent,
                cameras={},
                use_degrees=True,
                disable_torque_on_disconnect=True,
                max_relative_target=5.0,
                num_read_retries=2,
            )
            robot = SO101Follower(config)
            try:
                robot.bus.connect()
                if not robot.bus.is_calibrated:
                    raise RuntimeError(
                        "Motor calibration does not match the saved calibration file. "
                        "Reconnect with the LeRobot calibration command before using the UI."
                    )

                raw_positions = robot.bus.sync_read(
                    "Present_Position", normalize=False, num_retry=2
                )
                self._validate_raw_positions(raw_positions)

                # Configure while released, align every stored goal with the measured pose,
                # then enable holding torque. This avoids a jump toward stale goals.
                robot.bus.disable_torque(num_retry=2)
                robot.bus.configure_motors(maximum_acceleration=100, acceleration=50)
                for joint in JOINTS:
                    robot.bus.write("Operating_Mode", joint, OperatingMode.POSITION.value)
                    robot.bus.write("P_Coefficient", joint, config.position_p_coefficient)
                    robot.bus.write("I_Coefficient", joint, config.position_i_coefficient)
                    robot.bus.write("D_Coefficient", joint, config.position_d_coefficient)
                robot.bus.write("Max_Torque_Limit", "gripper", 500)
                robot.bus.write("Protection_Current", "gripper", 250)
                robot.bus.write("Overload_Torque", "gripper", 25)

                positions = robot.bus.sync_read("Present_Position", num_retry=2)
                robot.bus.sync_write("Goal_Position", positions)
                robot.bus.enable_torque(num_retry=2)
            except Exception:
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=True)
                raise

            self._robot = robot
            self._serial_port = serial_port
            self._positions = {joint: float(positions[joint]) for joint in JOINTS}
            self._targets = dict(self._positions)
            self._commanded_positions = dict(self._positions)
            self._following_errors = {joint: 0.0 for joint in JOINTS}
            self._torque_enabled = True
            self._last_error = None
            self._start_worker_locked()
            self._event("info", f"Connected to {serial_port}; holding current pose")
            return self.snapshot()

    def _start_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._control_loop,
            name="so101-joint-control",
            daemon=True,
        )
        self._worker.start()

    def _control_loop(self) -> None:
        period = 1.0 / self.control_hz
        next_telemetry = 0.0
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                with self._lock:
                    robot = self._robot
                    if robot is None or not robot.bus.is_connected:
                        return

                    positions = robot.bus.sync_read("Present_Position", num_retry=2)
                    self._positions = {joint: float(positions[joint]) for joint in JOINTS}

                    if self._torque_enabled:
                        command = advance_commanded_positions(
                            self._commanded_positions,
                            self._targets,
                            self.max_joint_step,
                            self.max_gripper_step,
                        )
                        following_errors = {
                            joint: command[joint] - self._positions[joint] for joint in JOINTS
                        }
                        unsafe_errors = {
                            joint: error
                            for joint, error in following_errors.items()
                            if abs(error)
                            > (
                                self.max_gripper_following_error_percent
                                if joint == "gripper"
                                else self.max_following_error_deg
                            )
                        }
                        if unsafe_errors:
                            robot.bus.disable_torque(num_retry=2)
                            self._torque_enabled = False
                            self._returning_to_base = False
                            self._targets = dict(self._positions)
                            self._commanded_positions = dict(self._positions)
                            details = ", ".join(
                                f"{joint}={error:+.1f}" for joint, error in unsafe_errors.items()
                            )
                            self._last_error = f"Following-error safety stop: {details}"
                            self._event("error", self._last_error)
                        else:
                            robot.bus.sync_write("Goal_Position", command)
                            self._commanded_positions = command
                            self._following_errors = following_errors
                            if (
                                self._returning_to_base
                                and self._base_position is not None
                                and positions_reached(self._positions, self._base_position)
                                and positions_reached(command, self._base_position)
                            ):
                                self._returning_to_base = False
                                self._event("info", "Base position reached; holding torque")

                    now = time.monotonic()
                    if now >= next_telemetry:
                        self._temperatures = {
                            joint: int(value)
                            for joint, value in robot.bus.sync_read(
                                "Present_Temperature", normalize=False, num_retry=2
                            ).items()
                        }
                        self._voltages = {
                            joint: int(value)
                            for joint, value in robot.bus.sync_read(
                                "Present_Voltage", normalize=False, num_retry=2
                            ).items()
                        }
                        self._currents_raw = {
                            joint: int(value)
                            for joint, value in robot.bus.sync_read(
                                "Present_Current", normalize=False, num_retry=2
                            ).items()
                        }
                        self._currents_ma = {
                            joint: current_raw_to_ma(value)
                            for joint, value in self._currents_raw.items()
                        }
                        self._loads_raw = {
                            joint: int(value)
                            for joint, value in robot.bus.sync_read(
                                "Present_Load", normalize=False, num_retry=2
                            ).items()
                        }
                        self._loads_percent = {
                            joint: load_raw_to_percent(value)
                            for joint, value in self._loads_raw.items()
                        }
                        self._telemetry_sequence += 1
                        self._telemetry_log.append(
                            {
                                "sequence": self._telemetry_sequence,
                                "time": time.strftime("%H:%M:%S"),
                                "currents_ma": dict(self._currents_ma),
                                "loads_percent": dict(self._loads_percent),
                            }
                        )
                        hottest = max(self._temperatures.values(), default=0)
                        if hottest >= self.max_temperature_c:
                            robot.bus.disable_torque(num_retry=2)
                            self._torque_enabled = False
                            self._returning_to_base = False
                            self._targets = dict(self._positions)
                            self._commanded_positions = dict(self._positions)
                            self._last_error = f"Temperature safety stop at {hottest}°C"
                            self._event("error", self._last_error)
                        next_telemetry = now + 1.0
            except Exception as exc:
                with self._lock:
                    self._last_error = f"Communication safety stop: {exc}"
                    self._torque_enabled = False
                    self._returning_to_base = False
                    robot = self._robot
                    if robot is not None and robot.bus.is_connected:
                        try:
                            robot.bus.disable_torque(num_retry=2)
                        except Exception:
                            LOGGER.exception("Failed to disable torque after communication error")
                    self._worker = None
                    self._event("error", self._last_error)
                return

            remaining = period - (time.monotonic() - started)
            self._stop_event.wait(max(0.0, remaining))

    def set_target(self, joint: str, value: float) -> dict[str, Any]:
        with self._lock:
            if not self.is_connected:
                raise RuntimeError("Robot is not connected")
            if not self._torque_enabled:
                raise RuntimeError("Torque is released; enable hold before moving joints")
            if joint not in self.limits:
                raise ValueError(f"Unknown joint: {joint}")

            limit = self.limits[joint]
            bounded = min(limit.maximum, max(limit.minimum, float(value)))
            if self._returning_to_base:
                self._returning_to_base = False
                self._event("warning", "Base return cancelled by manual target")
            self._targets[joint] = bounded
            self._event("info", f"Target {joint}: {bounded:.1f}{limit.unit}")
            return self.snapshot()

    def set_torque(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            robot = self._robot
            if robot is None or not robot.bus.is_connected:
                raise RuntimeError("Robot is not connected")

            if enabled:
                raw_positions = robot.bus.sync_read(
                    "Present_Position", normalize=False, num_retry=2
                )
                self._validate_raw_positions(raw_positions)
                positions = robot.bus.sync_read("Present_Position", num_retry=2)
                robot.bus.sync_write("Goal_Position", positions)
                robot.bus.enable_torque(num_retry=2)
                self._positions = {joint: float(positions[joint]) for joint in JOINTS}
                self._targets = dict(self._positions)
                self._commanded_positions = dict(self._positions)
                self._following_errors = {joint: 0.0 for joint in JOINTS}
                self._torque_enabled = True
                self._returning_to_base = False
                self._last_error = None
                self._start_worker_locked()
                self._event("info", "Holding torque enabled at the measured pose")
            else:
                robot.bus.disable_torque(num_retry=2)
                self._torque_enabled = False
                self._returning_to_base = False
                self._targets = dict(self._positions)
                self._commanded_positions = dict(self._positions)
                self._following_errors = {joint: 0.0 for joint in JOINTS}
                self._event("warning", "Torque released")
            return self.snapshot()

    def capture_base_position(self) -> dict[str, Any]:
        """Capture and persist the arm's freshly measured pose without commanding motion."""
        with self._lock:
            robot = self._robot
            if robot is None or not robot.bus.is_connected:
                raise RuntimeError("Robot is not connected")

            raw_positions = robot.bus.sync_read(
                "Present_Position", normalize=False, num_retry=2
            )
            self._validate_raw_positions(raw_positions)
            measured = robot.bus.sync_read("Present_Position", num_retry=2)
            positions = validate_base_position(
                {joint: float(measured[joint]) for joint in JOINTS}, self.limits
            )
            save_base_position_file(self.base_position_file, self.robot_id, positions)
            self._base_position = positions
            self._event("info", f"Captured base position at {self.base_position_file}")
            return self.snapshot()

    def return_to_base(self) -> dict[str, Any]:
        """Start a rate-limited coordinated move to the captured base pose."""
        with self._lock:
            robot = self._robot
            if robot is None or not robot.bus.is_connected:
                raise RuntimeError("Robot is not connected")
            if self._base_position is None:
                raise RuntimeError("No base position captured")

            if not self._torque_enabled:
                raw_positions = robot.bus.sync_read(
                    "Present_Position", normalize=False, num_retry=2
                )
                self._validate_raw_positions(raw_positions)
                measured = robot.bus.sync_read("Present_Position", num_retry=2)
                positions = {joint: float(measured[joint]) for joint in JOINTS}
                robot.bus.sync_write("Goal_Position", positions)
                robot.bus.enable_torque(num_retry=2)
                self._positions = positions
                self._commanded_positions = dict(positions)
                self._following_errors = {joint: 0.0 for joint in JOINTS}
                self._torque_enabled = True

            self._targets = dict(self._base_position)
            self._returning_to_base = not positions_reached(
                self._positions, self._base_position
            )
            self._last_error = None
            self._start_worker_locked()
            if self._returning_to_base:
                self._event("info", "Returning all joints to the captured base position")
            else:
                self._event("info", "Already at base position; holding torque")
            return self.snapshot()

    def emergency_stop(self) -> dict[str, Any]:
        with self._lock:
            robot = self._robot
            if robot is not None and robot.bus.is_connected:
                robot.bus.disable_torque(num_retry=2)
            self._torque_enabled = False
            self._returning_to_base = False
            self._targets = dict(self._positions)
            self._commanded_positions = dict(self._positions)
            self._following_errors = {joint: 0.0 for joint in JOINTS}
            self._event("warning", "Emergency stop: torque released")
            return self.snapshot()

    def disconnect(self) -> dict[str, Any]:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

        with self._lock:
            robot = self._robot
            if robot is not None and robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=True)
            self._robot = None
            self._worker = None
            self._torque_enabled = False
            self._returning_to_base = False
            self._targets = dict(self._positions)
            self._commanded_positions = dict(self._positions)
            self._following_errors = {joint: 0.0 for joint in JOINTS}
            self._event("info", "Disconnected; torque released")
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self.is_connected,
                "torque_enabled": self._torque_enabled,
                "serial_port": self._serial_port,
                "robot_id": self.robot_id,
                "calibration_file": str(self.calibration_file),
                "base_position_file": str(self.base_position_file),
                "base_position": (
                    dict(self._base_position) if self._base_position is not None else None
                ),
                "returning_to_base": self._returning_to_base,
                "positions": dict(self._positions),
                "targets": dict(self._targets),
                "commanded_positions": dict(self._commanded_positions),
                "following_errors": dict(self._following_errors),
                "temperatures": dict(self._temperatures),
                "voltages": dict(self._voltages),
                "currents_raw": dict(self._currents_raw),
                "currents_ma": dict(self._currents_ma),
                "loads_raw": dict(self._loads_raw),
                "loads_percent": dict(self._loads_percent),
                "telemetry_log": list(self._telemetry_log),
                "limits": {joint: asdict(limit) for joint, limit in self.limits.items()},
                "last_error": self._last_error,
                "events": list(self._events),
            }
