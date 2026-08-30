"""Shared physical-runtime helpers for SO-101 teleoperation front ends."""

from __future__ import annotations

import json
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

from lerobot.robots.so_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep


FPS = 30
DEFAULT_ROBOT_PORT = "/dev/cu.usbmodem5B610338651"
BASE_POSITION_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json"
)
MOTOR_CALIBRATION_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json"
)
JOINT_SPEED_DEG_S = 40.0
GRIPPER_SPEED_PERCENT_S = 25.0
MAX_FOLLOWING_ERROR = 15.0
BASE_MOVE_TIMEOUT_S = 20.0
BASE_JOINT_TOLERANCE_DEG = 2.0
BASE_GRIPPER_TOLERANCE_PERCENT = 3.0
DESKTOP_UI_PORT = 8001


class TerminalKeyReader:
    """Read single keys from the launching terminal without blocking the control loop."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._original_attributes: list | None = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._original_attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        else:
            print("Keyboard shortcut disabled because stdin is not a terminal.")
        return self

    def poll(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([self._fd], [], [], 0)
        return sys.stdin.read(1).lower() if readable else None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._fd is not None and self._original_attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_attributes)


def load_base_position(path: Path, joint_names: list[str]) -> dict[str, float]:
    payload = json.loads(path.read_text())
    positions = payload.get("positions") if isinstance(payload, dict) else None
    if not isinstance(positions, dict) or set(positions) != set(joint_names):
        raise ValueError(f"Invalid six-joint base-position file: {path}")

    base = {joint: float(positions[joint]) for joint in joint_names}
    if not all(math.isfinite(value) for value in base.values()):
        raise ValueError(f"Base-position file contains a non-finite value: {path}")
    return base


def bounded_step(current: float, target: float, maximum_step: float) -> float:
    delta = target - current
    if abs(delta) <= maximum_step:
        return target
    return current + math.copysign(maximum_step, delta)


def return_to_base(
    robot: SO100Follower,
    base: dict[str, float],
    joint_names: list[str],
    recorder: Any,
    telemetry_sampler: Any,
    visualizer: Any,
    runtime_state: Any | None,
    *,
    rerun_enabled: bool = False,
    clutch_label: str | None = "Hold",
) -> None:
    """Move all joints together to the captured base pose with rate/error limits."""
    if clutch_label is None:
        print("Returning to base position...")
    else:
        print(f"Returning to base position. Keep {clutch_label} released...")
    recorder.set_phase("base.get_initial_observation")
    observation = robot.get_observation()
    command = {joint: float(observation[f"{joint}.pos"]) for joint in joint_names}
    period = 1.0 / FPS
    started = time.monotonic()

    while time.monotonic() - started < BASE_MOVE_TIMEOUT_S:
        command = {
            joint: bounded_step(
                command[joint],
                base[joint],
                (GRIPPER_SPEED_PERCENT_S if joint == "gripper" else JOINT_SPEED_DEG_S) / FPS,
            )
            for joint in joint_names
        }
        action = {f"{joint}.pos": value for joint, value in command.items()}
        recorder.set_phase("base.send_action")
        sent_action = robot.send_action(action)
        precise_sleep(period)
        recorder.set_phase("base.get_observation")
        observation = robot.get_observation()
        recorder.set_phase("base.read_electrical")
        electrical = telemetry_sampler.maybe_read(robot.bus)
        following_errors = {
            joint: abs(
                float(sent_action[f"{joint}.pos"])
                - float(observation[f"{joint}.pos"])
            )
            for joint in joint_names
        }
        worst_joint = max(following_errors, key=following_errors.get)
        if following_errors[worst_joint] > MAX_FOLLOWING_ERROR:
            raise RuntimeError(
                "Base return stopped because "
                f"{worst_joint} is not following its command "
                f"({following_errors[worst_joint]:.1f} error)."
            )

        cartesian, urdf_render = visualizer.log(observation, sent_action)
        recorder.record(
            observation=observation,
            action=sent_action,
            requested_action=action,
            electrical=electrical,
            cartesian=cartesian.to_dict(),
            event="return_to_base",
        )
        recorder.record_electrical_summary(telemetry_sampler)
        if rerun_enabled:
            from lerobot.utils.visualization_utils import log_rerun_data

            log_rerun_data(observation=observation, action=sent_action)
        if runtime_state is not None:
            runtime_state.publish(
                connected=True,
                phase="return_to_base",
                phone_enabled=False,
                positions={
                    joint: float(observation[f"{joint}.pos"])
                    for joint in joint_names
                },
                commands={
                    joint: float(sent_action[f"{joint}.pos"])
                    for joint in joint_names
                },
                electrical=electrical,
                cartesian=cartesian.to_dict(),
                robot=urdf_render.to_dict(),
            )
        reached = all(
            abs(float(observation[f"{joint}.pos"]) - base[joint])
            <= (
                BASE_GRIPPER_TOLERANCE_PERCENT
                if joint == "gripper"
                else BASE_JOINT_TOLERANCE_DEG
            )
            for joint in joint_names
        )
        if reached:
            if clutch_label is None:
                print("Base position reached.")
            else:
                print(
                    "Base position reached. Control is paused until "
                    f"{clutch_label} is released."
                )
            return

    residuals = {
        joint: abs(float(observation[f"{joint}.pos"]) - base[joint]) for joint in joint_names
    }
    residual_text = ", ".join(f"{joint}={error:.1f}" for joint, error in residuals.items())
    raise TimeoutError(
        f"Base return did not complete within {BASE_MOVE_TIMEOUT_S:.0f} seconds. "
        f"Residual errors: {residual_text}"
    )
