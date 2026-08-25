#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot's phone-to-SO100 example, configured for this SO-101 and Android."""

import json
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from flight_recorder import ElectricalTelemetrySampler, FlightRecorder


FPS = 30
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs" / "phone_teleop"
BASE_POSITION_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json"
)
JOINT_SPEED_DEG_S = 30.0
GRIPPER_SPEED_PERCENT_S = 25.0
MAX_FOLLOWING_ERROR = 15.0
BASE_MOVE_TIMEOUT_S = 20.0
BASE_JOINT_TOLERANCE_DEG = 2.0
BASE_GRIPPER_TOLERANCE_PERCENT = 3.0


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
    recorder: FlightRecorder,
    telemetry_sampler: ElectricalTelemetrySampler,
) -> None:
    """Move all joints together to the captured base pose with rate/error limits."""
    print("Returning to base position. Release Hold to move...")
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
        robot.send_action(action)
        precise_sleep(period)
        recorder.set_phase("base.get_observation")
        observation = robot.get_observation()
        recorder.set_phase("base.read_electrical")
        electrical = telemetry_sampler.maybe_read(robot.bus)
        recorder.record(
            observation=observation,
            action=action,
            electrical=electrical,
            event="return_to_base",
        )
        recorder.record_electrical_summary(telemetry_sampler)

        following_errors = {
            joint: abs(command[joint] - float(observation[f"{joint}.pos"]))
            for joint in joint_names
        }
        worst_joint = max(following_errors, key=following_errors.get)
        if following_errors[worst_joint] > MAX_FOLLOWING_ERROR:
            raise RuntimeError(
                "Base return stopped because "
                f"{worst_joint} is not following its command "
                f"({following_errors[worst_joint]:.1f} error)."
            )

        log_rerun_data(observation=observation, action=action)
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
            print("Base position reached. Phone control is paused until Hold to move is released.")
            return

    residuals = {
        joint: abs(float(observation[f"{joint}.pos"]) - base[joint]) for joint in joint_names
    }
    residual_text = ", ".join(f"{joint}={error:.1f}" for joint, error in residuals.items())
    raise TimeoutError(
        f"Base return did not complete within {BASE_MOVE_TIMEOUT_S:.0f} seconds. "
        f"Residual errors: {residual_text}"
    )


def wait_for_phone_release(teleop_device: Phone) -> None:
    while True:
        phone_action = teleop_device.get_action()
        if not phone_action or not bool(phone_action.get("phone.enabled", False)):
            return
        precise_sleep(0.02)


def main():
    # Initialize the robot and teleoperator. "handy_bot" selects the existing
    # ~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json.
    robot_config = SO100FollowerConfig(
        port="/dev/cu.usbmodem5B610338651",
        id="handy_bot",
        use_degrees=True,
        num_read_retries=10,
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)
    joint_names = list(robot.bus.motors.keys())
    base_position = load_base_position(BASE_POSITION_PATH, joint_names)

    # This is the calibrated SO-101 URDF already checked into this branch.
    kinematics_solver = RobotKinematics(
        urdf_path=str(Path(__file__).resolve().parent / "SO101" / "so101_kinematics.urdf"),
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )

    # Build pipeline to convert phone action to EE pose action to joint action.
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=joint_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(speed_factor=20.0),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=joint_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    recorder = FlightRecorder(LOG_DIR, joint_names, fps=FPS)
    telemetry_sampler = ElectricalTelemetrySampler(joint_names, frequency_hz=5.0)

    print(f"Flight-recorder session log: {recorder.session_path}")
    try:
        # Calibrate the phone and spawn Rerun before opening the servo bus. This
        # avoids leaving the serial connection idle during phone calibration.
        recorder.set_phase("phone.connect_and_calibrate")
        teleop_device.connect()
        recorder.set_phase("rerun.start")
        init_rerun(session_name="phone_so100_teleop")
        recorder.set_phase("robot.connect")
        robot.connect()
        recorder.set_phase("robot.initial_electrical")
        telemetry_sampler.maybe_read(robot.bus, force=True)
        recorder.record(electrical=telemetry_sampler.latest, event="robot_connected")
        recorder.record_electrical_summary(telemetry_sampler)

        if not robot.is_connected or not teleop_device.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting teleop loop. Move your phone to teleoperate the robot...")
        print(
            "Press B in this terminal to return to base and restart phone calibration."
        )
        with TerminalKeyReader() as keyboard:
            while True:
                t0 = time.perf_counter()

                if keyboard.poll() == "b":
                    try:
                        return_to_base(
                            robot,
                            base_position,
                            joint_names,
                            recorder,
                            telemetry_sampler,
                        )
                    except TimeoutError as exc:
                        # A timeout is recoverable: keep holding the last base
                        # command and require Move to be released before the
                        # phone pipeline can establish a fresh reference.
                        print(f"WARNING: {exc}")
                        recorder.record(event=f"base_timeout: {exc}")
                    recorder.set_phase("base.wait_for_phone_release")
                    wait_for_phone_release(teleop_device)
                    phone_to_robot_joints_processor.reset()
                    print("Arm is holding base. Restarting phone calibration...")
                    recorder.set_phase("base.phone_recalibration")
                    teleop_device.calibrate()
                    print("Phone pose captured. Release Hold to move once...")
                    recorder.set_phase("base.wait_after_phone_recalibration")
                    wait_for_phone_release(teleop_device)
                    phone_to_robot_joints_processor.reset()
                    recorder.record(event="phone_recalibrated_at_base")
                    print("Phone recalibrated at base. Hold to move when ready.")
                    continue

                # Get robot observation.
                recorder.set_phase("teleop.get_observation")
                robot_obs = robot.get_observation()

                # Get teleop action.
                recorder.set_phase("teleop.get_phone_action")
                phone_obs = teleop_device.get_action()

                # Phone -> EE pose -> joints transition.
                joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

                # Send action to robot.
                recorder.set_phase("teleop.send_action")
                robot.send_action(joint_action)

                # Poll electrical feedback at 5 Hz and retain the latest values
                # in every 30 Hz control sample.
                recorder.set_phase("teleop.read_electrical")
                electrical = telemetry_sampler.maybe_read(robot.bus)
                loop_ms = (time.perf_counter() - t0) * 1000.0
                recorder.set_phase("teleop")
                recorder.record(
                    observation=robot_obs,
                    action=joint_action,
                    phone_action=phone_obs,
                    electrical=electrical,
                    loop_ms=loop_ms,
                )
                recorder.record_electrical_summary(telemetry_sampler)

                # Visualize.
                log_rerun_data(
                    observation={**phone_obs, **telemetry_sampler.rerun_scalars()},
                    action=joint_action,
                )

                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except Exception as exc:
        incident_path = recorder.capture_incident(exc)
        print(f"INCIDENT CAPTURED: {incident_path}")
        raise
    finally:
        if robot.is_connected:
            recorder.set_phase("shutdown.robot_disconnect")
            try:
                robot.disconnect()
            except Exception as exc:
                incident_path = recorder.capture_incident(exc, during="robot_disconnect")
                print(f"DISCONNECT FAILURE CAPTURED: {incident_path}")
                print(
                    "WARNING: Torque-disable could not reach the servos. "
                    "Switch off motor power before handling the arm."
                )
                if robot.bus.is_connected:
                    try:
                        robot.bus.disconnect(disable_torque=False)
                    except Exception as close_exc:
                        print(f"WARNING: Could not close servo port cleanly: {close_exc}")
        if teleop_device.is_connected:
            recorder.set_phase("shutdown.phone_disconnect")
            try:
                teleop_device.disconnect()
            except Exception as exc:
                incident_path = recorder.capture_incident(exc, during="phone_disconnect")
                print(f"PHONE DISCONNECT FAILURE CAPTURED: {incident_path}")
        recorder.close()


if __name__ == "__main__":
    main()
