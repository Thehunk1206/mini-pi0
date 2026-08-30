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

import argparse
import time
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.utils.robot_utils import precise_sleep

from so101.teleop.cartesian import build_cartesian_ik_processor
from so101.teleop.calibration import (
    effective_joint_limits,
    load_motor_calibration,
    positions_outside_limits,
)
from so101.teleop.flight_recorder import ElectricalTelemetrySampler, FlightRecorder
from so101.teleop.control_ui import DesktopControlServer, RuntimeControlState
from so101.teleop.model_assets import (
    DEFAULT_MODEL_CACHE,
    KINEMATIC_URDF_PATH,
    ensure_model_cache,
    verify_kinematic_urdf,
)
from so101.teleop.runtime import (
    BASE_POSITION_PATH,
    DEFAULT_ROBOT_PORT,
    DESKTOP_UI_PORT,
    FPS,
    JOINT_SPEED_DEG_S,
    MOTOR_CALIBRATION_PATH,
    TerminalKeyReader,
    load_base_position,
    return_to_base,
)
from so101.teleop.urdf_model import URDFKinematicModel
from so101.teleop.visualization import (
    EndEffector3DVisualizer,
    configure_rerun_batching,
)

from .control_stack import PhoneControlStack
from .phone_control import DisablePhoneOrientation


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs" / "phone_teleop"
PHONE_TRANSLATION_GAIN = 0.5
MAX_EE_STEP_M = 0.10
GRIPPER_SPEED_FACTOR = 20.0
# Set to True to restore roll/pitch/yaw from the LeRobot phone example.
ENABLE_PHONE_ORIENTATION = False
RERUN_LOG_EVERY_N_FRAMES = 3


def print_phone_reference_pose() -> None:
    """Describe the reference pose required by LeRobot's Android mapping."""
    print("\nPhone reference pose:")
    print("  1. Lay the phone flat with its screen facing upward.")
    print("  2. Point the TOP edge straight forward, away from the robot base.")
    print("  3. Keep it still, then touch Hold to move to capture calibration.")
    print(
        "After capture: phone forward/back -> robot forward/back; "
        "left/right -> robot left/right.\n"
    )


def build_phone_processor(
    phone_os: PhoneOS,
    kinematics_solver: RobotKinematics,
    joint_names: list[str],
    *,
    enable_orientation: bool | None = None,
) -> RobotProcessorPipeline:
    """Build the LeRobot phone pipeline with an optional XYZ-only mode."""
    if enable_orientation is None:
        enable_orientation = ENABLE_PHONE_ORIENTATION

    steps = [MapPhoneActionToRobotAction(platform=phone_os)]
    if not enable_orientation:
        steps.append(DisablePhoneOrientation())
    return build_cartesian_ik_processor(
        kinematics_solver,
        joint_names,
        input_steps=steps,
        end_effector_step_sizes={
            "x": PHONE_TRANSLATION_GAIN,
            "y": PHONE_TRANSLATION_GAIN,
            "z": PHONE_TRANSLATION_GAIN,
        },
        max_ee_step_m=MAX_EE_STEP_M,
        gripper_speed_factor=GRIPPER_SPEED_FACTOR,
    )


def reset_phone_pipeline_on_hold_rising(
    pipeline: RobotProcessorPipeline,
    *,
    hold_active: bool,
    previous_hold_active: bool,
) -> bool:
    """Relatch Cartesian state when the phone clutch is re-engaged.

    While Hold is released, Ruckig may stop before reaching the last requested
    Cartesian target. Resetting the stateful LeRobot steps on the next rising
    edge prevents ``EEBoundsAndSafety`` from comparing the newly latched,
    measured pose with that stale target and reporting a false EE jump.
    """
    if not hold_active or previous_hold_active:
        return False
    pipeline.reset()
    return True


def phone_gripper_direction(phone_action: dict | None) -> int:
    """Return ``+1`` for Android A/open, ``-1`` for B/close, or zero."""
    raw_inputs = (phone_action or {}).get("phone.raw_inputs", {})
    if not isinstance(raw_inputs, dict):
        return 0
    open_pressed = bool(raw_inputs.get("reservedButtonA", False))
    close_pressed = bool(raw_inputs.get("reservedButtonB", False))
    return int(open_pressed) - int(close_pressed)


def phone_gripper_button_active(phone_action: dict | None) -> bool:
    """Return whether exactly one Android gripper button is pressed."""
    return phone_gripper_direction(phone_action) != 0


def wait_for_phone_release(teleop_device: Phone) -> None:
    while True:
        phone_action = teleop_device.get_action()
        if not phone_action or not bool(phone_action.get("phone.enabled", False)):
            return
        precise_sleep(0.02)


def main(*, enable_rerun: bool = False) -> None:
    # Initialize the robot and teleoperator. "handy_bot" selects the existing
    # ~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json.
    robot_config = SO100FollowerConfig(
        port=DEFAULT_ROBOT_PORT,
        id="handy_bot",
        use_degrees=True,
        # Keep retries bounded so a corrupted status packet cannot turn one
        # 33 ms control frame into a several-hundred-millisecond stall.
        num_read_retries=2,
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)
    joint_names = list(robot.bus.motors.keys())
    base_position = load_base_position(BASE_POSITION_PATH, joint_names)
    motor_calibration = load_motor_calibration(MOTOR_CALIBRATION_PATH, joint_names)

    urdf_path = KINEMATIC_URDF_PATH
    # This is the calibrated SO-101 URDF already checked into this branch.
    kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )
    urdf_model = URDFKinematicModel.from_file(urdf_path)
    model_metadata = ensure_model_cache()
    verify_kinematic_urdf(urdf_path, DEFAULT_MODEL_CACHE / model_metadata["urdf"])

    # Build pipeline to convert phone action to EE pose action to joint action.
    orientation_enabled = ENABLE_PHONE_ORIENTATION
    phone_to_robot_joints_processor = build_phone_processor(
        teleop_config.phone_os,
        kinematics_solver,
        joint_names,
        enable_orientation=orientation_enabled,
    )

    recorder = FlightRecorder(LOG_DIR, joint_names, fps=FPS)
    telemetry_sampler = ElectricalTelemetrySampler(joint_names, frequency_hz=5.0)
    visualizer = EndEffector3DVisualizer(
        kinematics_solver,
        joint_names,
        urdf_model,
        rerun_enabled=enable_rerun,
        rerun_log_every_n_frames=RERUN_LOG_EVERY_N_FRAMES,
    )
    runtime_state = RuntimeControlState()
    joint_limits = effective_joint_limits(
        urdf_model.revolute_limits_degrees(), motor_calibration
    )
    control_stack = PhoneControlStack(joint_names, cycle_s=1.0 / FPS, joint_limits=joint_limits)
    desktop_ui = DesktopControlServer(runtime_state, port=DESKTOP_UI_PORT)
    zero_links = urdf_model.link_positions(dict.fromkeys(joint_names, 0.0))
    control_mapping = {
        "orientation_enabled": orientation_enabled,
        "translation_gain": PHONE_TRANSLATION_GAIN,
        "max_ee_step_m": MAX_EE_STEP_M,
        "gripper_speed_factor": GRIPPER_SPEED_FACTOR,
        "translation_mapping": "LeRobot Android default",
    }
    runtime_state.publish(
        control_mapping=control_mapping,
        active_profile=control_stack.profile,
        filter_settings=control_stack.phone_filter_settings,
        control=control_stack.latest,
        calibration={
            "source": str(MOTOR_CALIBRATION_PATH),
            "motors": {
                name: info.to_dict() for name, info in motor_calibration.items()
            },
            "effective_joint_limits": joint_limits,
        },
        robot={
            "name": urdf_model.robot_name,
            "root_link": urdf_model.root_link,
            "edges": urdf_model.edges,
            "actual_links_m": zero_links,
            "target_links_m": zero_links,
        }
    )

    print(f"Flight-recorder session log: {recorder.session_path}")
    base_limit_violations = positions_outside_limits(base_position, joint_limits)
    if base_limit_violations:
        print(
            "WARNING: the saved base pose lies outside the effective URDF/calibration "
            f"envelope: {base_limit_violations}. Re-capture it before unloaded testing."
        )
    print(
        "Phone mapping: "
        f"{'XYZ plus orientation' if orientation_enabled else 'XYZ only'}, "
        f"translation gain={PHONE_TRANSLATION_GAIN}, "
        f"EE step={MAX_EE_STEP_M:.2f}m, "
        f"gripper speed={GRIPPER_SPEED_FACTOR:.1f}, "
        "no joint target clamp, default servo acceleration"
    )
    try:
        recorder.set_phase("desktop_ui.start")
        desktop_ui.start()
        print(f"Desktop control UI: {desktop_ui.url}")
        # Calibrate the phone before opening the servo bus. This avoids leaving
        # the serial connection idle during phone calibration.
        recorder.set_phase("phone.connect_and_calibrate")
        print_phone_reference_pose()
        teleop_device.connect()
        if enable_rerun:
            recorder.set_phase("rerun.start")
            from lerobot.utils.visualization_utils import init_rerun

            configure_rerun_batching()
            init_rerun(session_name="phone_so101_teleop")
            print("Rerun visualization enabled.")
        else:
            print("Rerun visualization disabled (start with --rerun to enable it).")
        visualizer.initialize()
        recorder.set_phase("robot.connect")
        robot.connect()
        recorder.set_phase("robot.initial_electrical")
        telemetry_sampler.maybe_read(robot.bus, force=True)
        recorder.record(electrical=telemetry_sampler.latest, event="robot_connected")
        recorder.record_electrical_summary(telemetry_sampler)
        runtime_state.publish(connected=True, phase="ready", electrical=telemetry_sampler.latest)

        if not robot.is_connected or not teleop_device.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting teleop loop. Move your phone to teleoperate the robot...")
        print(
            "Press B in this terminal to return to base and restart phone calibration."
        )
        rerun_frame_index = 0
        with TerminalKeyReader() as keyboard:
            while True:
                t0 = time.perf_counter()

                requested_settings = runtime_state.consume_settings()
                if requested_settings is not None:
                    control_stack.set_profile(requested_settings["profile"])
                    control_stack.set_filter_settings(
                        requested_settings["filter_settings"]
                    )
                    requested_orientation = requested_settings["orientation_enabled"]
                    if requested_orientation != orientation_enabled:
                        orientation_enabled = requested_orientation
                        phone_to_robot_joints_processor = build_phone_processor(
                            teleop_config.phone_os,
                            kinematics_solver,
                            joint_names,
                            enable_orientation=orientation_enabled,
                        )
                        control_mapping["orientation_enabled"] = orientation_enabled
                    runtime_state.publish(
                        active_profile=control_stack.profile,
                        filter_settings=control_stack.phone_filter_settings,
                        control_mapping=control_mapping,
                    )

                if keyboard.poll() == "b" or runtime_state.consume_base_return():
                    try:
                        return_to_base(
                            robot,
                            base_position,
                            joint_names,
                            recorder,
                            telemetry_sampler,
                            visualizer,
                            runtime_state,
                            rerun_enabled=enable_rerun,
                        )
                    except TimeoutError as exc:
                        # A timeout is recoverable: keep holding the last base
                        # command and require Move to be released before the
                        # phone pipeline can establish a fresh reference.
                        print(f"WARNING: {exc}")
                        recorder.record(event=f"base_timeout: {exc}")
                    recorder.set_phase("base.wait_for_phone_release")
                    runtime_state.publish(phase="base.wait_for_phone_release")
                    wait_for_phone_release(teleop_device)
                    phone_to_robot_joints_processor.reset()
                    print("Arm is holding base. Restarting phone calibration...")
                    recorder.set_phase("base.phone_recalibration")
                    runtime_state.publish(phase="base.phone_recalibration")
                    print_phone_reference_pose()
                    teleop_device.calibrate()
                    print("Phone pose captured. Release Hold to move once...")
                    recorder.set_phase("base.wait_after_phone_recalibration")
                    wait_for_phone_release(teleop_device)
                    phone_to_robot_joints_processor.reset()
                    measured_after_base = robot.get_observation()
                    control_stack.reset(
                        {
                            joint: float(measured_after_base[f"{joint}.pos"])
                            for joint in joint_names
                        },
                        reason="base_return_and_calibration",
                    )
                    recorder.record(event="phone_recalibrated_at_base")
                    print("Phone recalibrated at base. Hold to move when ready.")
                    continue

                # Get robot observation.
                recorder.set_phase("teleop.get_observation")
                robot_obs = robot.get_observation()

                # Get teleop action.
                recorder.set_phase("teleop.get_phone_action")
                phone_obs = teleop_device.get_action()
                hold_active = bool(
                    phone_obs and phone_obs.get("phone.enabled", False)
                )
                gripper_direction = phone_gripper_direction(phone_obs)
                gripper_active = gripper_direction != 0

                # The phone client zeros translation when Hold rises. Reset the
                # matching LeRobot Cartesian/IK state at that same boundary so
                # its jump checker does not retain the target from before the
                # clutch was released.
                reset_phone_pipeline_on_hold_rising(
                    phone_to_robot_joints_processor,
                    hold_active=hold_active,
                    previous_hold_active=control_stack.hold_active,
                )

                measured_positions = {
                    joint: float(robot_obs[f"{joint}.pos"]) for joint in joint_names
                }
                if (
                    not phone_obs
                    or phone_obs.get("phone.pos") is None
                    or phone_obs.get("phone.rot") is None
                ):
                    # A missing WebXR pose is a stream interruption. Reset both
                    # filter and OTG from the measured arm and hold position.
                    control_stack.reset(measured_positions, reason="stream_interruption")
                    raw_joint_action = {
                        f"{joint}.pos": value for joint, value in measured_positions.items()
                    }
                    joint_action = raw_joint_action.copy()
                    phone_obs = phone_obs or {}
                else:
                    filtered_phone_action = control_stack.prepare_phone_action(
                        phone_obs, time.monotonic()
                    )
                    # Filtered phone -> official Cartesian map -> raw IK target.
                    raw_joint_action = phone_to_robot_joints_processor(
                        (filtered_phone_action, robot_obs)
                    )
                    # The hardware receives only the jerk-limited Ruckig state.
                    joint_action = control_stack.step(
                        measured_positions,
                        raw_joint_action,
                        hold_active=hold_active,
                        gripper_active=gripper_active,
                        gripper_direction=gripper_direction,
                    )

                # Send action to robot.
                recorder.set_phase("teleop.send_action")
                sent_action = robot.send_action(joint_action)

                # Poll electrical feedback at 5 Hz and retain the latest values
                # in every 30 Hz control sample.
                recorder.set_phase("teleop.read_electrical")
                ruckig_velocity = control_stack.latest.get("ruckig", {}).get(
                    "velocity", {}
                )
                otg_is_moving = any(
                    abs(float(value)) > 1e-3 for value in ruckig_velocity.values()
                )
                # Electrical register reads can take hundreds of milliseconds
                # on a loaded STS3215 bus. Keep the latest cached sample during
                # active tracking and the complete quick stop so telemetry
                # cannot stall the 30 Hz command loop.
                electrical = telemetry_sampler.maybe_read(
                    robot.bus,
                    allow_bus_read=(
                        not hold_active
                        and not gripper_active
                        and not otg_is_moving
                    ),
                )
                # Visualize.
                recorder.set_phase("teleop.visualize")
                cartesian, urdf_render = visualizer.log(robot_obs, sent_action)
                if (
                    enable_rerun
                    and rerun_frame_index % RERUN_LOG_EVERY_N_FRAMES == 0
                ):
                    from lerobot.utils.visualization_utils import log_rerun_data

                    log_rerun_data(
                        observation={**phone_obs, **telemetry_sampler.rerun_scalars()},
                        action=sent_action,
                    )
                rerun_frame_index += 1
                loop_ms = (time.perf_counter() - t0) * 1000.0
                recorder.set_phase("teleop")
                recorder.record(
                    observation=robot_obs,
                    action=sent_action,
                    requested_action=raw_joint_action,
                    phone_action=phone_obs,
                    electrical=electrical,
                    cartesian=cartesian.to_dict(),
                    control=control_stack.latest,
                    loop_ms=loop_ms,
                )
                recorder.record_electrical_summary(telemetry_sampler)
                runtime_state.publish(
                    connected=True,
                    phase="teleop",
                    phone_enabled=hold_active,
                    loop_ms=loop_ms,
                    positions={
                        joint: float(robot_obs[f"{joint}.pos"])
                        for joint in joint_names
                    },
                    commands={
                        joint: float(sent_action[f"{joint}.pos"])
                        for joint in joint_names
                    },
                    active_profile=control_stack.profile,
                    filter_settings=control_stack.phone_filter_settings,
                    control=control_stack.latest,
                    electrical=electrical,
                    cartesian=cartesian.to_dict(),
                    robot=urdf_render.to_dict(),
                )

                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except Exception as exc:
        incident_path = recorder.capture_incident(exc)
        print(f"INCIDENT CAPTURED: {incident_path}")
        raise
    finally:
        runtime_state.publish(phase="shutdown", phone_enabled=False)
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
        runtime_state.publish(connected=False, phase="stopped", phone_enabled=False)
        desktop_ui.stop()
        recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="open the optional Rerun visualization application",
    )
    main(enable_rerun=parser.parse_args().rerun)
