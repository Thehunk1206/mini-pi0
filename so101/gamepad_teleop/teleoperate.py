"""Directly teleoperate the physical SO-101 with an Xbox-style controller.

Stick commands are deadzone-rescaled into planar wrist-pivot IK plus direct
pan/wrist targets. Validated joint targets are sent without Ruckig.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from so101.teleop.calibration import (
    calibrated_joint_limits,
    load_motor_calibration,
    positions_outside_limits,
)
from so101.teleop.flight_recorder import ElectricalTelemetrySampler, FlightRecorder
from so101.teleop.model_assets import (
    DEFAULT_MODEL_CACHE,
    KINEMATIC_URDF_PATH,
    ensure_model_cache,
    verify_kinematic_urdf,
)
from so101.teleop.runtime import (
    BASE_POSITION_PATH,
    DEFAULT_ROBOT_PORT,
    FPS,
    MOTOR_CALIBRATION_PATH,
    load_base_position,
    return_to_base,
)
from so101.teleop.urdf_model import URDFKinematicModel
from so101.teleop.visualization import (
    EndEffector3DVisualizer,
    configure_rerun_batching,
)

from .gamepad import (
    PAN_CONTROL_MODES,
    GamepadMotionSettings,
    GamepadTargetIntegrator,
    PygameGamepad,
    diagnose_controller,
    find_elbow_singularity_deg,
    test_controller_vibration,
)
from .direct_control import DirectGamepadControl
from .pipeline import (
    GRIPPER_SPEED_PERCENT_S,
    MAX_IK_TARGET_STEP_DEG,
    apply_calibrated_ik_limits,
    build_gamepad_processor,
)


LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "gamepad_teleop"
RERUN_LOG_EVERY_N_FRAMES = 3
# LeRobot configures STS3215 acceleration to 254 at connect. Keep the arm's
# direct response gentler, but leave the gripper at the maximum so it can track
# the faster trigger increments. These RAM values are overwritten on the next
# robot connection/power cycle and do not alter calibration or EEPROM limits.
GAMEPAD_ARM_SERVO_ACCELERATION = 90
GAMEPAD_GRIPPER_SERVO_ACCELERATION = 254


def configure_gamepad_servo_response(bus, joint_names: list[str]) -> dict[str, int]:
    """Apply and verify gamepad-only STS3215 RAM acceleration settings."""

    values = {
        joint: (
            GAMEPAD_GRIPPER_SERVO_ACCELERATION
            if joint == "gripper"
            else GAMEPAD_ARM_SERVO_ACCELERATION
        )
        for joint in joint_names
    }
    for joint, value in values.items():
        bus.write("Acceleration", joint, value, normalize=False, num_retry=2)
        observed = int(
            bus.read("Acceleration", joint, normalize=False, num_retry=2)
        )
        if observed != value:
            raise RuntimeError(
                f"{joint} acceleration readback {observed} != requested {value}"
            )
    return values


def print_controls(settings: GamepadMotionSettings) -> None:
    print("\nGamepad controls:")
    print("  Left stick horizontal Shoulder pan left / right")
    print("  Left stick vertical   Planar reach forward / backward")
    print("  D-pad up / down       Wrist-pivot height up / down")
    print("  Right stick vertical  Wrist flex down / up")
    print("  Right stick horizontal Wrist roll")
    print(
        "  LT / RT               Open / close gripper "
        f"({GRIPPER_SPEED_PERCENT_S:.0f}%/s, no Ruckig)"
    )
    print("  B                     Return to saved base pose")
    print("  Y / X                 Mark episode success / failure")
    print("  Back                  Mark episode for rerecord")
    print("  A / Start / RB        Unused")
    pan_behavior = (
        "stick position -> calibrated span; center -> calibrated midpoint"
        if settings.pan_control_mode == "absolute"
        else "stick deflection -> pan velocity; center -> hold"
    )
    print(f"  Pan mode              {settings.pan_control_mode}: {pan_behavior}")
    print(
        f"  Shaping               deadzone={settings.deadzone:.2f}, "
        f"expo={settings.expo:.2f}, ramp={1.0 / settings.axis_slew_rate_per_s:.2f}s, "
        f"reach/height={settings.planar_velocity_m_s:.2f}/"
        f"{settings.height_velocity_m_s:.2f} m/s, "
        f"pan={settings.shoulder_pan_velocity_deg_s:.0f} deg/s\n"
    )


def main(
    *,
    enable_rerun: bool = True,
    controller_index: int = 0,
    robot_port: str = DEFAULT_ROBOT_PORT,
    pan_control_mode: str = "velocity",
    pan_speed_deg_s: float = 45.0,
) -> None:
    settings = GamepadMotionSettings(
        pan_control_mode=pan_control_mode,
        shoulder_pan_velocity_deg_s=pan_speed_deg_s,
    )
    gamepad = PygameGamepad(controller_index)

    robot_config = SO100FollowerConfig(
        port=robot_port,
        id="handy_bot",
        use_degrees=True,
        num_read_retries=2,
    )
    robot = SO100Follower(robot_config)
    joint_names = list(robot.bus.motors.keys())
    base_position = load_base_position(BASE_POSITION_PATH, joint_names)
    motor_calibration = load_motor_calibration(MOTOR_CALIBRATION_PATH, joint_names)
    joint_limits = calibrated_joint_limits(motor_calibration)

    urdf_path = KINEMATIC_URDF_PATH
    kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="wrist_link",
        joint_names=joint_names,
    )
    apply_calibrated_ik_limits(kinematics_solver, joint_limits)
    elbow_singularity_deg = find_elbow_singularity_deg(
        kinematics_solver,
        joint_names,
        joint_limits,
    )
    urdf_model = URDFKinematicModel.from_file(urdf_path)
    model_metadata = ensure_model_cache()
    visual_urdf_path = DEFAULT_MODEL_CACHE / model_metadata["urdf"]
    verify_kinematic_urdf(urdf_path, visual_urdf_path)
    visual_urdf_model = URDFKinematicModel.from_file(visual_urdf_path)

    processor = build_gamepad_processor(kinematics_solver, joint_names)
    integrator = GamepadTargetIntegrator(
        kinematics_solver,
        joint_names,
        settings,
        elbow_singularity_deg=elbow_singularity_deg,
    )
    recorder = FlightRecorder(LOG_DIR, joint_names, fps=FPS)
    telemetry_sampler = ElectricalTelemetrySampler(joint_names, frequency_hz=5.0)
    visualizer = EndEffector3DVisualizer(
        kinematics_solver,
        joint_names,
        urdf_model,
        rerun_enabled=enable_rerun,
        visual_urdf_model=visual_urdf_model,
        show_skeleton=False,
        show_trail=False,
        rerun_log_every_n_frames=RERUN_LOG_EVERY_N_FRAMES,
    )
    print(f"Motor safety limits loaded from: {MOTOR_CALIBRATION_PATH}")
    print(
        "Calibrated command envelope: "
        + ", ".join(
            f"{name}=[{lower:.1f}, {upper:.1f}]"
            for name, (lower, upper) in joint_limits.items()
        )
    )
    direct_control = DirectGamepadControl(
        joint_names,
        joint_limits=joint_limits,
        max_target_step_deg=MAX_IK_TARGET_STEP_DEG,
        elbow_singularity_deg=elbow_singularity_deg,
    )
    print(
        "Model-derived straight-elbow boundary: "
        f"{elbow_singularity_deg:.2f} deg "
        f"(workspace margin {settings.extended_elbow_stop_deg:.1f} deg)"
    )
    print(f"Flight-recorder session log: {recorder.session_path}")
    base_limit_violations = positions_outside_limits(base_position, joint_limits)
    if base_limit_violations:
        print(
            "WARNING: saved base pose lies outside the calibrated motor "
            f"envelope: {base_limit_violations}. B-button base return will be "
            "blocked until it is re-captured."
        )

    try:
        # Prove pygame can poll the controller before opening the motor bus.
        recorder.set_phase("gamepad.connect")
        gamepad.connect()
        print(f"Detected controller: {gamepad.name}")
        print_controls(settings)

        if enable_rerun:
            recorder.set_phase("rerun.start")
            from lerobot.utils.visualization_utils import init_rerun

            configure_rerun_batching()
            init_rerun(session_name="gamepad_so101_teleop")
            print(
                "Rerun visualization enabled with "
                f"{len(visual_urdf_model.visuals)} official SO-101 mesh parts."
            )
        else:
            print("Rerun visualization disabled by --no-rerun.")
        visualizer.initialize()

        # This is the first operation that opens the servo bus.
        recorder.set_phase("robot.connect")
        robot.connect()
        recorder.set_phase("robot.gamepad_servo_response")
        servo_acceleration = configure_gamepad_servo_response(robot.bus, joint_names)
        print(
            "Gamepad servo acceleration: "
            f"arm={GAMEPAD_ARM_SERVO_ACCELERATION}, "
            f"gripper={GAMEPAD_GRIPPER_SERVO_ACCELERATION} (volatile RAM)"
        )
        recorder.record(
            event="gamepad_servo_response_configured",
            control={"servo_acceleration": servo_acceleration},
        )
        recorder.set_phase("robot.initial_electrical")
        telemetry_sampler.maybe_read(robot.bus, force=True)
        recorder.record(electrical=telemetry_sampler.latest, event="robot_connected")
        recorder.record_electrical_summary(telemetry_sampler)
        if not robot.is_connected or not gamepad.is_connected:
            raise RuntimeError("Robot or gamepad is not connected")

        initial_observation = robot.get_observation()
        initial_positions = {
            joint: float(initial_observation[f"{joint}.pos"])
            for joint in joint_names
        }
        processor.reset()
        integrator.reset()
        direct_control.reset(initial_positions, reason="hardware_start")
        recorder.record(
            observation=initial_observation,
            control=direct_control.latest,
            event="gamepad_control_initialized_from_measured_pose",
        )

        print("Starting direct gamepad teleoperation. Stick motion is immediately active.")
        rerun_frame_index = 0
        # Controller buttons own all runtime actions; there is no web or
        # terminal-key control surface in gamepad teleoperation.
        if gamepad.is_connected:
            while True:
                t0 = time.perf_counter()

                recorder.set_phase("teleop.get_gamepad_action")
                gamepad_sample = gamepad.read()

                if gamepad_sample.return_to_base:
                    if base_limit_violations:
                        print(
                            "BASE RETURN BLOCKED: saved pose is outside the "
                            "calibrated motor envelope."
                        )
                        gamepad.safety_feedback("joint_limit")
                        recorder.record(
                            event="base_return_blocked_by_calibration",
                            control={
                                "safety_event": "joint_limit",
                                "base_limit_violations": base_limit_violations,
                            },
                        )
                        continue
                    try:
                        return_to_base(
                            robot,
                            base_position,
                            joint_names,
                            recorder,
                            telemetry_sampler,
                            visualizer,
                            None,
                            rerun_enabled=enable_rerun,
                            clutch_label=None,
                        )
                    except TimeoutError as exc:
                        print(f"WARNING: {exc}")
                        recorder.record(event=f"base_timeout: {exc}")
                    processor.reset()
                    integrator.reset()
                    measured_after_base = robot.get_observation()
                    direct_control.reset(
                        {
                            joint: float(measured_after_base[f"{joint}.pos"])
                            for joint in joint_names
                        },
                        reason="base_return",
                    )
                    recorder.record(event="gamepad_reset_at_base")
                    print("Arm is at base. The next sample establishes a fresh reference.")
                    continue

                recorder.set_phase("teleop.get_observation")
                robot_obs = robot.get_observation()
                measured_positions = {
                    joint: float(robot_obs[f"{joint}.pos"]) for joint in joint_names
                }

                gamepad_action = integrator.update(
                    gamepad_sample,
                    measured_positions=measured_positions,
                    joint_limits_deg=joint_limits,
                )
                raw_joint_action = processor((gamepad_action, robot_obs))
                # Wrist-pivot IK supplies shoulder lift and elbow. Base pan,
                # wrist flex, and wrist roll remain direct gamepad commands.
                for joint, target in integrator.direct_joint_targets.items():
                    raw_joint_action[f"{joint}.pos"] = target
                joint_action = direct_control.step(
                    measured_positions,
                    raw_joint_action,
                    arm_input_active=bool(
                        integrator.latest.get("arm_input_active", False)
                    ),
                )
                if direct_control.latest.get("status") == "extension_limited":
                    integrator.rollback_latest_cartesian_step()
                integrated_gamepad_state = dict(integrator.latest)
                direct_control.latest["gamepad"] = integrated_gamepad_state
                if not direct_control.latest.get("target_valid", False):
                    # The rejected target never reaches the arm. Re-latch both
                    # Cartesian reference and integration state next cycle so
                    # open-loop offsets cannot accumulate into repeated jumps.
                    processor.reset()
                    integrator.reset()
                    direct_control.latest["reference_reset_after_rejection"] = True
                clamped_joints = integrated_gamepad_state.get(
                    "joint_limit_clamped", []
                )
                if clamped_joints and direct_control.latest.get("safety_event") is None:
                    direct_control.latest["safety_event"] = "joint_limit"
                    direct_control.latest["joint_limit_clamped"] = list(
                        clamped_joints
                    )
                if (
                    integrated_gamepad_state.get("workspace_clamped", False)
                    and direct_control.latest.get("safety_event") is None
                ):
                    direct_control.latest["safety_event"] = "workspace_limit"
                safety_event = direct_control.latest.get("safety_event")
                if safety_event is not None:
                    direct_control.latest["haptic_pulse_played"] = (
                        gamepad.safety_feedback(str(safety_event))
                    )
                    direct_control.latest["haptic_backends"] = dict(
                        gamepad.last_rumble_result
                    )
                else:
                    gamepad.clear_safety_feedback()

                recorder.set_phase("teleop.send_action")
                sent_action = robot.send_action(joint_action)

                recorder.set_phase("teleop.read_electrical")
                arm_input_active = bool(
                    integrated_gamepad_state.get("arm_input_active", False)
                )
                electrical = telemetry_sampler.maybe_read(
                    robot.bus,
                    allow_bus_read=(
                        not arm_input_active and gamepad_sample.gripper_direction == 0
                    ),
                )

                recorder.set_phase("teleop.visualize")
                cartesian, _urdf_render = visualizer.log(robot_obs, sent_action)
                if (
                    enable_rerun
                    and rerun_frame_index % RERUN_LOG_EVERY_N_FRAMES == 0
                ):
                    from lerobot.utils.visualization_utils import log_rerun_data

                    gamepad_scalars = {
                        f"gamepad.{name}": float(value)
                        for name, value in integrated_gamepad_state.get(
                            "shaped_axes", {}
                        ).items()
                    }
                    gamepad_scalars.update(
                        {
                            "gamepad.left_trigger": gamepad_sample.left_trigger,
                            "gamepad.right_trigger": gamepad_sample.right_trigger,
                            "gamepad.gripper_direction": float(
                                gamepad_sample.gripper_direction
                            ),
                        }
                    )
                    log_rerun_data(
                        observation={
                            **robot_obs,
                            **gamepad_scalars,
                            **telemetry_sampler.rerun_scalars(),
                        },
                        action=sent_action,
                    )
                rerun_frame_index += 1

                loop_ms = (time.perf_counter() - t0) * 1000.0
                recorder.set_phase("teleop")
                recorder.record(
                    observation=robot_obs,
                    action=sent_action,
                    requested_action=raw_joint_action,
                    gamepad_action={
                        **gamepad_sample.to_dict(),
                        "integrated": integrated_gamepad_state,
                    },
                    electrical=electrical,
                    cartesian=cartesian.to_dict(),
                    control=direct_control.latest,
                    loop_ms=loop_ms,
                    event=(
                        f"episode_{gamepad_sample.episode_event}"
                        if gamepad_sample.episode_event
                        else None
                    ),
                )
                recorder.record_electrical_summary(telemetry_sampler)
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
        gamepad.disconnect()
        recorder.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--rerun",
        dest="enable_rerun",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    rerun_group.add_argument(
        "--no-rerun",
        dest="enable_rerun",
        action="store_false",
        help="run controller-only without opening Rerun",
    )
    parser.set_defaults(enable_rerun=True)
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument(
        "--pan-mode",
        choices=PAN_CONTROL_MODES,
        default="velocity",
        help=(
            "velocity holds pan when centered; absolute maps the stick to the "
            "full calibrated span and returns to its midpoint when centered"
        ),
    )
    parser.add_argument(
        "--pan-speed-deg-s",
        type=float,
        default=45.0,
        help="maximum shoulder-pan command rate in either pan mode",
    )
    parser.add_argument(
        "--diagnose-controller",
        action="store_true",
        help="print gamepad inputs without opening the robot or servo bus",
    )
    parser.add_argument(
        "--test-vibration",
        action="store_true",
        help="play a controller vibration test without opening the robot bus",
    )
    parser.add_argument("--diagnose-seconds", type=float, default=8.0)
    args = parser.parse_args()
    if args.diagnose_controller:
        diagnose_controller(args.controller_index, args.diagnose_seconds)
        return
    if args.test_vibration:
        test_controller_vibration(args.controller_index)
        return
    main(
        enable_rerun=args.enable_rerun,
        controller_index=args.controller_index,
        robot_port=args.robot_port,
        pan_control_mode=args.pan_mode,
        pan_speed_deg_s=args.pan_speed_deg_s,
    )


if __name__ == "__main__":
    cli()
