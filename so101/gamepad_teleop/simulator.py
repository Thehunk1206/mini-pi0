"""Hardware-free pygame -> planar IK -> direct command -> Rerun simulator.

This module deliberately does not import a robot, motor bus, serial port, or
the physical teleoperation runtime. It is a kinematic control-stack simulator,
not a gravity, collision, or contact-physics simulator.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.model.kinematics import RobotKinematics

from so101.teleop.calibration import calibrated_joint_limits, load_motor_calibration
from so101.teleop.model_assets import (
    KINEMATIC_URDF_PATH,
    ensure_model_cache,
    verify_kinematic_urdf,
)
from so101.teleop.runtime import MOTOR_CALIBRATION_PATH
from so101.teleop.urdf_model import URDFKinematicModel
from so101.teleop.visualization import (
    EndEffector3DVisualizer,
    configure_rerun_batching,
)

from .gamepad import (
    PAN_CONTROL_MODES,
    GamepadMotionSettings,
    GamepadSample,
    GamepadTargetIntegrator,
    PygameGamepad,
    find_elbow_singularity_deg,
)
from .direct_control import DirectGamepadControl
from .pipeline import (
    FPS,
    MAX_IK_TARGET_STEP_DEG,
    apply_calibrated_ik_limits,
    build_gamepad_processor,
)


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# Zero degrees is nearly fully extended and singular. Start the kinematic-only
# simulator bent and centered so small XYZ commands have usable workspace.
HOME = np.array([0.0, -40.0, 30.0, 0.0, 0.0, 35.0], dtype=float)
RERUN_TELEMETRY_HZ = 10
RERUN_STATUS_HZ = 1
RERUN_TELEMETRY_EVERY_N_FRAMES = max(1, round(FPS / RERUN_TELEMETRY_HZ))
RERUN_STATUS_EVERY_N_FRAMES = max(1, round(FPS / RERUN_STATUS_HZ))


@dataclass(frozen=True)
class SimulationFrame:
    """State produced by one controller/control-stack cycle."""

    observation: dict[str, float]
    raw_joint_target: dict[str, float]
    command: dict[str, float]
    gamepad: dict[str, Any]
    control: dict[str, Any]
    event: str | None = None


def _position_dict(values: np.ndarray) -> dict[str, float]:
    return {
        f"{joint}.pos": float(value)
        for joint, value in zip(JOINT_NAMES, values, strict=True)
    }


class GamepadKinematicSimulation:
    """Pure gamepad/IK/direct simulation with ideal one-cycle joint following."""

    def __init__(
        self,
        *,
        visual_urdf_path: Path | None = None,
        pan_control_mode: str = "velocity",
        pan_speed_deg_s: float = 45.0,
    ) -> None:
        self.joint_names = list(JOINT_NAMES)
        self.kinematics = RobotKinematics(
            urdf_path=str(KINEMATIC_URDF_PATH),
            target_frame_name="wrist_link",
            joint_names=self.joint_names,
        )
        self.urdf_model = URDFKinematicModel.from_file(KINEMATIC_URDF_PATH)
        self.visual_urdf_model = (
            URDFKinematicModel.from_file(visual_urdf_path)
            if visual_urdf_path is not None
            else None
        )
        joint_limits = self.urdf_model.revolute_limits_degrees()
        joint_limits["gripper"] = (0.0, 100.0)
        if MOTOR_CALIBRATION_PATH.exists():
            joint_limits = calibrated_joint_limits(
                load_motor_calibration(MOTOR_CALIBRATION_PATH, self.joint_names)
            )
        apply_calibrated_ik_limits(self.kinematics, joint_limits)
        self.elbow_singularity_deg = find_elbow_singularity_deg(
            self.kinematics,
            self.joint_names,
            joint_limits,
        )

        self.processor = build_gamepad_processor(self.kinematics, self.joint_names)
        self.integrator = GamepadTargetIntegrator(
            self.kinematics,
            self.joint_names,
            GamepadMotionSettings(
                pan_control_mode=pan_control_mode,
                shoulder_pan_velocity_deg_s=pan_speed_deg_s,
            ),
            elbow_singularity_deg=self.elbow_singularity_deg,
        )
        self.control = DirectGamepadControl(
            self.joint_names,
            joint_limits=joint_limits,
            max_target_step_deg=MAX_IK_TARGET_STEP_DEG,
            elbow_singularity_deg=self.elbow_singularity_deg,
        )
        self.measured = HOME.copy()
        self.control.reset(self.measured_positions, reason="simulator_start")

    @property
    def observation(self) -> dict[str, float]:
        return _position_dict(self.measured)

    @property
    def measured_positions(self) -> dict[str, float]:
        return {
            joint: float(value)
            for joint, value in zip(self.joint_names, self.measured, strict=True)
        }

    def reset(self, *, reason: str = "controller_start") -> SimulationFrame:
        self.measured = HOME.copy()
        self.processor.reset()
        self.integrator.reset()
        self.control.reset(self.measured_positions, reason=reason)
        home = self.observation
        return SimulationFrame(
            observation=home.copy(),
            raw_joint_target=home.copy(),
            command=home.copy(),
            gamepad={},
            control=self.control.latest,
            event=reason,
        )

    def step(self, sample: GamepadSample) -> SimulationFrame:
        if sample.return_to_base:
            return self.reset(reason="controller_start")

        observation = self.observation
        measured_positions = self.measured_positions

        gamepad_action = self.integrator.update(
            sample,
            measured_positions=measured_positions,
            joint_limits_deg=self.control.joint_limits,
        )
        raw_joint_target = self.processor((gamepad_action, observation))
        for joint, target in self.integrator.direct_joint_targets.items():
            raw_joint_target[f"{joint}.pos"] = target
        command = self.control.step(
            measured_positions,
            raw_joint_target,
            arm_input_active=bool(
                self.integrator.latest.get("arm_input_active", False)
            ),
        )
        if self.control.latest.get("status") == "extension_limited":
            self.integrator.rollback_latest_cartesian_step()
        integrated_gamepad_state = dict(self.integrator.latest)
        self.control.latest["gamepad"] = integrated_gamepad_state
        if not self.control.latest.get("target_valid", False):
            self.processor.reset()
            self.integrator.reset()
            self.control.latest["reference_reset_after_rejection"] = True
        clamped_joints = integrated_gamepad_state.get("joint_limit_clamped", [])
        if clamped_joints and self.control.latest.get("safety_event") is None:
            self.control.latest["safety_event"] = "joint_limit"
            self.control.latest["joint_limit_clamped"] = list(clamped_joints)
        if (
            integrated_gamepad_state.get("workspace_clamped", False)
            and self.control.latest.get("safety_event") is None
        ):
            self.control.latest["safety_event"] = "workspace_limit"

        # The previous command becomes the next cycle's ideal measured state.
        # Keeping the pre-command observation in this frame makes Rerun show the
        # commanded ghost and exactly one control period of tracking latency.
        self.measured = np.asarray(
            [command[f"{joint}.pos"] for joint in self.joint_names], dtype=float
        )
        return SimulationFrame(
            observation=observation,
            raw_joint_target={key: float(value) for key, value in raw_joint_target.items()},
            command={key: float(value) for key, value in command.items()},
            gamepad=integrated_gamepad_state,
            control=self.control.latest,
            event=(
                f"episode_{sample.episode_event}"
                if sample.episode_event is not None
                else None
            ),
        )


def _install_rerun_layout(controller_name: str, pan_control_mode: str) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents=["world/**"],
                name="SO-101 kinematic simulator",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="instructions", name="Gamepad controls"),
                    rrb.TextDocumentView(
                        origin="controller_status", name="Live controller values"
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["gamepad/**"],
                    name="Controller check",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["joints/**", "metrics/**"],
                    name="IK and direct joint state",
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(
        "instructions",
        rr.TextDocument(
            f"""# SO-101 gamepad simulator

**Controller:** {controller_name}<br>
**Control:** validated direct joint commands (no Ruckig)<br>
**Pan mode:** {pan_control_mode}<br>
**Mode:** hardware-free kinematic simulation (no gravity/contact physics)

- Left horizontal pans the shoulder.
- Left forward/back changes planar reach.
- D-pad up/down changes wrist-pivot height.
- Right forward/back flexes the wrist down/up.
- Right horizontal controls wrist roll.
- **LT/RT** opens/closes the simulated gripper.
- **B** resets the simulated robot to home.
- **Y/X/Back** records success/failure/rerecord events.
- **A/Start/RB are currently unused.** Press **Ctrl+C** to exit.
""",
            media_type="text/markdown",
        ),
        static=True,
    )


def _log_scalar(path: str, value: float | bool) -> None:
    import rerun as rr

    rr.log(path, rr.Scalars(float(value)))


def _log_frame(
    sample: GamepadSample,
    frame: SimulationFrame,
    *,
    frame_index: int,
) -> None:
    import rerun as rr

    # Preserve sparse episode/reset events even when telemetry is decimated.
    if frame.event:
        rr.log("events", rr.TextLog(frame.event, level=rr.TextLogLevel.INFO))

    if frame_index % RERUN_TELEMETRY_EVERY_N_FRAMES != 0:
        return

    pressed_buttons = [
        str(index) for index, pressed in enumerate(sample.raw_buttons) if pressed
    ]
    translation_offset = frame.gamepad.get("translation_offset_m", [0.0, 0.0, 0.0])
    target_rejection = frame.control.get("target_rejection")
    if frame_index % RERUN_STATUS_EVERY_N_FRAMES == 0:
        rr.log(
            "controller_status",
            rr.TextDocument(
                "# Live input\n\n"
                f"**Axes:** `{[round(value, 3) for value in sample.raw_axes]}`  \n"
                f"**Pressed buttons:** `{', '.join(pressed_buttons) or 'none'}`  \n"
                f"**LT / RT:** `{sample.left_trigger:.3f} / {sample.right_trigger:.3f}`  \n"
                f"**Gripper direction:** `{sample.gripper_direction:+d}`  \n"
                f"**XYZ offset:** `{[round(float(value), 4) for value in translation_offset]}`  \n"
                f"**Workspace clamp:** `{'ON' if frame.gamepad.get('workspace_clamped') else 'off'}`  \n"
                f"**Straight-elbow stop:** `{'ON' if frame.gamepad.get('extension_clamped') else 'off'}`  \n"
                f"**Joint clamp:** `{frame.gamepad.get('joint_limit_clamped') or 'off'}`  \n"
                f"**IK target:** `{'REJECTED: ' + str(target_rejection) if target_rejection else 'valid'}`  \n"
                f"**Direct control:** `{frame.control.get('status', 'unknown')}`",
                media_type="text/markdown",
            ),
        )

    raw_axes = {
        "left_x": sample.left_x,
        "left_y": sample.left_y,
        "right_x": sample.right_x,
        "right_y": sample.right_y,
        "left_trigger": sample.left_trigger,
        "right_trigger": sample.right_trigger,
        "dpad_vertical": sample.dpad_vertical,
    }
    for name, value in raw_axes.items():
        _log_scalar(f"gamepad/raw/{name}", value)
    for name, value in frame.gamepad.get("shaped_axes", {}).items():
        _log_scalar(f"gamepad/shaped/{name}", value)
    for axis, value in zip(
        "xyz", frame.gamepad.get("translation_offset_m", (0.0, 0.0, 0.0)), strict=True
    ):
        _log_scalar(f"gamepad/cartesian_offset/{axis}_m", value)
    _log_scalar(
        "gamepad/safety/workspace_clamped",
        bool(frame.gamepad.get("workspace_clamped", False)),
    )
    _log_scalar(
        "gamepad/safety/extension_clamped",
        bool(frame.gamepad.get("extension_clamped", False)),
    )
    _log_scalar("gamepad/safety/ik_target_valid", bool(frame.control.get("target_valid")))
    _log_scalar("gamepad/buttons/gripper_direction", sample.gripper_direction)

    raw_target = frame.control.get("raw_ik_joint_target", {})
    for joint in JOINT_NAMES:
        _log_scalar(f"joints/{joint}/measured", frame.observation[f"{joint}.pos"])
        _log_scalar(f"joints/{joint}/command", frame.command[f"{joint}.pos"])
        if joint in raw_target and raw_target[joint] is not None:
            _log_scalar(f"joints/{joint}/raw_ik_target", raw_target[joint])


def main(
    *,
    controller_index: int = 0,
    duration_s: float = 0.0,
    pan_control_mode: str = "velocity",
    pan_speed_deg_s: float = 45.0,
) -> None:
    """Run the real controller against the simulated robot and Rerun viewer."""
    gamepad = PygameGamepad(controller_index)
    model_metadata = ensure_model_cache()
    visual_urdf_path = (
        Path(model_metadata["cache_dir"]) / str(model_metadata["urdf"])
    )
    verify_kinematic_urdf(KINEMATIC_URDF_PATH, visual_urdf_path)
    simulation = GamepadKinematicSimulation(
        visual_urdf_path=visual_urdf_path,
        pan_control_mode=pan_control_mode,
        pan_speed_deg_s=pan_speed_deg_s,
    )
    visualizer = EndEffector3DVisualizer(
        simulation.kinematics,
        simulation.joint_names,
        simulation.urdf_model,
        rerun_enabled=True,
        visual_urdf_model=simulation.visual_urdf_model,
        show_skeleton=False,
        show_trail=False,
        rerun_log_every_n_frames=RERUN_TELEMETRY_EVERY_N_FRAMES,
    )

    from lerobot.utils.visualization_utils import init_rerun, shutdown_rerun

    rerun_started = False
    try:
        gamepad.connect()
        # Invoking ``.venv/bin/python`` does not activate the environment, so
        # its sibling ``rerun`` viewer may not be on PATH even though the
        # matching SDK/CLI package is installed there.
        executable_dir = str(Path(sys.executable).parent)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if executable_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join([executable_dir, *path_entries])
        configure_rerun_batching()
        init_rerun(session_name="so101_gamepad_kinematic_simulator")
        rerun_started = True
        visualizer.initialize()
        _install_rerun_layout(gamepad.name, pan_control_mode)
        print(f"Controller: {gamepad.name}")
        print(
            "Official SO-101 mesh: "
            f"{len(simulation.visual_urdf_model.visuals)} articulated visual parts"
        )
        print("Rerun gamepad simulator started. No robot or servo bus was opened.")
        print("Sticks move immediately; LT/RT controls the gripper; B resets home; Ctrl+C exits.")

        started = time.monotonic()
        frame_index = 0
        while duration_s <= 0.0 or time.monotonic() - started < duration_s:
            cycle_started = time.perf_counter()
            sample = gamepad.read()
            rr_time_s = time.monotonic() - started
            import rerun as rr

            rr.set_time("sim_time", duration=rr_time_s)
            frame = simulation.step(sample)
            safety_event = frame.control.get("safety_event")
            if safety_event is not None:
                frame.control["haptic_pulse_played"] = gamepad.safety_feedback(
                    str(safety_event)
                )
                frame.control["haptic_backends"] = dict(
                    gamepad.last_rumble_result
                )
            else:
                gamepad.clear_safety_feedback()
            visualizer.log(frame.observation, frame.command)
            _log_frame(sample, frame, frame_index=frame_index)
            frame_index += 1
            time.sleep(max(1.0 / FPS - (time.perf_counter() - cycle_started), 0.0))
    except KeyboardInterrupt:
        print("Gamepad simulator stopped.")
    finally:
        gamepad.disconnect()
        if rerun_started:
            shutdown_rerun()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument(
        "--pan-mode",
        choices=PAN_CONTROL_MODES,
        default="velocity",
        help="choose velocity pan or full-span calibrated absolute pan",
    )
    parser.add_argument(
        "--pan-speed-deg-s",
        type=float,
        default=45.0,
        help="maximum shoulder-pan command rate in either pan mode",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds; zero runs until Ctrl+C",
    )
    args = parser.parse_args()
    main(
        controller_index=args.controller_index,
        duration_s=args.duration,
        pan_control_mode=args.pan_mode,
        pan_speed_deg_s=args.pan_speed_deg_s,
    )


if __name__ == "__main__":
    cli()
