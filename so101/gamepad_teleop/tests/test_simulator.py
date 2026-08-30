import inspect
import unittest

import numpy as np

from so101.gamepad_teleop.gamepad import GamepadSample
from so101.gamepad_teleop.simulator import GamepadKinematicSimulation, HOME


def sample(
    timestamp_s: float,
    *,
    gripper_direction: int = 0,
    return_to_base: bool = False,
    left_x: float = 0.0,
    left_y: float = 0.0,
    right_x: float = 0.0,
    right_y: float = 0.0,
    dpad_vertical: int = 0,
) -> GamepadSample:
    return GamepadSample(
        timestamp_s=timestamp_s,
        controller_name="test controller",
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        gripper_direction=gripper_direction,
        dpad_vertical=dpad_vertical,
        return_to_base=return_to_base,
    )


class GamepadKinematicSimulationTest(unittest.TestCase):
    def setUp(self):
        self.simulation = GamepadKinematicSimulation()

    def test_simulator_source_has_no_physical_robot_or_bus(self):
        import so101.gamepad_teleop.simulator as simulator

        source = inspect.getsource(simulator)
        self.assertNotIn("SO100Follower", source)
        self.assertNotIn("robot.connect", source)
        self.assertNotIn("send_action", source)
        self.assertNotIn("SO101ControlStack", source)

    def test_hardware_gamepad_has_rerun_but_no_web_control_surface(self):
        import so101.gamepad_teleop.teleoperate as teleoperate

        source = inspect.getsource(teleoperate)
        self.assertNotIn("DesktopControlServer", source)
        self.assertNotIn("RuntimeControlState", source)
        self.assertNotIn("control_ui", source)
        self.assertNotIn("DESKTOP_UI_PORT", source)
        self.assertNotIn("TerminalKeyReader", source)
        self.assertIn("calibrated_joint_limits(motor_calibration)", source)
        self.assertIn("visual_urdf_model=visual_urdf_model", source)
        self.assertIn("direct_control.reset(initial_positions", source)
        self.assertIn('target_frame_name="wrist_link"', source)
        self.assertNotIn("floor_height_m", source)
        self.assertNotIn("minimum_tip_height", source)
        self.assertTrue(
            inspect.signature(teleoperate.main)
            .parameters["enable_rerun"]
            .default
        )

    def test_hardware_gamepad_applies_verified_volatile_servo_acceleration(self):
        from so101.gamepad_teleop.teleoperate import (
            GAMEPAD_ARM_SERVO_ACCELERATION,
            GAMEPAD_GRIPPER_SERVO_ACCELERATION,
            configure_gamepad_servo_response,
        )

        class Bus:
            def __init__(self):
                self.values = {}

            def write(self, register, joint, value, **kwargs):
                self.values[(register, joint)] = value

            def read(self, register, joint, **kwargs):
                return self.values[(register, joint)]

        bus = Bus()
        values = configure_gamepad_servo_response(bus, list(self.simulation.joint_names))

        self.assertEqual(values["shoulder_pan"], GAMEPAD_ARM_SERVO_ACCELERATION)
        self.assertEqual(values["gripper"], GAMEPAD_GRIPPER_SERVO_ACCELERATION)
        self.assertTrue(
            all(register == "Acceleration" for register, _joint in bus.values)
        )

    def test_gamepad_runs_through_ik_and_direct_control(self):
        self.simulation.step(sample(1.0, left_y=1.0))
        frame = None
        for index in range(1, 31):
            frame = self.simulation.step(
                sample(1.0 + index / 30.0, left_y=1.0)
            )
        assert frame is not None
        self.assertTrue(frame.control["target_valid"])
        self.assertEqual(frame.control["mode"], "direct")
        self.assertNotIn("ruckig", frame.control)
        self.assertNotEqual(
            [frame.command[f"{joint}.pos"] for joint in self.simulation.joint_names[:5]],
            HOME[:5].tolist(),
        )

        # The input ramp decelerates rather than stopping discontinuously.
        neutral = None
        for index in range(1, 9):
            neutral = self.simulation.step(sample(2.0 + index / 30.0))
        assert neutral is not None
        self.assertEqual(neutral.control["status"], "direct_tracking")
        self.assertEqual(neutral.command, neutral.observation)

    def test_single_nudge_does_not_flip_ik_branches(self):
        raw_targets = []
        frames = []
        for index in range(120):
            frame = self.simulation.step(
                sample(
                    1.0 + index / 30.0,
                    left_y=1.0 if index < 3 else 0.0,
                )
            )
            frames.append(frame)
            raw_targets.append(
                [
                    frame.raw_joint_target[f"{joint}.pos"]
                    for joint in self.simulation.joint_names[:5]
                ]
            )

        target_steps = np.abs(np.diff(np.asarray(raw_targets), axis=0))
        self.assertLess(float(np.max(target_steps)), 5.0)
        self.assertTrue(all(frame.control["target_valid"] for frame in frames))
        self.assertTrue(all("ruckig" not in frame.control for frame in frames))
        self.assertAlmostEqual(
            frames[-1].gamepad["planar_offset_m"],
            0.002133333333333332,
        )

    def test_sustained_input_is_clamped_without_otg_failure(self):
        frames = [
            self.simulation.step(
                sample(1.0 + index / 30.0, left_y=1.0)
            )
            for index in range(180)
        ]
        self.assertGreaterEqual(
            frames[-1].command["elbow_flex.pos"]
            - self.simulation.elbow_singularity_deg,
            self.simulation.integrator.settings.extended_elbow_stop_deg - 0.1,
        )
        self.assertTrue(frames[-1].gamepad["extension_clamped"])
        self.assertTrue(frames[-1].gamepad["workspace_clamped"])
        self.assertEqual(frames[-1].control["safety_event"], "workspace_limit")
        self.assertTrue(all(frame.control["target_valid"] for frame in frames))
        self.assertTrue(all("ruckig" not in frame.control for frame in frames))

    def test_height_slides_tangentially_along_full_reach_boundary(self):
        reach_frames = [
            self.simulation.step(
                sample(1.0 + index / 30.0, left_y=1.0)
            )
            for index in range(240)
        ]
        boundary = reach_frames[-1]
        start_height = boundary.gamepad["height_offset_m"]
        start_planar = boundary.gamepad["planar_offset_m"]

        slide_frames = [
            self.simulation.step(
                sample(9.0 + index / 30.0, dpad_vertical=1)
            )
            for index in range(90)
        ]
        final = slide_frames[-1]

        self.assertGreater(final.gamepad["height_offset_m"], start_height + 0.10)
        self.assertLess(final.gamepad["planar_offset_m"], start_planar - 0.05)
        self.assertTrue(all(frame.control["target_valid"] for frame in slide_frames))
        self.assertTrue(
            all(frame.control["status"] == "direct_tracking" for frame in slide_frames)
        )
        self.assertTrue(any(frame.gamepad["workspace_projected"] for frame in slide_frames))

    def test_folded_physical_style_pose_can_reach_near_full_extension(self):
        self.simulation.measured = np.array(
            [0.0, -90.0, 90.0, 60.0, -170.0, 2.0], dtype=float
        )
        self.simulation.processor.reset()
        self.simulation.integrator.reset()
        self.simulation.control.reset(
            self.simulation.measured_positions,
            reason="folded_pose",
        )

        frames = [
            self.simulation.step(sample(1.0 + index / 30.0, left_y=1.0))
            for index in range(240)
        ]
        final = frames[-1]

        self.assertGreater(final.gamepad["x_offset_m"], 0.14)
        self.assertGreaterEqual(
            final.command["elbow_flex.pos"]
            - self.simulation.elbow_singularity_deg,
            self.simulation.integrator.settings.extended_elbow_stop_deg - 0.1,
        )
        self.assertTrue(final.gamepad["extension_clamped"])
        self.assertTrue(all(frame.control["target_valid"] for frame in frames))

    def test_reference_style_planar_and_direct_joint_mapping(self):
        cartesian_cases = {
            "forward reach": ({"left_y": 1.0}, "planar_offset_m", 1.0),
            "height up": ({"dpad_vertical": 1}, "height_offset_m", 1.0),
            "height down": ({"dpad_vertical": -1}, "height_offset_m", -1.0),
        }
        for label, (axes, field, expected_sign) in cartesian_cases.items():
            simulation = GamepadKinematicSimulation()
            simulation.step(sample(1.0, **axes))
            frame = None
            for index in range(1, 12):
                frame = simulation.step(
                    sample(1.0 + index / 30.0, **axes)
                )
            assert frame is not None
            self.assertGreater(expected_sign * frame.gamepad[field], 0.0, label)
            self.assertTrue(frame.control["target_valid"], label)

        direct_cases = {
            "pan": ({"left_x": 1.0}, "shoulder_pan", HOME[0]),
            "wrist flex down": ({"right_y": 1.0}, "wrist_flex", HOME[3]),
            "wrist roll": ({"right_x": 1.0}, "wrist_roll", HOME[4]),
        }
        for label, (axes, joint, initial) in direct_cases.items():
            simulation = GamepadKinematicSimulation()
            simulation.step(sample(1.0, **axes))
            frame = None
            for index in range(1, 12):
                frame = simulation.step(sample(1.0 + index / 30.0, **axes))
            assert frame is not None
            self.assertGreater(frame.command[f"{joint}.pos"], initial, label)
            self.assertTrue(frame.control["target_valid"], label)

    def test_absolute_pan_reaches_calibrated_full_span(self):
        simulation = GamepadKinematicSimulation(
            pan_control_mode="absolute",
            pan_speed_deg_s=100.0,
        )
        lower, upper = simulation.control.joint_limits["shoulder_pan"]

        frames = [
            simulation.step(sample(1.0 + index / 30.0, left_x=1.0))
            for index in range(120)
        ]

        self.assertAlmostEqual(frames[-1].command["shoulder_pan.pos"], upper)
        self.assertAlmostEqual(
            frames[-1].gamepad["desired_absolute_pan_deg"], upper
        )
        self.assertTrue(all(frame.control["target_valid"] for frame in frames))

        frames = [
            simulation.step(sample(5.0 + index / 30.0, left_x=-1.0))
            for index in range(120)
        ]

        self.assertAlmostEqual(frames[-1].command["shoulder_pan.pos"], lower)
        self.assertAlmostEqual(
            frames[-1].gamepad["desired_absolute_pan_deg"], lower
        )
        self.assertTrue(all(frame.control["target_valid"] for frame in frames))

    def test_trigger_gripper_and_b_reset(self):
        opened = self.simulation.step(sample(1.0, gripper_direction=1))
        self.assertAlmostEqual(
            opened.command["gripper.pos"] - HOME[5],
            200.0 / 30.0,
        )
        reset = self.simulation.step(sample(1.1, return_to_base=True))
        self.assertEqual(
            [reset.command[f"{joint}.pos"] for joint in self.simulation.joint_names],
            HOME.tolist(),
        )
        self.assertEqual(reset.event, "controller_start")


if __name__ == "__main__":
    unittest.main()
