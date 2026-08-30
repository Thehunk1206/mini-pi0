import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)

from so101.gamepad_teleop.gamepad import (
    GamepadMotionSettings,
    GamepadSample,
    GamepadTargetIntegrator,
    PygameGamepad,
    XboxLayout,
    find_elbow_singularity_deg,
    shape_axis,
)
from so101.gamepad_teleop.direct_control import DirectGamepadControl
from so101.gamepad_teleop.pipeline import (
    GRIPPER_SPEED_PERCENT_S,
    GRIPPER_STEP_PERCENT,
    MAX_IK_TARGET_STEP_DEG,
    MAX_EE_STEP_M,
    apply_calibrated_ik_limits,
    build_gamepad_processor,
)
from so101.teleop.model_assets import KINEMATIC_URDF_PATH


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class _PlanarKinematics:
    """Small deterministic FK stub retained for pure integrator tests."""

    def forward_kinematics(self, joints):
        angle = np.deg2rad(float(joints[0]))
        transform = np.eye(4)
        transform[:3, 3] = [
            0.0388353 + 0.25 * np.cos(angle),
            -0.25 * np.sin(angle),
            0.2,
        ]
        return transform


def positions(**updates):
    result = dict.fromkeys(JOINTS, 0.0)
    result.update(updates)
    return result


def limits():
    result = {name: (-100.0, 100.0) for name in JOINTS}
    result["gripper"] = (0.0, 100.0)
    return result


def integrator():
    return GamepadTargetIntegrator(_PlanarKinematics(), JOINTS)


def sample(timestamp_s: float, **axes) -> GamepadSample:
    return GamepadSample(
        timestamp_s=timestamp_s,
        controller_name="test controller",
        left_x=axes.get("left_x", 0.0),
        left_y=axes.get("left_y", 0.0),
        right_x=axes.get("right_x", 0.0),
        right_y=axes.get("right_y", 0.0),
        gripper_direction=axes.get("gripper_direction", 0),
        dpad_vertical=axes.get("dpad_vertical", 0),
    )


class AxisShapingTest(unittest.TestCase):
    def test_pan_control_mode_is_validated(self):
        self.assertEqual(GamepadMotionSettings().pan_control_mode, "velocity")
        self.assertEqual(
            GamepadMotionSettings(pan_control_mode="absolute").pan_control_mode,
            "absolute",
        )
        with self.assertRaisesRegex(ValueError, "pan_control_mode"):
            GamepadMotionSettings(pan_control_mode="unknown")

    def test_xbox_layout_uses_sdl_gamecontroller_button_semantics(self):
        layout = XboxLayout()
        self.assertEqual(layout.left_trigger_axis, 4)
        self.assertEqual(layout.right_trigger_axis, 5)
        self.assertEqual(layout.rerecord_button, 4)
        self.assertEqual(layout.base_button, 1)
        self.assertFalse(hasattr(layout, "open_button"))
        self.assertFalse(hasattr(layout, "close_button"))
        self.assertFalse(hasattr(layout, "deadman_button"))

    def test_triggers_control_gripper_and_b_is_edge_triggered_base(self):
        class EventQueue:
            @staticmethod
            def get():
                return []

        class Pygame:
            error = RuntimeError
            event = EventQueue()

        class Controller:
            @staticmethod
            def attached():
                return True

        gamepad = PygameGamepad()
        gamepad._pygame = Pygame()
        gamepad._controller = Controller()
        gamepad._previous_buttons = (False,) * 15
        axes = [0.0] * 6
        buttons = [False] * 15
        gamepad._axes = lambda: tuple(axes)
        gamepad._buttons = lambda: tuple(buttons)

        axes[4] = 0.8
        buttons[1] = True
        left_trigger = gamepad.read(timestamp_s=1.0)
        self.assertEqual(left_trigger.gripper_direction, 1)
        self.assertTrue(left_trigger.return_to_base)

        axes[4], axes[5] = 0.0, 0.9
        right_trigger = gamepad.read(timestamp_s=1.1)
        self.assertEqual(right_trigger.gripper_direction, -1)
        self.assertFalse(right_trigger.return_to_base)

        axes[4] = 0.8
        both_triggers = gamepad.read(timestamp_s=1.2)
        self.assertEqual(both_triggers.gripper_direction, 0)

    def test_deadzone_is_rescaled_and_full_range_is_preserved(self):
        self.assertEqual(shape_axis(0.12, deadzone=0.12, expo=0.65), 0.0)
        self.assertEqual(shape_axis(-0.05, deadzone=0.12, expo=0.65), 0.0)
        self.assertAlmostEqual(shape_axis(1.0, deadzone=0.12, expo=0.65), 1.0)
        self.assertAlmostEqual(shape_axis(-1.0, deadzone=0.12, expo=0.65), -1.0)

    def test_cubic_expo_reduces_small_stick_commands(self):
        linear = shape_axis(0.4, deadzone=0.12, expo=0.0)
        shaped = shape_axis(0.4, deadzone=0.12, expo=0.65)
        self.assertGreater(shaped, 0.0)
        self.assertLess(shaped, linear)

    def test_safety_rumble_pulses_once_until_safe_state_rearms_it(self):
        class Controller:
            def __init__(self):
                self.calls = []

            def rumble(self, low, high, duration):
                self.calls.append((low, high, duration))
                return True

        gamepad = PygameGamepad()
        gamepad._controller = Controller()
        self.assertTrue(gamepad.safety_feedback("joint_limit", timestamp_s=1.0))
        self.assertFalse(gamepad.safety_feedback("workspace_limit", timestamp_s=1.1))
        self.assertFalse(gamepad.safety_feedback("joint_limit", timestamp_s=2.0))
        gamepad.clear_safety_feedback(timestamp_s=2.1)
        gamepad.clear_safety_feedback(timestamp_s=2.4)
        self.assertFalse(gamepad.safety_feedback("workspace_limit", timestamp_s=2.5))
        gamepad.clear_safety_feedback(timestamp_s=2.7)
        self.assertTrue(gamepad.safety_feedback("workspace_limit", timestamp_s=2.8))
        self.assertFalse(gamepad.safety_feedback("floor", timestamp_s=3.5))
        self.assertEqual(gamepad._controller.calls[0], (0.8, 1.0, 350))
        self.assertEqual(gamepad._controller.calls[1], (1.0, 0.8, 400))
        self.assertEqual(len(gamepad._controller.calls), 2)

    def test_rumble_falls_back_to_joystick_haptics(self):
        class Joystick:
            @staticmethod
            def rumble(low, high, duration):
                return (low, high, duration) == (1.0, 0.5, 250)

        class Controller:
            @staticmethod
            def rumble(_low, _high, _duration):
                return False

            @staticmethod
            def as_joystick():
                return Joystick()

        gamepad = PygameGamepad()
        gamepad._controller = Controller()
        self.assertTrue(gamepad.rumble(1.0, 0.5, 250))

    def test_direct_hid_rumble_uses_prefixed_xusb_packet(self):
        class Joystick:
            @staticmethod
            def rumble(_low, _high, _duration):
                return False

        class Controller:
            @staticmethod
            def rumble(_low, _high, _duration):
                return False

            @staticmethod
            def as_joystick():
                return Joystick()

        gamepad = PygameGamepad()
        gamepad._controller = Controller()
        gamepad._direct_hid_path = b"controller-path"
        gamepad._write_direct_hid_packet = MagicMock(return_value=True)
        timer = MagicMock()

        with patch(
            "so101.gamepad_teleop.gamepad.threading.Timer", return_value=timer
        ) as timer_factory:
            self.assertTrue(gamepad.rumble(0.5, 0.25, 250))

        self.assertEqual(
            gamepad._write_direct_hid_packet.call_args.args[0],
            bytes((0x00, 0x00, 0x08, 0x00, 128, 64, 0x00, 0x00, 0x00)),
        )
        timer_factory.assert_called_once_with(0.25, gamepad._stop_direct_hid_rumble)
        self.assertTrue(timer.daemon)
        timer.start.assert_called_once_with()
        self.assertTrue(gamepad.last_rumble_result["direct_hid"])


class GamepadTargetIntegratorTest(unittest.TestCase):
    def test_model_derived_elbow_singularity_is_not_motor_zero(self):
        kinematics = RobotKinematics(
            urdf_path=str(KINEMATIC_URDF_PATH),
            target_frame_name="wrist_link",
            joint_names=JOINTS,
        )
        singularity = find_elbow_singularity_deg(
            kinematics,
            JOINTS,
            limits(),
        )

        self.assertLess(singularity, -70.0)
        self.assertGreater(singularity, -78.0)

    def test_absolute_pan_maps_full_stick_to_calibrated_endpoints(self):
        settings = GamepadMotionSettings(
            pan_control_mode="absolute",
            shoulder_pan_velocity_deg_s=100.0,
        )

        for stick, expected in ((1.0, 117.5), (-1.0, -117.5)):
            target = GamepadTargetIntegrator(_PlanarKinematics(), JOINTS, settings)
            calibrated_limits = limits()
            calibrated_limits["shoulder_pan"] = (-117.5, 117.5)
            target.update(
                sample(1.0, left_x=stick),
                measured_positions=positions(),
                joint_limits_deg=calibrated_limits,
            )
            for index in range(1, 31):
                target.update(
                    sample(1.0 + index / 10.0, left_x=stick),
                    measured_positions=positions(),
                    joint_limits_deg=calibrated_limits,
                )

            self.assertAlmostEqual(target.shoulder_pan_target_deg, expected)
            self.assertAlmostEqual(
                target.latest["desired_absolute_pan_deg"], expected
            )

    def test_absolute_pan_center_returns_to_midpoint_at_bounded_speed(self):
        settings = GamepadMotionSettings(
            pan_control_mode="absolute",
            shoulder_pan_velocity_deg_s=45.0,
        )
        target = GamepadTargetIntegrator(_PlanarKinematics(), JOINTS, settings)
        asymmetric_limits = limits()
        asymmetric_limits["shoulder_pan"] = (-90.0, 110.0)
        measured = positions(shoulder_pan=30.0)
        target.update(
            sample(1.0),
            measured_positions=measured,
            joint_limits_deg=asymmetric_limits,
        )
        target.update(
            sample(1.1),
            measured_positions=measured,
            joint_limits_deg=asymmetric_limits,
        )

        self.assertAlmostEqual(target.latest["desired_absolute_pan_deg"], 10.0)
        self.assertAlmostEqual(target.shoulder_pan_target_deg, 25.5)
        self.assertTrue(target.latest["arm_input_active"])

    def test_stick_velocity_ramps_up_and_down_without_a_step(self):
        target = integrator()
        target.update(
            sample(1.0, left_y=1.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        target.update(
            sample(1.02, left_y=1.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        ramping_up = target.latest["shaped_axes"]["planar_reach"]
        target.update(
            sample(1.04, left_y=1.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        full_before_release = target.latest["shaped_axes"]["planar_reach"]
        target.update(
            sample(1.06, left_y=0.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        ramping_down = target.latest["shaped_axes"]["planar_reach"]

        self.assertAlmostEqual(ramping_up, 0.08)
        self.assertAlmostEqual(full_before_release, 0.16)
        self.assertAlmostEqual(ramping_down, 0.08)
        self.assertEqual(target.latest["raw_shaped_axes"]["planar_reach"], 0.0)

    def test_velocity_integration_and_axis_signs(self):
        target = GamepadTargetIntegrator(
            _PlanarKinematics(), JOINTS, GamepadMotionSettings()
        )
        full_deflection = {
            "left_y": 1.0,
            "left_x": 1.0,
            "right_y": 1.0,
            "right_x": 1.0,
            "dpad_vertical": 1,
        }

        first = target.update(
            sample(1.0, **full_deflection),
            measured_positions=positions(elbow_flex=30.0),
            joint_limits_deg=limits(),
        )
        self.assertEqual([first["target_x"], first["target_y"], first["target_z"]], [0.0] * 3)

        second = target.update(
            sample(1.1, **full_deflection),
            measured_positions=positions(elbow_flex=30.0),
            joint_limits_deg=limits(),
        )
        self.assertAlmostEqual(target.latest["planar_offset_m"], 0.0048)
        self.assertAlmostEqual(target.latest["height_offset_m"], 0.0048)
        self.assertGreater(target.translation_offset_m[0], 0.0)
        self.assertLess(target.translation_offset_m[1], 0.0)
        self.assertAlmostEqual(target.translation_offset_m[2], 0.0048)
        self.assertAlmostEqual(target.shoulder_pan_target_deg, 1.8)
        self.assertAlmostEqual(target.wrist_flex_target_deg, 2.0)
        self.assertAlmostEqual(target.wrist_roll_target_deg, 2.8)
        for joint, expected in {
            "shoulder_pan": 1.8,
            "wrist_flex": 2.0,
            "wrist_roll": 2.8,
        }.items():
            self.assertAlmostEqual(target.direct_joint_targets[joint], expected)
        self.assertEqual(
            [second["target_x"], second["target_y"], second["target_z"]],
            target.translation_offset_m.tolist(),
        )

    def test_reset_relatches_measured_state(self):
        target = integrator()
        moving = {"left_y": 1.0}
        target.update(
            sample(1.0, **moving),
            measured_positions=positions(elbow_flex=30.0, wrist_roll=3.0),
            joint_limits_deg=limits(),
        )
        target.update(
            sample(1.1, **moving),
            measured_positions=positions(elbow_flex=30.0, wrist_roll=3.0),
            joint_limits_deg=limits(),
        )
        self.assertAlmostEqual(target.translation_offset_m[0], 0.0048)

        target.reset()
        relatched = target.update(
            sample(1.3, **moving),
            measured_positions=positions(elbow_flex=30.0, wrist_roll=12.0),
            joint_limits_deg=limits(),
        )
        self.assertEqual([relatched["target_x"], relatched["target_y"], relatched["target_z"]], [0.0] * 3)
        self.assertEqual(target.wrist_roll_target_deg, 12.0)

    def test_rejected_cartesian_step_rolls_back_without_reverting_pan(self):
        target = integrator()
        target.update(
            sample(1.0, left_y=1.0, left_x=1.0),
            measured_positions=positions(elbow_flex=5.0),
            joint_limits_deg=limits(),
        )
        target.update(
            sample(1.1, left_y=1.0, left_x=1.0),
            measured_positions=positions(elbow_flex=5.0),
            joint_limits_deg=limits(),
        )
        advanced_pan = target.shoulder_pan_target_deg
        self.assertGreater(target.latest["planar_offset_m"], 0.0)

        self.assertTrue(target.rollback_latest_cartesian_step())

        self.assertEqual(target.latest["planar_offset_m"], 0.0)
        self.assertEqual(target.latest["height_offset_m"], 0.0)
        self.assertTrue(target.latest["cartesian_step_rolled_back"])
        self.assertTrue(target.latest["extension_clamped"])
        self.assertEqual(target.shoulder_pan_target_deg, advanced_pan)
        self.assertNotEqual(target.translation_offset_m.tolist(), [0.0, 0.0, 0.0])
        self.assertFalse(target.rollback_latest_cartesian_step())

    def test_long_stream_gap_does_not_integrate_the_gap(self):
        target = integrator()
        target.update(
            sample(1.0, left_y=1.0),
            measured_positions=positions(elbow_flex=30.0),
            joint_limits_deg=limits(),
        )
        action = target.update(
            sample(2.0, left_y=1.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        self.assertEqual(action["target_x"], 0.0)
        self.assertTrue(target.latest["stream_gap"])

    def test_sustained_stick_input_stops_at_cartesian_workspace_boundary(self):
        target = integrator()
        target.update(
            sample(1.0, left_y=1.0),
            measured_positions=positions(),
            joint_limits_deg=limits(),
        )
        action = None
        for index in range(1, 181):
            action = target.update(
                sample(1.0 + index / 30.0, left_y=1.0),
                measured_positions=positions(elbow_flex=30.0),
                joint_limits_deg=limits(),
            )
        assert action is not None
        self.assertAlmostEqual(target.translation_offset_m[0], 0.40)
        self.assertTrue(target.latest["workspace_clamped"])

    def test_motor_zero_is_not_assumed_to_be_straight_without_a_model_boundary(self):
        target = integrator()
        near_motor_zero = positions(elbow_flex=1.0)
        target.update(
            sample(1.0, left_y=1.0),
            measured_positions=near_motor_zero,
            joint_limits_deg=limits(),
        )
        target.update(
            sample(1.1, left_y=1.0),
            measured_positions=near_motor_zero,
            joint_limits_deg=limits(),
        )

        self.assertGreater(target.translation_offset_m[0], 0.0)
        self.assertFalse(target.latest["extension_clamped"])
        self.assertFalse(target.latest["workspace_clamped"])


class GamepadProcessorTest(unittest.TestCase):
    def test_real_calibrated_folded_pose_accepts_neutral_and_small_nudge(self):
        """Regression for the measured pose that previously flipped IK branch."""

        measured = {
            "shoulder_pan": -0.7472527472527457,
            "shoulder_lift": -91.2967032967033,
            "elbow_flex": 97.01098901098902,
            "wrist_flex": 62.549450549450555,
            "wrist_roll": -175.86813186813185,
            "gripper": 1.7904509283819627,
        }
        calibrated_limits = {
            "shoulder_pan": (-117.5, 117.5),
            "shoulder_lift": (-111.3, 111.3),
            "elbow_flex": (-98.1, 98.1),
            "wrist_flex": (-104.6, 104.6),
            "wrist_roll": (-180.0, 180.0),
            "gripper": (0.0, 100.0),
        }
        kinematics = RobotKinematics(
            urdf_path=str(KINEMATIC_URDF_PATH),
            target_frame_name="wrist_link",
            joint_names=JOINTS,
        )
        apply_calibrated_ik_limits(kinematics, calibrated_limits)
        pipeline = build_gamepad_processor(kinematics, JOINTS)
        target = GamepadTargetIntegrator(kinematics, JOINTS)
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=calibrated_limits,
            max_target_step_deg=MAX_IK_TARGET_STEP_DEG,
        )
        control.reset(measured, reason="hardware_start")
        observation = {f"{name}.pos": value for name, value in measured.items()}

        def process(gamepad_sample):
            cartesian_action = target.update(
                gamepad_sample,
                measured_positions=measured,
                joint_limits_deg=calibrated_limits,
            )
            raw_action = pipeline((cartesian_action, observation))
            for name, value in target.direct_joint_targets.items():
                raw_action[f"{name}.pos"] = value
            command = control.step(
                measured,
                raw_action,
                arm_input_active=bool(target.latest["arm_input_active"]),
            )
            return raw_action, command

        _neutral_target, neutral_command = process(sample(1.0))
        self.assertTrue(control.latest["target_valid"])
        self.assertEqual(
            neutral_command,
            {f"{name}.pos": value for name, value in measured.items()},
        )

        nudged_target, nudged_command = process(sample(1.0 + 1.0 / 30.0, left_y=0.4))
        self.assertTrue(control.latest["target_valid"], control.latest["target_rejection"])
        self.assertLess(nudged_target["shoulder_lift.pos"], 0.0)
        self.assertLess(
            abs(nudged_command["shoulder_lift.pos"] - measured["shoulder_lift"]),
            MAX_IK_TARGET_STEP_DEG[1],
        )
        self.assertAlmostEqual(
            nudged_command["wrist_roll.pos"], measured["wrist_roll"]
        )

    def test_gamepad_direct_control_has_no_otg_state(self):
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        control.reset(positions(gripper=50.0), reason="test")
        action = {f"{name}.pos": value for name, value in positions(gripper=51.0).items()}
        command = control.step(positions(gripper=50.0), action)
        self.assertEqual(command["gripper.pos"], 51.0)
        self.assertEqual(control.latest["mode"], "direct")
        self.assertNotIn("ruckig", control.latest)

    def test_direct_control_refuses_measured_start_outside_calibration(self):
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        with self.assertRaisesRegex(ValueError, "outside calibrated range"):
            control.reset(positions(shoulder_lift=101.0), reason="test")

    def test_direct_control_rejects_joint_limit_atomically(self):
        measured = positions(gripper=50.0)
        action = {f"{name}.pos": value for name, value in measured.items()}
        limit_control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        limit_control.reset(measured, reason="test")
        action["shoulder_pan.pos"] = 101.0
        self.assertEqual(limit_control.step(measured, action), {
            f"{name}.pos": value for name, value in measured.items()
        })
        self.assertEqual(limit_control.latest["safety_event"], "joint_limit")

    def test_calibrated_endpoint_emits_joint_limit_feedback(self):
        measured = positions(shoulder_pan=99.75, elbow_flex=20.0, gripper=50.0)
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        control.reset(measured, reason="test")
        action = {f"{name}.pos": value for name, value in measured.items()}
        action["shoulder_pan.pos"] = 100.0

        control.step(measured, action, arm_input_active=True)

        self.assertTrue(control.latest["target_valid"])
        self.assertEqual(control.latest["safety_event"], "joint_limit")
        self.assertEqual(control.latest["joint_limit_joints"], ["shoulder_pan"])

    def test_static_near_limit_joint_does_not_vibrate_for_unrelated_motion(self):
        measured = positions(
            shoulder_pan=0.0,
            elbow_flex=99.0,
            wrist_roll=-99.0,
            gripper=50.0,
        )
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        control.reset(measured, reason="saved_base")
        action = {f"{name}.pos": value for name, value in measured.items()}
        action["shoulder_pan.pos"] = 1.0

        control.step(measured, action, arm_input_active=True)

        self.assertIsNone(control.latest["safety_event"])
        self.assertEqual(control.latest["joint_limit_joints"], [])

    def test_motion_away_from_near_limit_does_not_vibrate(self):
        measured = positions(shoulder_pan=99.0, gripper=50.0)
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        control.reset(measured, reason="near_upper_limit")
        action = {f"{name}.pos": value for name, value in measured.items()}
        action["shoulder_pan.pos"] = 98.0

        control.step(measured, action, arm_input_active=True)

        self.assertIsNone(control.latest["safety_event"])
        self.assertEqual(control.latest["joint_limit_joints"], [])

    def test_direct_control_does_not_cross_straight_elbow_branch(self):
        measured = positions(elbow_flex=0.2, gripper=50.0)
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
        )
        control.reset(measured, reason="test")
        action = {f"{name}.pos": value for name, value in measured.items()}
        action["shoulder_pan.pos"] = 1.0
        action["shoulder_lift.pos"] = 4.0
        action["elbow_flex.pos"] = -0.2
        action["wrist_flex.pos"] = 2.0
        action["wrist_roll.pos"] = 3.0
        action["gripper.pos"] = 51.0

        command = control.step(measured, action, arm_input_active=True)

        self.assertEqual(command["elbow_flex.pos"], 0.2)
        self.assertEqual(command["shoulder_lift.pos"], 0.0)
        self.assertEqual(command["shoulder_pan.pos"], 1.0)
        self.assertEqual(command["wrist_flex.pos"], 2.0)
        self.assertEqual(command["wrist_roll.pos"], 3.0)
        self.assertEqual(command["gripper.pos"], 51.0)
        self.assertEqual(control.latest["status"], "extension_limited")
        self.assertEqual(control.latest["safety_event"], "workspace_limit")
        self.assertEqual(
            control.latest["extension_held_joints"],
            ["shoulder_lift", "elbow_flex"],
        )

        # Even a subsequently accumulated raw IK request cannot create a jump:
        # the IK pair stays at the boundary while independent joints advance.
        action["shoulder_pan.pos"] = 2.0
        action["shoulder_lift.pos"] = 90.0
        action["elbow_flex.pos"] = -50.0
        action["wrist_flex.pos"] = 4.0
        repeated = control.step(measured, action, arm_input_active=True)

        self.assertTrue(control.latest["target_valid"])
        self.assertIsNone(control.latest["target_rejection"])
        self.assertEqual(repeated["shoulder_lift.pos"], 0.0)
        self.assertEqual(repeated["elbow_flex.pos"], 0.2)
        self.assertEqual(repeated["shoulder_pan.pos"], 2.0)
        self.assertEqual(repeated["wrist_flex.pos"], 4.0)

    def test_model_singularity_allows_motor_zero_crossing(self):
        measured = positions(elbow_flex=5.0, gripper=50.0)
        control = DirectGamepadControl(
            JOINTS,
            joint_limits=limits(),
            max_target_step_deg=(12.0, 15.0, 18.0, 18.0, 25.0),
            elbow_singularity_deg=-74.0,
        )
        control.reset(measured, reason="test")
        action = {f"{name}.pos": value for name, value in measured.items()}
        action["elbow_flex.pos"] = -1.0

        command = control.step(measured, action, arm_input_active=True)

        self.assertEqual(command["elbow_flex.pos"], -1.0)
        self.assertEqual(control.latest["status"], "direct_tracking")

    def test_pipeline_uses_position_only_ik_and_direct_gripper_step(self):
        joints = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        pipeline = build_gamepad_processor(_PlanarKinematics(), joints)
        self.assertEqual(
            [type(step) for step in pipeline.steps],
            [
                EEReferenceAndDelta,
                EEBoundsAndSafety,
                GripperVelocityToJoint,
                InverseKinematicsEEToJoints,
            ],
        )
        safety = pipeline.steps[1]
        gripper = pipeline.steps[2]
        ik = pipeline.steps[3]
        self.assertEqual(safety.max_ee_step_m, MAX_EE_STEP_M)
        self.assertFalse(safety.raise_on_jump)
        self.assertAlmostEqual(gripper.speed_factor, GRIPPER_STEP_PERCENT)
        self.assertEqual(GRIPPER_SPEED_PERCENT_S, 200.0)
        self.assertTrue(ik.initial_guess_current_joints)
        self.assertEqual(ik.orientation_weight, 0.0)

    def test_calibrated_limits_replace_narrower_urdf_ik_limits(self):
        kinematics = RobotKinematics(
            urdf_path=str(KINEMATIC_URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=JOINTS,
        )
        calibrated_limits = limits()
        calibrated_limits["shoulder_lift"] = (-111.25, 111.25)
        calibrated_limits["elbow_flex"] = (-98.07, 98.07)

        apply_calibrated_ik_limits(kinematics, calibrated_limits)

        shoulder_limits = np.rad2deg(
            kinematics.robot.get_joint_limits("shoulder_lift")
        )
        elbow_limits = np.rad2deg(
            kinematics.robot.get_joint_limits("elbow_flex")
        )
        self.assertTrue(np.allclose(shoulder_limits, [-111.25, 111.25]))
        self.assertTrue(np.allclose(elbow_limits, [-98.07, 98.07]))


if __name__ == "__main__":
    unittest.main()
