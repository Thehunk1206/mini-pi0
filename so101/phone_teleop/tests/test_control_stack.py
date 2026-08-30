import unittest

import numpy as np

from so101.phone_teleop.control_stack import HoldActiveError, PhoneControlStack


JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
]


def positions(values):
    return dict(zip(JOINTS, values, strict=True))


def target(values):
    return {f"{name}.pos": value for name, value in zip(JOINTS, values, strict=True)}


class PhoneControlStackTest(unittest.TestCase):
    def test_filter_resets_on_hold_rising_edge(self):
        stack = PhoneControlStack(JOINTS)
        action = {"phone.enabled": False, "phone.pos": np.zeros(3)}
        stack.prepare_phone_action(action, 0.0)
        stack.step(positions(np.zeros(6)), target(np.zeros(6)), hold_active=False)
        prepared = stack.prepare_phone_action(
            {"phone.enabled": True, "phone.pos": np.array([0.2, 0.0, 0.0])}, 1 / 30
        )
        self.assertEqual(prepared["phone.pos"].tolist(), [0.2, 0.0, 0.0])
        self.assertTrue(stack._last_filter_sample.reset)

    def test_ruckig_never_exceeds_safe_limits(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions(np.zeros(6))
        request = target([80, -70, 60, 50, 100, 100])
        velocities, accelerations, jerks = [], [], []
        for _ in range(300):
            command = stack.step(measured, request, hold_active=True)
            measured = {name: command[f"{name}.pos"] for name in JOINTS}
            ruckig = stack.latest["ruckig"]
            velocities.append(list(ruckig["velocity"].values()))
            accelerations.append(list(ruckig["acceleration"].values()))
            jerks.append(list(ruckig["jerk"].values()))
        constraints = stack.latest["constraints"]
        vmax = np.asarray(constraints["arm_velocity"])
        amax = np.asarray(constraints["arm_acceleration"])
        jmax = np.asarray(constraints["arm_jerk"])
        self.assertTrue(np.all(np.max(np.abs(velocities), axis=0) <= vmax + 1e-8))
        self.assertTrue(np.all(np.max(np.abs(accelerations), axis=0) <= amax + 1e-8))
        self.assertTrue(np.all(np.max(np.abs(jerks), axis=0) <= jmax + 1e-8))

    def test_hold_release_starts_quick_stop_on_first_cycle(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions(np.zeros(6))
        request = target([80, 0, 0, 0, 0, 0])
        for _ in range(20):
            command = stack.step(measured, request, hold_active=True)
            measured = {name: command[f"{name}.pos"] for name in JOINTS}
        velocity_before = stack.latest["ruckig"]["velocity"]["shoulder_pan"]
        stack.step(measured, request, hold_active=False)
        self.assertEqual(stack.latest["ruckig"]["status"], "quick_stop")
        self.assertLess(stack.latest["ruckig"]["velocity"]["shoulder_pan"], velocity_before)

    def test_invalid_target_is_rejected_and_last_valid_is_retained(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions(np.zeros(6))
        good = target([10, 10, 10, 10, 10, 10])
        stack.step(measured, good, hold_active=True)
        bad = target([float("nan"), 10, 10, 10, 10, 10])
        stack.step(measured, bad, hold_active=True)
        self.assertFalse(stack.latest["target_valid"])
        self.assertEqual(stack.latest["ruckig"]["status"], "holding_last_valid_target")

    def test_tracking_fault_latches_until_hold_is_released(self):
        stack = PhoneControlStack(JOINTS)
        zero = positions(np.zeros(6))
        stack.step(zero, target(np.zeros(6)), hold_active=True)
        lagging = positions([-20, 0, 0, 0, 0, 0])
        for _ in range(3):
            stack.step(lagging, target(np.zeros(6)), hold_active=True)
        self.assertTrue(stack.latest["tracking"]["paused"])
        stack.step(lagging, target(np.zeros(6)), hold_active=True)
        self.assertTrue(stack.latest["tracking"]["paused"])
        stack.step(lagging, target(np.zeros(6)), hold_active=False)
        self.assertFalse(stack.latest["tracking"]["paused"])

    def test_profiles_can_only_change_with_hold_released(self):
        stack = PhoneControlStack(JOINTS)
        stack.step(positions(np.zeros(6)), target(np.zeros(6)), hold_active=True)
        with self.assertRaises(HoldActiveError):
            stack.set_profile("Balanced")
        stack.step(positions(np.zeros(6)), target(np.zeros(6)), hold_active=False)
        stack.set_profile("Balanced")
        self.assertEqual(stack.profile, "Balanced")

    def test_smooth_profile_reduces_jerk_more_than_velocity(self):
        stack = PhoneControlStack(JOINTS)
        stack.set_profile("Smooth")
        constraints = stack.commissioning_limits.scaled(stack.profile)

        np.testing.assert_allclose(
            constraints["arm_velocity"], [20, 20, 20, 30, 40]
        )
        np.testing.assert_allclose(
            constraints["arm_acceleration"], [60, 60, 60, 90, 120]
        )
        np.testing.assert_allclose(
            constraints["arm_jerk"], [180, 180, 180, 270, 360]
        )
        np.testing.assert_allclose(constraints["gripper_velocity"], [15])
        np.testing.assert_allclose(constraints["gripper_acceleration"], [45])
        np.testing.assert_allclose(constraints["gripper_jerk"], [180])

    def test_responsive_profile_is_twenty_five_percent_more_aggressive(self):
        stack = PhoneControlStack(JOINTS)
        stack.set_profile("Responsive")
        constraints = stack.commissioning_limits.scaled(stack.profile)

        np.testing.assert_allclose(
            constraints["arm_velocity"], [75, 75, 75, 112.5, 150]
        )
        np.testing.assert_allclose(
            constraints["arm_acceleration"], [225, 225, 225, 337.5, 450]
        )
        np.testing.assert_allclose(
            constraints["arm_jerk"], [1125, 1125, 1125, 1687.5, 2250]
        )
        np.testing.assert_allclose(constraints["gripper_velocity"], [62.5])

    def test_filter_settings_can_only_change_with_hold_released(self):
        stack = PhoneControlStack(JOINTS)
        stack.set_filter_settings({"beta": 1.5})
        self.assertEqual(stack.phone_filter_settings["beta"], 1.5)
        stack.step(positions(np.zeros(6)), target(np.zeros(6)), hold_active=True)
        with self.assertRaises(HoldActiveError):
            stack.set_filter_settings({"beta": 2.0})

    def test_continuous_target_has_bounded_nonzero_terminal_velocity(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions(np.zeros(6))
        stack.step(measured, target(np.zeros(6)), hold_active=True)
        for requested_position in (0.2, 0.4, 0.6, 0.8):
            command = stack.step(
                measured,
                target([requested_position, 0, 0, 0, 0, 0]),
                hold_active=True,
            )
            measured = {name: command[f"{name}.pos"] for name in JOINTS}

        target_velocity = stack.latest["ruckig"]["target_velocity"]["shoulder_pan"]
        self.assertGreater(target_velocity, 0.0)
        self.assertLessEqual(target_velocity, 30.0 * 0.8)

        stack.step(measured, target([0.8, 0, 0, 0, 0, 0]), hold_active=False)
        self.assertEqual(
            stack.latest["ruckig"]["target_velocity"]["shoulder_pan"], 0.0
        )

    def test_continuous_retarget_remains_inside_smooth_constraints(self):
        stack = PhoneControlStack(JOINTS)
        stack.set_profile("Smooth")
        measured = positions(np.zeros(6))
        velocities, accelerations, jerks = [], [], []
        for index in range(600):
            time_s = index / 30.0
            requested = [
                45.0 * np.sin(0.7 * time_s),
                40.0 * np.sin(0.5 * time_s + 0.3),
                35.0 * np.sin(0.8 * time_s + 0.7),
                30.0 * np.sin(0.6 * time_s),
                50.0 * np.sin(0.4 * time_s),
                50.0 + 30.0 * np.sin(0.5 * time_s),
            ]
            command = stack.step(measured, target(requested), hold_active=True)
            measured = {name: command[f"{name}.pos"] for name in JOINTS}
            ruckig = stack.latest["ruckig"]
            velocities.append(list(ruckig["velocity"].values()))
            accelerations.append(list(ruckig["acceleration"].values()))
            jerks.append(list(ruckig["jerk"].values()))

        constraints = stack.latest["constraints"]
        vmax = np.asarray(constraints["arm_velocity"])
        amax = np.asarray(constraints["arm_acceleration"])
        jmax = np.asarray(constraints["arm_jerk"])
        self.assertTrue(np.all(np.max(np.abs(velocities), axis=0) <= vmax + 1e-8))
        self.assertTrue(
            np.all(np.max(np.abs(accelerations), axis=0) <= amax + 1e-8)
        )
        self.assertTrue(np.all(np.max(np.abs(jerks), axis=0) <= jmax + 1e-8))

    def test_released_hold_does_not_move_measured_pose_already_outside_limits(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions([0, -108.7, 0, 0, 0, 0])
        command = stack.step(measured, target([0, -108.7, 0, 0, 0, 0]), hold_active=False)
        self.assertAlmostEqual(command["shoulder_lift.pos"], -108.7)

    def test_gripper_button_tracks_without_moving_released_arm(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions(np.zeros(6))
        request = target([40, 30, 20, 10, 5, 20])

        command = stack.step(
            measured,
            request,
            hold_active=False,
            gripper_active=True,
            gripper_direction=1,
        )

        np.testing.assert_allclose(
            [command[f"{name}.pos"] for name in JOINTS[:5]],
            np.zeros(5),
            atol=1e-12,
        )
        self.assertEqual(command["gripper.pos"], 20.0)
        self.assertNotIn("gripper", stack.latest["ruckig"]["position"])
        self.assertEqual(
            stack.latest["gripper_direct"]["status"], "direct_button"
        )

        released = stack.step(
            measured,
            target([0, 0, 0, 0, 0, 0]),
            hold_active=False,
            gripper_active=False,
            gripper_direction=0,
        )
        self.assertEqual(released["gripper.pos"], 20.0)
        self.assertEqual(
            stack.latest["gripper_direct"]["status"], "direct_hold"
        )

    def test_direct_gripper_command_is_not_paused_by_following_error(self):
        stack = PhoneControlStack(JOINTS)
        measured = positions([0, 0, 0, 0, 0, 30])
        close_request = target([0, 0, 0, 0, 0, 10])

        command = stack.step(
            measured,
            close_request,
            hold_active=False,
            gripper_active=True,
            gripper_direction=-1,
        )

        self.assertEqual(command["gripper.pos"], 10.0)
        self.assertFalse(stack.latest["tracking"]["fault"])
        self.assertFalse(stack.latest["tracking"]["paused"])
        self.assertEqual(stack.latest["tracking"]["errors"]["gripper"], 20.0)


if __name__ == "__main__":
    unittest.main()
