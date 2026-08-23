import json
import tempfile
import unittest
from pathlib import Path

from so101_joint_ui.controller import (
    BODY_JOINTS,
    JOINTS,
    advance_commanded_positions,
    bounded_step,
    calibrated_joint_limits,
    current_raw_to_ma,
    load_base_position_file,
    load_calibration_file,
    load_raw_to_percent,
    positions_reached,
    RobotController,
    save_base_position_file,
    validate_base_position,
)


VALID_CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1,
        "range_min": 700,
        "range_max": 3400,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -2,
        "range_min": 800,
        "range_max": 3300,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -3,
        "range_min": 900,
        "range_max": 3100,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -4,
        "range_min": 850,
        "range_max": 3200,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 5,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 6,
        "range_min": 1900,
        "range_max": 3500,
    },
}


class FakeBus:
    def __init__(self, normalized_positions):
        self.is_connected = True
        self.normalized_positions = dict(normalized_positions)
        self.raw_positions = dict.fromkeys(JOINTS, 2000)
        self.goal_writes = []
        self.torque_enabled = False

    def sync_read(self, register, *, normalize=True, num_retry=2):
        if register != "Present_Position":
            raise AssertionError(f"Unexpected register read: {register}")
        return dict(self.normalized_positions if normalize else self.raw_positions)

    def sync_write(self, register, values):
        self.goal_writes.append((register, dict(values)))

    def enable_torque(self, *, num_retry=2):
        self.torque_enabled = True


class FakeRobot:
    def __init__(self, positions):
        self.bus = FakeBus(positions)


class ControllerHelpersTest(unittest.TestCase):
    def write_calibration(self, payload):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "robot.json"
        path.write_text(json.dumps(payload))
        self.addCleanup(directory.cleanup)
        return path

    def test_loads_expected_joint_ids(self):
        calibration = load_calibration_file(self.write_calibration(VALID_CALIBRATION))
        self.assertEqual(
            [calibration[joint].id for joint in VALID_CALIBRATION], list(range(1, 7))
        )

    def test_rejects_missing_joint(self):
        payload = dict(VALID_CALIBRATION)
        payload.pop("wrist_flex")
        with self.assertRaisesRegex(ValueError, "wrist_flex"):
            load_calibration_file(self.write_calibration(payload))

    def test_limits_match_lerobot_normalized_units(self):
        calibration = load_calibration_file(self.write_calibration(VALID_CALIBRATION))
        limits = calibrated_joint_limits(calibration)
        for joint in BODY_JOINTS:
            self.assertAlmostEqual(limits[joint].minimum, -limits[joint].maximum)
            self.assertEqual(limits[joint].unit, "°")
        self.assertEqual((limits["gripper"].minimum, limits["gripper"].maximum), (0.0, 100.0))

    def test_bounded_step_moves_in_both_directions(self):
        self.assertEqual(bounded_step(0.0, 10.0, 2.5), 2.5)
        self.assertEqual(bounded_step(10.0, 0.0, 2.5), 7.5)
        self.assertEqual(bounded_step(0.0, 1.0, 2.5), 1.0)

    def test_commanded_trajectory_accumulates_when_measurement_is_stationary(self):
        commanded = dict.fromkeys(JOINTS, 0.0)
        targets = dict(commanded)
        targets["wrist_flex"] = 10.0

        for _ in range(4):
            commanded = advance_commanded_positions(
                commanded,
                targets,
                max_joint_step=2.5,
                max_gripper_step=1.0,
            )

        self.assertEqual(commanded["wrist_flex"], 10.0)
        self.assertEqual(commanded["shoulder_pan"], 0.0)

    def test_gripper_uses_its_own_rate_limit(self):
        commanded = dict.fromkeys(JOINTS, 0.0)
        targets = dict(commanded)
        targets["gripper"] = 10.0

        advanced = advance_commanded_positions(
            commanded,
            targets,
            max_joint_step=2.5,
            max_gripper_step=1.0,
        )

        self.assertEqual(advanced["gripper"], 1.0)

    def test_current_register_converts_to_milliamps(self):
        self.assertEqual(current_raw_to_ma(100), 650.0)
        self.assertEqual(current_raw_to_ma(-10), 65.0)

    def test_load_register_preserves_direction_as_percent(self):
        self.assertEqual(load_raw_to_percent(250), 25.0)
        self.assertEqual(load_raw_to_percent(-125), -12.5)

    def test_base_position_round_trip(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "base.json"
        calibration = load_calibration_file(self.write_calibration(VALID_CALIBRATION))
        limits = calibrated_joint_limits(calibration)
        positions = dict.fromkeys(JOINTS, 0.0)
        positions["gripper"] = 25.0

        save_base_position_file(path, "test_bot", positions)

        self.assertEqual(load_base_position_file(path, limits), positions)
        payload = json.loads(path.read_text())
        self.assertEqual(payload["robot_id"], "test_bot")
        self.assertIn("captured_at", payload)

    def test_base_position_rejects_out_of_range_joint(self):
        calibration = load_calibration_file(self.write_calibration(VALID_CALIBRATION))
        limits = calibrated_joint_limits(calibration)
        positions = dict.fromkeys(JOINTS, 0.0)
        positions["gripper"] = 101.0

        with self.assertRaisesRegex(ValueError, "gripper"):
            validate_base_position(positions, limits)

    def test_base_position_completion_uses_separate_gripper_tolerance(self):
        targets = dict.fromkeys(JOINTS, 0.0)
        positions = dict(targets)
        positions["wrist_flex"] = 0.9
        positions["gripper"] = 1.9
        self.assertTrue(positions_reached(positions, targets))

        positions["gripper"] = 2.1
        self.assertFalse(positions_reached(positions, targets))

    def test_controller_captures_fresh_measured_base_position(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        calibration_path = self.write_calibration(VALID_CALIBRATION)
        base_path = Path(directory.name) / "base.json"
        measured = dict.fromkeys(JOINTS, 0.0)
        measured["shoulder_pan"] = 12.5
        measured["gripper"] = 30.0
        controller = RobotController(calibration_path, base_position_file=base_path)
        controller._robot = FakeRobot(measured)

        snapshot = controller.capture_base_position()

        self.assertEqual(snapshot["base_position"], measured)
        self.assertEqual(load_base_position_file(base_path, controller.limits), measured)
        self.assertFalse(controller._robot.bus.goal_writes)

    def test_return_to_base_sets_all_targets_and_keeps_torque_enabled(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        calibration_path = self.write_calibration(VALID_CALIBRATION)
        base_path = Path(directory.name) / "base.json"
        base = dict.fromkeys(JOINTS, 0.0)
        base["wrist_flex"] = 20.0
        save_base_position_file(base_path, "robot", base)
        controller = RobotController(calibration_path, base_position_file=base_path)
        controller._robot = FakeRobot(dict.fromkeys(JOINTS, 0.0))
        controller._positions = dict.fromkeys(JOINTS, 0.0)
        controller._torque_enabled = True
        controller._start_worker_locked = lambda: None

        snapshot = controller.return_to_base()

        self.assertEqual(snapshot["targets"], base)
        self.assertTrue(snapshot["returning_to_base"])
        self.assertTrue(snapshot["torque_enabled"])


if __name__ == "__main__":
    unittest.main()
