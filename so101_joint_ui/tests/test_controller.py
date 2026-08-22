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
    load_calibration_file,
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


if __name__ == "__main__":
    unittest.main()
