import json
import tempfile
import unittest
from pathlib import Path

from so101.phone_teleop.calibration import (
    effective_joint_limits,
    load_motor_calibration,
    positions_outside_limits,
)


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class CalibrationDataTest(unittest.TestCase):
    def test_raw_ranges_convert_to_lerobot_degrees_and_gripper_percent(self):
        payload = {
            name: {
                "id": index + 1, "drive_mode": 0, "homing_offset": 0,
                "range_min": 1000, "range_max": 3000,
            }
            for index, name in enumerate(JOINTS)
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handy_bot.json"
            path.write_text(json.dumps(payload))
            calibration = load_motor_calibration(path, JOINTS)

        self.assertAlmostEqual(calibration["shoulder_pan"].normalized_max, 2000 * 180 / 4095)
        self.assertAlmostEqual(calibration["shoulder_pan"].normalized_min, -2000 * 180 / 4095)
        self.assertEqual(
            (calibration["gripper"].normalized_min, calibration["gripper"].normalized_max),
            (0.0, 100.0),
        )

    def test_effective_limits_are_intersection_and_report_bad_base_pose(self):
        payload = {
            name: {
                "id": index + 1, "drive_mode": 0, "homing_offset": 0,
                "range_min": 0, "range_max": 4095,
            }
            for index, name in enumerate(JOINTS)
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handy_bot.json"
            path.write_text(json.dumps(payload))
            calibration = load_motor_calibration(path, JOINTS)
        urdf = {name: (-100.0, 100.0) for name in JOINTS}
        limits = effective_joint_limits(urdf, calibration)
        self.assertEqual(limits["shoulder_lift"], (-100.0, 100.0))
        self.assertEqual(limits["gripper"], (0.0, 100.0))
        outside = positions_outside_limits(
            {name: (-108.7 if name == "shoulder_lift" else 0.0) for name in JOINTS}, limits
        )
        self.assertEqual(set(outside), {"shoulder_lift"})


if __name__ == "__main__":
    unittest.main()
