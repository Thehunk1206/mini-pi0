import unittest

import numpy as np

from so101.phone_teleop.visualization import (
    calculate_cartesian_snapshot,
    ordered_joint_positions,
)


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class FakeKinematics:
    def forward_kinematics(self, joints):
        transform = np.eye(4)
        transform[:3, 3] = joints[:3]
        return transform


def joint_dict(values):
    return {f"{joint}.pos": value for joint, value in zip(JOINTS, values, strict=True)}


class CartesianVisualizationTest(unittest.TestCase):
    def test_joint_positions_follow_explicit_motor_order(self):
        shuffled = dict(reversed(list(joint_dict(range(6)).items())))
        np.testing.assert_array_equal(
            ordered_joint_positions(shuffled, JOINTS), np.arange(6, dtype=float)
        )

    def test_snapshot_contains_actual_target_and_error(self):
        observation = joint_dict([0, 0, 0, 0, 0, 0])
        action = joint_dict([0.03, 0.04, 0, 0, 0, 0])

        snapshot, actual_transform, target_transform = calculate_cartesian_snapshot(
            FakeKinematics(), JOINTS, observation, action
        )

        self.assertEqual(snapshot.actual_position_m, [0.0, 0.0, 0.0])
        self.assertEqual(snapshot.target_position_m, [0.03, 0.04, 0.0])
        self.assertAlmostEqual(snapshot.error_m, 0.05)
        np.testing.assert_array_equal(actual_transform, np.eye(4))
        np.testing.assert_array_equal(target_transform[:3, 3], [0.03, 0.04, 0.0])

    def test_missing_joint_is_rejected(self):
        incomplete = joint_dict(range(6))
        incomplete.pop("gripper.pos")
        with self.assertRaisesRegex(ValueError, "gripper.pos"):
            ordered_joint_positions(incomplete, JOINTS)


if __name__ == "__main__":
    unittest.main()
