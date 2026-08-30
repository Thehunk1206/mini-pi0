import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from so101.phone_teleop.visualization import (
    EndEffector3DVisualizer,
    calculate_cartesian_snapshot,
    ordered_joint_positions,
)
from so101.phone_teleop.urdf_model import URDFKinematicModel
from so101.teleop.model_assets import KINEMATIC_URDF_PATH
from so101.teleop.visualization import configure_rerun_batching


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

    def test_snapshots_work_with_rerun_disabled(self):
        visualizer = EndEffector3DVisualizer(
            FakeKinematics(),
            JOINTS,
            URDFKinematicModel.from_file(KINEMATIC_URDF_PATH),
            rerun_enabled=False,
        )
        visualizer.initialize()

        snapshot, render = visualizer.log(
            joint_dict([0, 0, 0, 0, 0, 0]),
            joint_dict([1, 2, 3, 4, 5, 6]),
        )

        self.assertEqual(snapshot.target_position_m, [1.0, 2.0, 3.0])
        self.assertEqual(render.name, "so101_new_calib")
        self.assertGreaterEqual(len(render.edges), 6)

    def test_rerun_decimation_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "decimation"):
            EndEffector3DVisualizer(
                FakeKinematics(),
                JOINTS,
                URDFKinematicModel.from_file(KINEMATIC_URDF_PATH),
                rerun_log_every_n_frames=0,
            )

    def test_rerun_geometry_is_decimated_without_decimating_snapshots(self):
        visualizer = EndEffector3DVisualizer(
            FakeKinematics(),
            JOINTS,
            URDFKinematicModel.from_file(KINEMATIC_URDF_PATH),
            rerun_enabled=False,
            show_skeleton=False,
            show_trail=False,
            rerun_log_every_n_frames=3,
        )
        visualizer.initialize()
        visualizer.rerun_enabled = True
        fake_rerun = MagicMock()
        observation = joint_dict([0, 0, 0, 0, 0, 0])
        action = joint_dict([1, 2, 3, 4, 5, 6])

        with patch.dict(sys.modules, {"rerun": fake_rerun}):
            snapshots = [visualizer.log(observation, action) for _ in range(4)]

        self.assertTrue(
            all(
                snapshot.target_position_m == [1.0, 2.0, 3.0]
                for snapshot, _ in snapshots
            )
        )
        # Calls 0 and 3 emit 11 Rerun entities each; calls 1 and 2 emit none.
        self.assertEqual(fake_rerun.log.call_count, 22)

    def test_rerun_batching_uses_large_chunks_and_respects_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_rerun_batching()
            self.assertEqual(os.environ["RERUN_FLUSH_NUM_BYTES"], "1048576")
            self.assertEqual(os.environ["RERUN_FLUSH_TICK_SECS"], "0.1")

        with patch.dict(
            os.environ,
            {
                "RERUN_FLUSH_NUM_BYTES": "2097152",
                "RERUN_FLUSH_TICK_SECS": "0.05",
            },
            clear=True,
        ):
            configure_rerun_batching()
            self.assertEqual(os.environ["RERUN_FLUSH_NUM_BYTES"], "2097152")
            self.assertEqual(os.environ["RERUN_FLUSH_TICK_SECS"], "0.05")


if __name__ == "__main__":
    unittest.main()
