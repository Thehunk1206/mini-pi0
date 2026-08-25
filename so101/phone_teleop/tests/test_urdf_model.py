import unittest
from pathlib import Path

import numpy as np
from lerobot.model.kinematics import RobotKinematics

from so101.phone_teleop.urdf_model import URDFKinematicModel


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
URDF_PATH = (
    Path(__file__).resolve().parents[1] / "kinematics" / "so101_kinematics.urdf"
)


class URDFKinematicModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = URDFKinematicModel.from_file(URDF_PATH)
        cls.lerobot_kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=JOINTS,
        )

    def test_checked_in_model_is_a_single_complete_tree(self):
        self.assertEqual(self.model.robot_name, "so101_new_calib")
        self.assertEqual(self.model.root_link, "base_link")
        self.assertEqual(len(self.model.joints), 7)
        self.assertEqual(
            {edge["joint"] for edge in self.model.edges},
            {*JOINTS, "gripper_frame_joint"},
        )

    def test_renderer_forward_kinematics_matches_lerobot(self):
        for joint_values in (
            [0, 0, 0, 0, 0, 0],
            [10, -20, 30, -15, 40, 20],
            [-30, 45, -10, 25, -60, 70],
        ):
            with self.subTest(joint_values=joint_values):
                expected = np.asarray(
                    self.lerobot_kinematics.forward_kinematics(
                        np.asarray(joint_values, dtype=float)
                    )
                )[:3, 3]
                positions = self.model.link_positions(
                    dict(zip(JOINTS, joint_values, strict=True))
                )
                np.testing.assert_allclose(
                    positions["gripper_frame_link"], expected, atol=1e-10
                )

    def test_gripper_jaw_is_rendered_as_an_articulated_branch(self):
        closed = self.model.link_transforms(dict.fromkeys(JOINTS, 0.0))
        open_pose = self.model.link_transforms(
            {**dict.fromkeys(JOINTS, 0.0), "gripper": 90.0}
        )

        np.testing.assert_allclose(
            closed["gripper_link"], open_pose["gripper_link"], atol=1e-12
        )
        self.assertFalse(
            np.allclose(
                closed["moving_jaw_so101_v1_link"][:3, :3],
                open_pose["moving_jaw_so101_v1_link"][:3, :3],
            )
        )


if __name__ == "__main__":
    unittest.main()
