import unittest
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction

from so101.phone_teleop.teleoperate import build_phone_processor


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


class OfficialPhonePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=JOINTS,
        )
        cls.pipeline = build_phone_processor(PhoneOS.ANDROID, kinematics, JOINTS)

    def test_processor_order_matches_lerobot_example(self):
        self.assertEqual(
            [type(step) for step in self.pipeline.steps],
            [
                MapPhoneActionToRobotAction,
                EEReferenceAndDelta,
                EEBoundsAndSafety,
                GripperVelocityToJoint,
                InverseKinematicsEEToJoints,
            ],
        )

    def test_lerobot_example_motion_values_are_preserved(self):
        reference = self.pipeline.steps[1]
        safety = self.pipeline.steps[2]
        gripper = self.pipeline.steps[3]

        self.assertEqual(
            reference.end_effector_step_sizes,
            {"x": 0.5, "y": 0.5, "z": 0.5},
        )
        self.assertTrue(reference.use_latched_reference)
        self.assertEqual(safety.max_ee_step_m, 0.10)
        self.assertTrue(safety.raise_on_jump)
        self.assertEqual(gripper.speed_factor, 20.0)


if __name__ == "__main__":
    unittest.main()
