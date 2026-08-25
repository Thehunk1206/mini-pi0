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

from so101.phone_teleop.phone_control import (
    DisablePhoneOrientation,
    RemapPhoneTranslation,
)
from so101.phone_teleop.teleoperate import (
    ENABLE_PHONE_ORIENTATION,
    PHONE_TRANSLATION_AXIS_MAP,
    PHONE_TRANSLATION_AXIS_SIGNS,
    build_phone_processor,
)


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
        cls.kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=JOINTS,
        )

    def build(self, enable_orientation):
        return build_phone_processor(
            PhoneOS.ANDROID,
            self.kinematics,
            JOINTS,
            enable_orientation=enable_orientation,
        )

    def test_default_mode_is_xyz_only(self):
        self.assertFalse(ENABLE_PHONE_ORIENTATION)
        pipeline = build_phone_processor(
            PhoneOS.ANDROID, self.kinematics, JOINTS
        )

        self.assertEqual(
            [type(step) for step in pipeline.steps],
            [
                MapPhoneActionToRobotAction,
                RemapPhoneTranslation,
                DisablePhoneOrientation,
                EEReferenceAndDelta,
                EEBoundsAndSafety,
                GripperVelocityToJoint,
                InverseKinematicsEEToJoints,
            ],
        )

    def test_orientation_switch_omits_only_orientation_filter(self):
        pipeline = self.build(enable_orientation=True)

        self.assertEqual(
            [type(step) for step in pipeline.steps],
            [
                MapPhoneActionToRobotAction,
                RemapPhoneTranslation,
                EEReferenceAndDelta,
                EEBoundsAndSafety,
                GripperVelocityToJoint,
                InverseKinematicsEEToJoints,
            ],
        )

    def test_lerobot_example_motion_values_are_preserved(self):
        pipeline = self.build(enable_orientation=False)
        reference = next(
            step for step in pipeline.steps if isinstance(step, EEReferenceAndDelta)
        )
        safety = next(
            step for step in pipeline.steps if isinstance(step, EEBoundsAndSafety)
        )
        gripper = next(
            step for step in pipeline.steps if isinstance(step, GripperVelocityToJoint)
        )

        self.assertEqual(
            reference.end_effector_step_sizes,
            {"x": 0.5, "y": 0.5, "z": 0.5},
        )
        self.assertTrue(reference.use_latched_reference)
        self.assertEqual(safety.max_ee_step_m, 0.10)
        self.assertTrue(safety.raise_on_jump)
        self.assertEqual(gripper.speed_factor, 20.0)

    def test_global_translation_map_swaps_xy_and_preserves_z(self):
        self.assertEqual(
            PHONE_TRANSLATION_AXIS_MAP,
            {"x": "y", "y": "x", "z": "z"},
        )
        self.assertEqual(
            PHONE_TRANSLATION_AXIS_SIGNS,
            {"x": 1.0, "y": 1.0, "z": 1.0},
        )
        action = {
            "target_x": 1.0,
            "target_y": 2.0,
            "target_z": 3.0,
        }

        result = RemapPhoneTranslation(
            PHONE_TRANSLATION_AXIS_MAP, PHONE_TRANSLATION_AXIS_SIGNS
        ).action(action)

        self.assertEqual(
            [result[key] for key in ("target_x", "target_y", "target_z")],
            [2.0, 1.0, 3.0],
        )

    def test_translation_signs_can_reverse_individual_robot_axes(self):
        action = {
            "target_x": 1.0,
            "target_y": 2.0,
            "target_z": 3.0,
        }

        result = RemapPhoneTranslation(
            {"x": "y", "y": "x", "z": "z"},
            {"x": -1.0, "y": 1.0, "z": 1.0},
        ).action(action)

        self.assertEqual(result["target_x"], -2.0)
        self.assertEqual(result["target_y"], 1.0)
        self.assertEqual(result["target_z"], 3.0)

    def test_xyz_only_filter_preserves_translation_and_zeros_rotation(self):
        action = {
            "target_x": 0.1,
            "target_y": -0.2,
            "target_z": 0.3,
            "target_wx": 0.4,
            "target_wy": -0.5,
            "target_wz": 0.6,
        }

        result = DisablePhoneOrientation().action(action)

        self.assertEqual(
            [result[key] for key in ("target_x", "target_y", "target_z")],
            [0.1, -0.2, 0.3],
        )
        self.assertEqual(
            [result[key] for key in ("target_wx", "target_wy", "target_wz")],
            [0.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
