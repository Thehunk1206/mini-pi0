import unittest
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.utils.rotation import Rotation

from so101.phone_teleop.phone_control import DisablePhoneOrientation
from so101.teleop.model_assets import KINEMATIC_URDF_PATH
from so101.phone_teleop.teleoperate import (
    ENABLE_PHONE_ORIENTATION,
    JOINT_SPEED_DEG_S,
    build_phone_processor,
    phone_gripper_button_active,
    phone_gripper_direction,
    reset_phone_pipeline_on_hold_rising,
)


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
URDF_PATH = KINEMATIC_URDF_PATH


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
        self.assertEqual(JOINT_SPEED_DEG_S, 40.0)

    def test_hold_rising_resets_stale_cartesian_safety_reference(self):
        pipeline = self.build(enable_orientation=False)
        reference = next(
            step for step in pipeline.steps if isinstance(step, EEReferenceAndDelta)
        )
        safety = next(
            step for step in pipeline.steps if isinstance(step, EEBoundsAndSafety)
        )
        reference.reference_ee_pose = np.eye(4)
        safety._last_pos = np.array([0.3, -0.2, 0.1])

        reset = reset_phone_pipeline_on_hold_rising(
            pipeline,
            hold_active=True,
            previous_hold_active=False,
        )

        self.assertTrue(reset)
        self.assertIsNone(reference.reference_ee_pose)
        self.assertIsNone(safety._last_pos)

    def test_continuous_hold_preserves_cartesian_safety_reference(self):
        pipeline = self.build(enable_orientation=False)
        safety = next(
            step for step in pipeline.steps if isinstance(step, EEBoundsAndSafety)
        )
        last_position = np.array([0.1, 0.2, 0.3])
        safety._last_pos = last_position.copy()

        reset = reset_phone_pipeline_on_hold_rising(
            pipeline,
            hold_active=True,
            previous_hold_active=True,
        )

        self.assertFalse(reset)
        np.testing.assert_array_equal(safety._last_pos, last_position)

    def test_android_translation_uses_lerobot_default_mapping(self):
        action = {
            "phone.enabled": True,
            "phone.pos": np.array([0.2, -0.1, 0.3]),
            "phone.rot": Rotation.from_rotvec(np.zeros(3)),
            "phone.raw_inputs": {},
        }

        result = MapPhoneActionToRobotAction(platform=PhoneOS.ANDROID).action(action)

        self.assertEqual(
            [result[key] for key in ("target_x", "target_y", "target_z")],
            [0.1, 0.2, 0.3],
        )

    def test_android_a_and_b_buttons_activate_gripper_independently(self):
        self.assertTrue(
            phone_gripper_button_active(
                {"phone.raw_inputs": {"reservedButtonA": True}}
            )
        )
        self.assertTrue(
            phone_gripper_button_active(
                {"phone.raw_inputs": {"reservedButtonB": True}}
            )
        )
        self.assertFalse(
            phone_gripper_button_active(
                {
                    "phone.raw_inputs": {
                        "reservedButtonA": True,
                        "reservedButtonB": True,
                    }
                }
            )
        )
        self.assertEqual(
            phone_gripper_direction(
                {"phone.raw_inputs": {"reservedButtonA": True}}
            ),
            1,
        )
        self.assertEqual(
            phone_gripper_direction(
                {"phone.raw_inputs": {"reservedButtonB": True}}
            ),
            -1,
        )

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
