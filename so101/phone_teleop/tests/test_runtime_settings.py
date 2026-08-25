import unittest
from types import SimpleNamespace

from so101.phone_teleop.control_ui import ControlSettings
from so101.phone_teleop.teleoperate import apply_control_settings


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class FakeBus:
    def __init__(self):
        self.acceleration = dict.fromkeys(JOINTS, 10)
        self.write_count = 0

    def sync_write(self, register, values, **_kwargs):
        self.assert_register(register)
        self.acceleration.update(values)
        self.write_count += 1

    def sync_read(self, register, **_kwargs):
        self.assert_register(register)
        return self.acceleration

    def assert_register(self, register):
        if register != "Acceleration":
            raise AssertionError(f"Unexpected register: {register}")


class RuntimeSettingsTest(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.robot = SimpleNamespace(
            bus=self.bus, config=SimpleNamespace(max_relative_target=3.0)
        )
        self.reference = SimpleNamespace(end_effector_step_sizes={})
        self.safety = SimpleNamespace(max_ee_step_m=0.01)
        self.gripper = SimpleNamespace(speed_factor=2.0)

    def test_profile_updates_hardware_and_live_processors(self):
        settings = ControlSettings(5.0, 0.2, 0.04, 10.0, 30)

        readback = apply_control_settings(
            settings,
            self.robot,
            JOINTS,
            self.reference,
            self.safety,
            self.gripper,
            previous=None,
        )

        self.assertEqual(readback, dict.fromkeys(JOINTS, 30))
        self.assertEqual(self.robot.config.max_relative_target, 5.0)
        self.assertEqual(
            self.reference.end_effector_step_sizes,
            {"x": 0.2, "y": 0.2, "z": 0.2},
        )
        self.assertEqual(self.safety.max_ee_step_m, 0.04)
        self.assertEqual(self.gripper.speed_factor, 10.0)

    def test_unchanged_acceleration_does_not_touch_the_bus(self):
        previous = ControlSettings(4.0, 0.15, 0.03, 8.0, 20)
        settings = ControlSettings(5.0, 0.2, 0.04, 10.0, 20)

        readback = apply_control_settings(
            settings,
            self.robot,
            JOINTS,
            self.reference,
            self.safety,
            self.gripper,
            previous=previous,
        )

        self.assertIsNone(readback)
        self.assertEqual(self.bus.write_count, 0)


if __name__ == "__main__":
    unittest.main()
