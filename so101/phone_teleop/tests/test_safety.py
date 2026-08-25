import unittest

from so101.phone_teleop.safety import (
    TranslationOnlyPhoneControl,
    configure_servo_acceleration,
)


class FakeBus:
    def __init__(self, motor_names):
        self.values = dict.fromkeys(motor_names, 254)
        self.writes = []

    def sync_write(self, register, values, *, normalize, num_retry):
        self.writes.append((register, dict(values), normalize, num_retry))
        self.values.update(values)

    def sync_read(self, register, *, normalize, num_retry):
        if register != "Acceleration":
            raise AssertionError(register)
        return dict(self.values)


class TranslationOnlyPhoneControlTest(unittest.TestCase):
    def test_zeroes_rotation_without_changing_translation_or_gripper(self):
        action = {
            "enabled": True,
            "target_x": 0.1,
            "target_y": -0.2,
            "target_z": 0.3,
            "target_wx": 1.0,
            "target_wy": 2.0,
            "target_wz": 3.0,
            "gripper_vel": -1.0,
        }

        result = TranslationOnlyPhoneControl().action(action)

        self.assertEqual(
            (result["target_wx"], result["target_wy"], result["target_wz"]),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            (result["target_x"], result["target_y"], result["target_z"]),
            (0.1, -0.2, 0.3),
        )
        self.assertEqual(result["gripper_vel"], -1.0)

    def test_rejects_incomplete_phone_orientation(self):
        with self.assertRaisesRegex(ValueError, "target_wz"):
            TranslationOnlyPhoneControl().action(
                {"target_wx": 0.0, "target_wy": 0.0}
            )


class ServoAccelerationTest(unittest.TestCase):
    def test_writes_and_verifies_every_motor(self):
        motors = ["shoulder_pan", "wrist_flex", "gripper"]
        bus = FakeBus(motors)

        readback = configure_servo_acceleration(bus, motors, 10)

        self.assertEqual(readback, dict.fromkeys(motors, 10))
        self.assertEqual(
            bus.writes,
            [("Acceleration", dict.fromkeys(motors, 10), False, 2)],
        )

    def test_rejects_out_of_range_acceleration(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 254"):
            configure_servo_acceleration(FakeBus(["gripper"]), ["gripper"], 0)


if __name__ == "__main__":
    unittest.main()
