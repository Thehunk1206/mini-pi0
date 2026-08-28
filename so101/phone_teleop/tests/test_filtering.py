import unittest

import numpy as np

from so101.phone_teleop.filtering import (
    DEFAULT_PHONE_FILTER_SETTINGS,
    ConstantVelocityKalmanXYZ,
    OneEuroXYZFilter,
    validated_phone_filter_settings,
)


class OneEuroXYZFilterTest(unittest.TestCase):
    def test_smoother_live_defaults(self):
        filter_ = OneEuroXYZFilter()

        self.assertEqual(filter_.min_cutoff_hz, 1.0)
        self.assertEqual(filter_.beta, 2.0)
        self.assertEqual(filter_.deadband_m, 0.00025)

    def test_adjustable_settings_are_bounded(self):
        updated = validated_phone_filter_settings(
            {"beta": 1.5}, base=DEFAULT_PHONE_FILTER_SETTINGS
        )
        self.assertEqual(updated["beta"], 1.5)
        with self.assertRaises(ValueError):
            validated_phone_filter_settings({"deadband_m": 0.5})

    def test_static_noise_rms_is_reduced_by_at_least_half(self):
        rng = np.random.default_rng(8128)
        raw = rng.normal(0.0, 0.003, size=(3000, 3))
        filter_ = OneEuroXYZFilter()
        filtered = np.asarray(
            [filter_.update(position, index / 30.0).deadband_position_m for index, position in enumerate(raw)]
        )

        raw_rms = float(np.sqrt(np.mean(raw[300:] ** 2)))
        filtered_rms = float(np.sqrt(np.mean(filtered[300:] ** 2)))
        self.assertLess(filtered_rms, raw_rms * 0.5)

    def test_gap_and_non_monotonic_timestamp_reset_state(self):
        filter_ = OneEuroXYZFilter()
        filter_.update([0.0, 0.0, 0.0], 1.0)
        self.assertFalse(filter_.update([0.1, 0.0, 0.0], 1.03).reset)

        gap = filter_.update([0.2, 0.0, 0.0], 1.24)
        self.assertTrue(gap.reset)
        self.assertEqual(gap.deadband_position_m, [0.2, 0.0, 0.0])
        self.assertTrue(filter_.update([0.3, 0.0, 0.0], 1.2).reset)

    def test_timestep_is_clamped_to_valid_range(self):
        filter_ = OneEuroXYZFilter()
        filter_.update([0.0, 0.0, 0.0], 0.0)
        fast = filter_.update([0.01, 0.0, 0.0], 0.001)
        slow = filter_.update([0.02, 0.0, 0.0], 0.151)
        self.assertAlmostEqual(fast.dt_s, 1.0 / 120.0)
        self.assertAlmostEqual(slow.dt_s, 0.1)


class ConstantVelocityKalmanXYZTest(unittest.TestCase):
    def test_state_contains_position_and_velocity(self):
        kalman = ConstantVelocityKalmanXYZ(measurement_std=0.001)
        states = [kalman.update([time_s, 0.0, 0.0], time_s) for time_s in np.arange(0.0, 2.0, 1 / 30)]
        self.assertEqual(states[-1].shape, (6,))
        self.assertAlmostEqual(states[-1][0], 59 / 30, places=3)
        self.assertAlmostEqual(states[-1][3], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
