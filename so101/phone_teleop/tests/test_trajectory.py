import unittest

import numpy as np

from so101.phone_teleop.trajectory import OnlineQuinticRetargeter, synchronized_quintic


class SynchronizedQuinticTest(unittest.TestCase):
    def setUp(self):
        self.velocity = np.array([30.0, 30.0, 30.0, 45.0, 60.0])
        self.acceleration = np.array([90.0, 90.0, 90.0, 135.0, 180.0])
        self.jerk = np.array([450.0, 450.0, 450.0, 675.0, 900.0])

    def test_exact_general_boundary_conditions_and_limits(self):
        position = np.array([2.0, -3.0, 4.0, 0.5, -1.0])
        velocity = np.array([1.0, -2.0, 0.5, 3.0, -4.0])
        acceleration = np.array([0.2, -0.1, 0.3, -0.4, 0.5])
        target = np.array([40.0, -25.0, 50.0, -30.0, 70.0])
        segment = synchronized_quintic(
            position, velocity, acceleration, target,
            self.velocity, self.acceleration, self.jerk,
        )

        start = segment.sample(0.0)
        end = segment.sample(segment.duration_s)
        np.testing.assert_allclose(start.position, position, atol=1e-12)
        np.testing.assert_allclose(start.velocity, velocity, atol=1e-12)
        np.testing.assert_allclose(start.acceleration, acceleration, atol=1e-12)
        np.testing.assert_allclose(end.position, target, atol=1e-10)
        np.testing.assert_allclose(end.velocity, 0.0, atol=1e-10)
        np.testing.assert_allclose(end.acceleration, 0.0, atol=1e-10)
        self.assertTrue(segment.within_limits(self.velocity, self.acceleration, self.jerk))

    def test_all_joints_share_the_same_arrival_time(self):
        segment = synchronized_quintic(
            np.zeros(5), np.zeros(5), np.zeros(5), [5, 25, -50, 10, 80],
            self.velocity, self.acceleration, self.jerk,
        )
        before = segment.sample(segment.duration_s * 0.9)
        end = segment.sample(segment.duration_s)
        self.assertTrue(np.all(np.abs(end.position - before.position) > 0.0))
        np.testing.assert_allclose(end.position, [5, 25, -50, 10, 80], atol=1e-10)

    def test_online_retarget_is_c2_continuous(self):
        retargeter = OnlineQuinticRetargeter(
            np.zeros(5), self.velocity, self.acceleration, self.jerk
        )
        retargeter.retarget([40, -20, 30, 10, 50], 0.0)
        boundary_before = retargeter.sample(0.37)
        replacement = retargeter.retarget([-20, 35, -10, 30, -40], 0.37)
        boundary_after = replacement.sample(0.0)

        np.testing.assert_allclose(boundary_after.position, boundary_before.position, atol=1e-12)
        np.testing.assert_allclose(boundary_after.velocity, boundary_before.velocity, atol=1e-12)
        np.testing.assert_allclose(boundary_after.acceleration, boundary_before.acceleration, atol=1e-12)
        self.assertTrue(replacement.within_limits(self.velocity, self.acceleration, self.jerk))


if __name__ == "__main__":
    unittest.main()
