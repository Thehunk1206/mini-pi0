import unittest

from mini_pi0.sim.isaaclab_tasks import list_isaaclab_tasks, resolve_isaaclab_task


class IsaacLabTaskRegistryTests(unittest.TestCase):
    def test_resolve_known_alias_returns_lift_spec(self):
        spec = resolve_isaaclab_task("lift")

        self.assertEqual(spec.key, "franka_lift_cube")
        self.assertEqual(spec.gym_id, "Isaac-Lift-Cube-Franka-v0")
        self.assertIn("robot0_eef_pos", spec.state_keys)

    def test_direct_isaac_task_id_is_allowed(self):
        spec = resolve_isaaclab_task("Isaac-Custom-Franka-v0")

        self.assertEqual(spec.gym_id, "Isaac-Custom-Franka-v0")

    def test_unknown_task_raises_with_options(self):
        with self.assertRaisesRegex(ValueError, "Known mini-pi0 task keys"):
            resolve_isaaclab_task("not-a-real-task")

    def test_list_tasks_contains_planned_targets(self):
        tasks = list_isaaclab_tasks()

        self.assertIn("franka_lift_cube", tasks)
        self.assertIn("franka_peg_insertion", tasks)


if __name__ == "__main__":
    unittest.main()
