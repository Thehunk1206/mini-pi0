import unittest
from unittest.mock import patch

import numpy as np

from mini_pi0.config.io import load_config
from mini_pi0.sim.isaaclab_adapter import IsaacLabAdapter
from mini_pi0.sim.isaaclab_tasks import resolve_isaaclab_task


class _Box:
    def __init__(self) -> None:
        self.low = -np.ones((7,), dtype=np.float32)
        self.high = np.ones((7,), dtype=np.float32)
        self.shape = (7,)


class _FakeEnv:
    action_space = _Box()

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, seed=None):
        return self._obs(), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        return self._obs(), 0.5, False, False, {"success": self.step_count > 1}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        return None

    @staticmethod
    def _obs():
        return {
            "rgb": np.full((8, 8, 3), 128, dtype=np.uint8),
            "policy": np.arange(16, dtype=np.float32),
            "object_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }


class _FakeGym:
    def __init__(self) -> None:
        self.env = _FakeEnv()

    def make(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        return self.env


class _Runtime:
    def __init__(self) -> None:
        self.gym = _FakeGym()
        self.task_spec = resolve_isaaclab_task("franka_lift_cube")


class IsaacLabAdapterTests(unittest.TestCase):
    def test_adapter_maps_reset_and_step_to_canonical_schema(self):
        cfg = load_config(
            overrides=[
                "simulator.backend='isaaclab'",
                "simulator.task='franka_lift_cube'",
                "robot.image_keys=['agentview_image']",
                "robot.state_keys=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','observation.state.object']",
            ]
        )
        runtime = _Runtime()
        with patch("mini_pi0.sim.isaaclab_adapter._load_runtime", return_value=runtime):
            adapter = IsaacLabAdapter(cfg)

        obs = adapter.reset(seed=3)
        step = adapter.step(np.zeros((7,), dtype=np.float32))

        self.assertEqual(runtime.gym.task_id, "Isaac-Lift-Cube-Franka-v0")
        self.assertEqual(obs["agentview_image"].shape, (8, 8, 3))
        self.assertEqual(obs["robot0_eef_pos"].shape, (3,))
        self.assertEqual(obs["robot0_eef_quat"].shape, (4,))
        self.assertEqual(step.reward, 0.5)
        self.assertFalse(step.done)

    def test_action_spec_returns_flat_bounds(self):
        cfg = load_config(overrides=["simulator.backend='isaaclab'", "simulator.task='franka_lift_cube'"])
        with patch("mini_pi0.sim.isaaclab_adapter._load_runtime", return_value=_Runtime()):
            adapter = IsaacLabAdapter(cfg)

        low, high = adapter.action_spec()

        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.allclose(low, -1.0))
        self.assertTrue(np.allclose(high, 1.0))


if __name__ == "__main__":
    unittest.main()
