import unittest
from types import SimpleNamespace
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


class _UnboundedBox:
    low = np.full((8,), -np.inf, dtype=np.float32)
    high = np.full((8,), np.inf, dtype=np.float32)
    shape = (8,)


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


class _FakeScene(dict):
    def __init__(self, *args, env_origins: np.ndarray, sensors=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_origins = env_origins
        self.sensors = sensors or {}


class _Runtime:
    def __init__(self) -> None:
        self.gym = _FakeGym()
        self.task_spec = resolve_isaaclab_task("franka_lift_cube")

    @staticmethod
    def parse_env_cfg(task_id, *, device, num_envs, use_fabric):
        return {
            "task_id": task_id,
            "device": device,
            "num_envs": num_envs,
            "use_fabric": use_fabric,
        }


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
        with (
            patch("mini_pi0.sim.isaaclab_adapter._load_runtime", return_value=runtime),
            patch("mini_pi0.sim.isaaclab_adapter._configure_front_camera"),
        ):
            adapter = IsaacLabAdapter(cfg)

        obs = adapter.reset(seed=3)
        step = adapter.step(np.zeros((7,), dtype=np.float32))

        self.assertEqual(runtime.gym.task_id, "Isaac-Lift-Cube-Franka-v0")
        self.assertEqual(runtime.gym.kwargs["cfg"]["task_id"], "Isaac-Lift-Cube-Franka-v0")
        self.assertEqual(obs["agentview_image"].shape, (8, 8, 3))
        self.assertEqual(obs["robot0_eef_pos"].shape, (3,))
        self.assertEqual(obs["robot0_eef_quat"].shape, (4,))
        self.assertEqual(step.reward, 0.5)
        self.assertFalse(step.done)

    def test_action_spec_returns_flat_bounds(self):
        cfg = load_config(overrides=["simulator.backend='isaaclab'", "simulator.task='franka_lift_cube'"])
        with (
            patch("mini_pi0.sim.isaaclab_adapter._load_runtime", return_value=_Runtime()),
            patch("mini_pi0.sim.isaaclab_adapter._configure_front_camera"),
        ):
            adapter = IsaacLabAdapter(cfg)

        low, high = adapter.action_spec()

        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.allclose(low, -1.0))
        self.assertTrue(np.allclose(high, 1.0))

    def test_action_spec_normalizes_unbounded_isaac_commands(self):
        cfg = load_config(
            overrides=[
                "simulator.backend='isaaclab'",
                "simulator.task='franka_lift_cube'",
                "robot.action_dim=8",
            ]
        )
        runtime = _Runtime()
        runtime.gym.env.action_space = _UnboundedBox()
        with (
            patch("mini_pi0.sim.isaaclab_adapter._load_runtime", return_value=runtime),
            patch("mini_pi0.sim.isaaclab_adapter._configure_front_camera"),
        ):
            adapter = IsaacLabAdapter(cfg)

        low, high = adapter.action_spec()

        self.assertEqual(low.shape, (8,))
        self.assertTrue(np.allclose(low, -1.0))
        self.assertTrue(np.allclose(high, 1.0))

    def test_image_from_sensor_selects_vector_row(self):
        adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
        sensor = SimpleNamespace(
            data=SimpleNamespace(
                output={
                    "rgb": np.stack(
                        [
                            np.zeros((4, 4, 3), dtype=np.uint8),
                            np.full((4, 4, 3), 127, dtype=np.uint8),
                        ]
                    )
                }
            )
        )
        adapter.env = SimpleNamespace(
            unwrapped=SimpleNamespace(scene=SimpleNamespace(sensors={"mini_pi0_front_camera": sensor}))
        )

        image = adapter._image_from_sensor(1)

        self.assertIsNotNone(image)
        self.assertEqual(int(np.asarray(image).mean()), 127)

    def test_scene_state_uses_semantic_isaac_entities(self):
        adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
        origins = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float32)
        ee_frame = SimpleNamespace(
            data=SimpleNamespace(
                target_pos_w=np.array([[[0.4, 0.0, 0.3]], [[2.9, 0.0, 0.3]]], dtype=np.float32),
                target_quat_w=np.array(
                    [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
                    dtype=np.float32,
                ),
            )
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=np.array(
                    [[0.0] * 7 + [0.03, 0.03], [0.0] * 7 + [0.04, 0.04]],
                    dtype=np.float32,
                ),
                root_pos_w=origins.copy(),
            )
        )
        obj = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=np.array([[0.5, 0.0, 0.3], [3.0, 0.0, 0.3]], dtype=np.float32)
            )
        )
        scene = _FakeScene(
            {"ee_frame": ee_frame, "robot": robot, "object": obj},
            env_origins=origins,
        )
        command_manager = SimpleNamespace(
            get_command=lambda name: np.array(
                [
                    [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        adapter.env = SimpleNamespace(
            unwrapped=SimpleNamespace(scene=scene, command_manager=command_manager)
        )

        state = adapter._scene_state(1)

        self.assertTrue(np.allclose(state["eef_pos"], [0.4, 0.0, 0.3]))
        self.assertTrue(np.allclose(state["gripper_pos"], [0.04, 0.04]))
        self.assertTrue(np.allclose(state["object_pos"], [0.5, 0.0, 0.3]))
        self.assertEqual(state["task_progress"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
