import unittest
from unittest.mock import patch

from mini_pi0.config.io import load_config
from mini_pi0.sim.maniskill3_adapter import (
    ManiSkill3Adapter,
    default_maniskill_reward_mode,
    make_maniskill_env_with_reward_fallback,
)
from mini_pi0.sim.registry import backend_status, make_sim_adapter


class SimRegistryTests(unittest.TestCase):
    def test_backend_status_has_expected_keys(self):
        status = backend_status()
        self.assertIn("maniskill3", status)
        self.assertIn("isaaclab", status)
        self.assertIn("implemented", status["maniskill3"]["status"])
        self.assertEqual(status["isaaclab"]["status"], "scaffolded only")

    def test_isaaclab_scaffold_raises(self):
        cfg = load_config(overrides=["simulator.backend=isaaclab"])
        adapter = make_sim_adapter(cfg)
        with self.assertRaises(Exception):
            adapter.reset(seed=0)

    def test_maniskill3_adapter_constructs_or_reports_dependency(self):
        cfg = load_config(overrides=["simulator.backend=maniskill3"])
        try:
            adapter = make_sim_adapter(cfg)
        except Exception as e:
            self.assertTrue(
                "mani_skill" in str(e).lower()
                or "render" in str(e).lower()
                or "device" in str(e).lower()
                or "name not found" in str(e).lower()
            )
            return
        try:
            adapter.reset(seed=0)
        except Exception as e:
            # In CI/headless machines a render/physics device may be unavailable.
            self.assertTrue("render" in str(e).lower() or "device" in str(e).lower())
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    def test_maniskill3_adapter_check_success_uses_native_success_flag(self):
        adapter = object.__new__(ManiSkill3Adapter)

        self.assertTrue(adapter.check_success(info={"success": True}))
        self.assertTrue(adapter.check_success(info={"success": 1.0}))
        self.assertFalse(adapter.check_success(info={"success": False, "success_fraction": 0.0}))

    def test_default_maniskill_reward_mode_respects_reward_shaping_flag(self):
        sparse_cfg = load_config(overrides=["simulator.reward_shaping=false"])
        dense_cfg = load_config(overrides=["simulator.reward_shaping=true"])

        self.assertEqual(default_maniskill_reward_mode(sparse_cfg), "sparse")
        self.assertEqual(default_maniskill_reward_mode(dense_cfg), "dense")

    def test_maniskill_env_creation_retries_sparse_when_dense_is_unsupported(self):
        calls = []
        expected_env = object()

        def fake_make(task_id, **kwargs):
            calls.append((task_id, kwargs["reward_mode"]))
            if kwargs["reward_mode"] == "dense":
                raise NotImplementedError("Unsupported reward mode: dense")
            return expected_env

        with patch("mini_pi0.sim.maniskill3_adapter.gym.make", side_effect=fake_make):
            env = make_maniskill_env_with_reward_fallback("StackPyramid-v1", {"reward_mode": "dense"})

        self.assertIs(env, expected_env)
        self.assertEqual(calls, [("StackPyramid-v1", "dense"), ("StackPyramid-v1", "sparse")])

if __name__ == "__main__":
    unittest.main()
