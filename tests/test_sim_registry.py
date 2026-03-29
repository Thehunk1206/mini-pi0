import unittest

import numpy as np

from mini_pi0.config.io import load_config
from mini_pi0.sim.registry import backend_status, make_sim_adapter


class SimRegistryTests(unittest.TestCase):
    def test_backend_status_has_expected_keys(self):
        status = backend_status()
        self.assertIn("robosuite", status)
        self.assertIn("maniskill3", status)
        self.assertIn("isaaclab", status)
        self.assertEqual(status["maniskill3"]["status"], "scaffolded only")
        self.assertEqual(status["isaaclab"]["status"], "scaffolded only")

    def test_isaaclab_scaffold_raises(self):
        cfg = load_config(overrides=["simulator.backend=isaaclab"])
        adapter = make_sim_adapter(cfg)
        with self.assertRaises(Exception):
            adapter.reset(seed=0)

    def test_maniskill3_scaffold_raises(self):
        cfg = load_config(overrides=["simulator.backend=maniskill3"])
        adapter = make_sim_adapter(cfg)
        with self.assertRaises(Exception):
            adapter.reset(seed=0)


class RobosuiteAdapterSmoke(unittest.TestCase):
    def test_reset_step_close(self):
        cfg = load_config(
            overrides=[
                "simulator.backend=robosuite",
                "simulator.task=Lift",
                "simulator.has_renderer=false",
                "simulator.has_offscreen_renderer=true",
                "simulator.camera_width=84",
                "simulator.camera_height=84",
                "simulator.horizon=50",
            ]
        )
        adapter = make_sim_adapter(cfg)
        obs = adapter.reset(seed=0)
        self.assertIn("agentview_image", obs)
        lo, _hi = adapter.action_spec()
        step = adapter.step(np.zeros_like(lo, dtype=np.float32))
        self.assertIsInstance(step.reward, float)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
