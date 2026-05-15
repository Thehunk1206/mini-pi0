import unittest

from mini_pi0.config.io import load_config
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

if __name__ == "__main__":
    unittest.main()
