import tempfile
import unittest
from pathlib import Path

from mini_pi0.config.io import load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_are_nested_dataclasses(self):
        cfg = load_config()
        self.assertEqual(cfg.simulator.backend, "robosuite")
        self.assertEqual(cfg.model.name, "mini_pi0_fm")
        self.assertEqual(cfg.data.format, "robomimic_hdf5")
        self.assertEqual(cfg.train.batch_size, 256)

    def test_yaml_and_overrides(self):
        text = """
experiment:
  name: exp-a
simulator:
  backend: maniskill3
train:
  epochs: 3
"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.yaml"
            p.write_text(text, encoding="utf-8")
            cfg = load_config(str(p), overrides=["train.epochs=7", "eval.record=true", "eval.grid_size=4"])

        self.assertEqual(cfg.experiment.name, "exp-a")
        self.assertEqual(cfg.simulator.backend, "maniskill3")
        self.assertEqual(cfg.train.epochs, 7)
        self.assertTrue(cfg.eval.record)
        self.assertEqual(cfg.eval.grid_size, 4)


if __name__ == "__main__":
    unittest.main()
