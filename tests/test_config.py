import tempfile
import unittest
from pathlib import Path

from mini_pi0.config.io import load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_are_nested_dataclasses(self):
        cfg = load_config()
        self.assertEqual(cfg.simulator.backend, "maniskill3")
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

    def test_eval_vectorized_overrides(self):
        cfg = load_config(overrides=["eval.vectorized=true", "eval.num_envs=8", "eval.grid_cameras=['hand_camera']"])

        self.assertTrue(cfg.eval.vectorized)
        self.assertEqual(cfg.eval.num_envs, 8)
        self.assertEqual(cfg.eval.grid_cameras, ["hand_camera"])

    def test_lerobot_v3_fields_are_supported(self):
        cfg = load_config(
            overrides=[
                "data.format='lerobot_v3'",
                "data.lerobot_repo_id='local/test'",
                "data.lerobot_root='data/lerobot/test'",
                "data.lerobot_episodes=[1,2]",
                "data.lerobot_image_keys=['observation.images.agentview_image']",
            ]
        )

        self.assertEqual(cfg.data.format, "lerobot_v3")
        self.assertEqual(cfg.data.lerobot_repo_id, "local/test")
        self.assertEqual(cfg.data.lerobot_root, "data/lerobot/test")
        self.assertEqual(cfg.data.lerobot_episodes, [1, 2])
        self.assertEqual(cfg.data.lerobot_state_key, "observation.state")

    def test_removed_feature_mode_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown config key"):
            load_config(overrides=["model.obs_mode='feature'"])

        with self.assertRaisesRegex(ValueError, "Unknown config key"):
            load_config(overrides=["data.observation_mode='precomputed'"])

    def test_so101_preprocessing_and_training_controls_are_supported(self):
        cfg = load_config(
            overrides=[
                "robot.image_resize_hw=[224,224]",
                "robot.image_resize_mode='letterbox'",
                "data.sample_stride=3",
                "data.normalize_state=true",
                "model.freeze_vision_batchnorm_stats=true",
                "train.lr_observation=1e-4",
                "train.checkpoint_every_epochs=10",
            ]
        )

        self.assertEqual(cfg.robot.image_resize_hw, [224, 224])
        self.assertEqual(cfg.data.sample_stride, 3)
        self.assertTrue(cfg.data.normalize_state)
        self.assertTrue(cfg.model.freeze_vision_batchnorm_stats)
        self.assertEqual(cfg.train.checkpoint_every_epochs, 10)


if __name__ == "__main__":
    unittest.main()
