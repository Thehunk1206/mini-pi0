import unittest

from mini_pi0.config.io import load_config
from mini_pi0.utils.parity import build_checkpoint_parity_report, config_diff


class ParityTests(unittest.TestCase):
    def test_parity_report_detects_mismatches(self):
        cfg = load_config()
        ckpt = {
            "sim_backend": "maniskill3",
            "sim_config": {"task": "PickCube-v1", "robot": "panda_wristcam", "controller": "pd_ee_delta_pose"},
            "model_config": {"action_dim": 7, "prop_dim": 9, "chunk_size": 16},
            "robot_config": {"action_dim": 7, "image_key": "agentview_image"},
        }
        report = build_checkpoint_parity_report(cfg, ckpt)
        self.assertFalse(report["ok"])
        keys = {item["key"] for item in report["issues"]}
        self.assertIn("simulator.controller", keys)

    def test_parity_report_detects_image_keys_mismatch(self):
        cfg = load_config(overrides=["robot.image_keys=['observation.images.right_wrist_0_rgb']"])
        ckpt = {
            "model_config": {"action_dim": 7, "prop_dim": 9, "chunk_size": 16},
            "robot_config": {
                "action_dim": 7,
                "image_keys": ["observation.images.base_0_rgb", "observation.images.right_wrist_0_rgb"],
            },
        }
        report = build_checkpoint_parity_report(cfg, ckpt)
        self.assertFalse(report["ok"])
        keys = {item["key"] for item in report["issues"]}
        self.assertIn("robot.image_keys", keys)

    def test_config_diff_tracks_changes(self):
        cfg_a = load_config()
        cfg_b = load_config(overrides=["eval.n_episodes=10", "train.ema_decay=0.999"])
        diff = config_diff(cfg_a, cfg_b)
        self.assertIn("eval.n_episodes", diff)
        self.assertIn("train.ema_decay", diff)

    def test_parity_report_treats_missing_fm_conditioning_as_legacy_compatible(self):
        cfg = load_config(overrides=["model.conditioning_mode=cross_attention", "model.obs_horizon=2"])
        ckpt = {
            "model_config": {
                "action_dim": 7,
                "prop_dim": 9,
                "chunk_size": 16,
            },
        }

        report = build_checkpoint_parity_report(cfg, ckpt)

        keys = {item["key"] for item in report["issues"]}
        self.assertNotIn("model.conditioning_mode", keys)
        self.assertNotIn("model.obs_horizon", keys)


if __name__ == "__main__":
    unittest.main()
