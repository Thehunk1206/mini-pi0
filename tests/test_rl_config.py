import tempfile
import unittest
from pathlib import Path

from mini_pi0.config.io import load_config
from mini_pi0.rl.config import validate_rl_config


class RLConfigTests(unittest.TestCase):
    def test_validate_rl_config_accepts_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "best.pt"
            stats = Path(tmpdir) / "action_stats.json"
            ckpt.write_bytes(b"placeholder")
            stats.write_text('{"mean":[0.0],"std":[1.0]}', encoding="utf-8")
            cfg = load_config(overrides=[f"rl.checkpoint='{ckpt}'", f"rl.action_stats_path='{stats}'"])

            validate_rl_config(cfg)

    def test_validate_rl_config_rejects_invalid_algorithm(self):
        cfg = load_config(overrides=["rl.algorithm='sac'"])

        with self.assertRaisesRegex(ValueError, "rl.algorithm"):
            validate_rl_config(cfg, require_files=False)

    def test_validate_rl_config_requires_checkpoint_when_enabled(self):
        cfg = load_config(overrides=["rl.checkpoint='missing.pt'"])

        with self.assertRaises(FileNotFoundError):
            validate_rl_config(cfg, require_files=True)


if __name__ == "__main__":
    unittest.main()
