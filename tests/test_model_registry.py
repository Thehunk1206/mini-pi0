import tempfile
import unittest
from pathlib import Path

from mini_pi0.config.io import load_config
from mini_pi0.models.registry import build_checkpoint_payload, count_params, load_checkpoint, make_model, save_checkpoint


class ModelRegistryTests(unittest.TestCase):
    def test_model_build_and_checkpoint_metadata(self):
        cfg = load_config(overrides=["train.epochs=1", "data.n_demos=1"])
        model = make_model(cfg)
        total, trainable = count_params(model)
        self.assertGreater(total, 0)
        self.assertGreater(trainable, 0)

        payload = build_checkpoint_payload(model=model, cfg=cfg, epoch=0, loss=1.23)
        self.assertIn("model", payload)
        self.assertEqual(payload["model_name"], cfg.model.name)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best.pt"
            save_checkpoint(p, payload)
            ckpt = load_checkpoint(p, map_location="cpu")

        self.assertIn("model", ckpt)
        self.assertIn("model_config", ckpt)
        self.assertEqual(ckpt["model_name"], cfg.model.name)


if __name__ == "__main__":
    unittest.main()
