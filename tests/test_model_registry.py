import tempfile
import unittest
from pathlib import Path

import torch

from mini_pi0.config.io import load_config
from mini_pi0.models.fm import ActionCNN1D, ActionUNet1D, CrossAttentionActionTransformer
from mini_pi0.models.registry import build_checkpoint_payload, count_params, list_models, load_checkpoint, make_model, save_checkpoint


class ModelRegistryTests(unittest.TestCase):
    def test_registry_exposes_supported_model(self):
        self.assertEqual(list_models(), ["mini_pi0_fm"])

    def test_removed_model_names_are_rejected(self):
        for model_name in ("mini_pi05", "mini_pi0_crossflow"):
            cfg = load_config(overrides=[f"model.name='{model_name}'"])
            with self.assertRaisesRegex(ValueError, "Unknown model"):
                make_model(cfg)

    def test_checkpoint_metadata_excludes_removed_fields(self):
        cfg = load_config(
            overrides=[
                "model.cond_dim=16",
                "model.d_model=16",
                "model.nhead=4",
                "model.nlayers=1",
            ]
        )
        model = make_model(cfg)
        total, trainable = count_params(model)

        self.assertGreater(total, 0)
        self.assertGreater(trainable, 0)

        payload = build_checkpoint_payload(model=model, cfg=cfg, epoch=0, loss=1.23)
        self.assertEqual(payload["model_name"], "mini_pi0_fm")
        self.assertNotIn("obs_mode", payload["model_config"])
        self.assertNotIn("vision_dim", payload["model_config"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            save_checkpoint(path, payload)
            ckpt = load_checkpoint(path, map_location="cpu")

        self.assertEqual(ckpt["model_name"], "mini_pi0_fm")
        self.assertIn("model", ckpt)

    def test_action_backbones_forward_and_sample(self):
        cases = (
            ("transformer", CrossAttentionActionTransformer, 6),
            ("cnn1d", ActionCNN1D, 6),
            ("unet1d", ActionUNet1D, 8),
        )
        for action_backbone, expected_type, chunk_size in cases:
            cfg = load_config(
                overrides=[
                    "model.action_dim=7",
                    "model.prop_dim=9",
                    f"model.chunk_size={chunk_size}",
                    "model.cond_dim=16",
                    "model.d_model=16",
                    "model.nhead=4",
                    "model.nlayers=1",
                    f"model.action_backbone='{action_backbone}'",
                    "model.conditioning_mode='cross_attention'",
                    "model.freeze_vision_backbone=true",
                    "model.vision_token_grid_size=2",
                ]
            )
            model = make_model(cfg)

            self.assertIsInstance(model.action_transformer, expected_type)

            img = torch.rand(2, 2, 3, 32, 32)
            prop = torch.randn(2, 9)
            actions = torch.randn(2, chunk_size, 7)
            loss = model(img, prop, actions)
            sample = model.sample(img[:1], prop[:1], n_steps=2)

            self.assertEqual(tuple(loss.shape), ())
            self.assertTrue(torch.isfinite(loss).item())
            self.assertEqual(tuple(sample.shape), (1, chunk_size, 7))


if __name__ == "__main__":
    unittest.main()
