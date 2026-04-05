import unittest

import torch

from mini_pi0.config.io import load_config
from mini_pi0.models.registry import make_model


class CrossFlowModelTests(unittest.TestCase):
    def test_crossflow_image_mode_forward_and_sample(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_crossflow'",
                "model.obs_mode='image'",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=8",
                "model.d_model=64",
                "model.nhead=4",
                "model.nlayers=2",
                "model.vision_token_grid_size=3",
                "model.use_dit_adaln=true",
            ]
        )
        model = make_model(cfg)
        model.eval()

        img = torch.rand(2, 3, 84, 84, dtype=torch.float32)
        prop = torch.randn(2, 9, dtype=torch.float32)
        actions = torch.randn(2, 8, 7, dtype=torch.float32)

        loss = model(img, prop, actions)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())

        out = model.sample(img[:1], prop[:1], n_steps=4)
        self.assertEqual(tuple(out.shape), (1, 8, 7))

    def test_crossflow_feature_mode_forward_and_sample(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_crossflow'",
                "model.obs_mode='feature'",
                "model.vision_dim=128",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=6",
                "model.d_model=64",
                "model.nhead=4",
                "model.nlayers=2",
            ]
        )
        model = make_model(cfg)
        model.eval()

        feat = torch.randn(3, 128, dtype=torch.float32)
        prop = torch.randn(3, 9, dtype=torch.float32)
        actions = torch.randn(3, 6, 7, dtype=torch.float32)

        loss = model(feat, prop, actions)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())

        out = model.sample(feat[:1], prop[:1], n_steps=3)
        self.assertEqual(tuple(out.shape), (1, 6, 7))


if __name__ == "__main__":
    unittest.main()

