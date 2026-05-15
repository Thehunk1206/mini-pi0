import tempfile
import unittest
from pathlib import Path

import torch

from mini_pi0.config.io import load_config
from mini_pi0.models.fm import ActionCNN1D, ActionUNet1D, AttentionFiLMPooler, CrossAttentionActionTransformer
from mini_pi0.models.registry import (
    build_checkpoint_payload,
    count_params,
    load_checkpoint,
    make_model,
    resolve_action_expert_intermediate_size,
    save_checkpoint,
)


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

    def test_action_expert_preset_resolves_size(self):
        size = resolve_action_expert_intermediate_size(
            action_model="ACTION_EXPERT_S",
            expert_intermediate_size=None,
        )

        self.assertEqual(size, 384)

    def test_action_expert_preset_rejects_conflicting_explicit_size(self):
        with self.assertRaises(ValueError):
            resolve_action_expert_intermediate_size(
                action_model="ACTION_EXPERT_S",
                expert_intermediate_size=768,
            )

    def test_action_expert_preset_is_saved_in_checkpoint_metadata(self):
        cfg = load_config(
            overrides=[
                "model.action_model=ACTION_EXPERT_M",
                "model.expert_intermediate_size=null",
            ]
        )
        model = make_model(cfg)

        payload = build_checkpoint_payload(model=model, cfg=cfg, epoch=0, loss=1.23)

        self.assertEqual(payload["model_config"]["action_model"], "ACTION_EXPERT_M")

    def test_minipi0_fm_can_use_cnn1d_action_backbone(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=32",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=6",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nlayers=2",
                "model.action_backbone='cnn1d'",
                "model.conditioning_mode='global'",
                "model.action_cnn_kernel_size=5",
            ]
        )
        model = make_model(cfg)

        self.assertIsInstance(model.action_transformer, ActionCNN1D)

        feat = torch.randn(2, 32)
        prop = torch.randn(2, 9)
        actions = torch.randn(2, 6, 7)
        loss = model(feat, prop, actions)
        sample = model.sample(feat[:1], prop[:1], n_steps=2)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(sample.shape), (1, 6, 7))

    def test_minipi0_fm_can_use_unet1d_action_backbone(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=32",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=8",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nlayers=2",
                "model.action_backbone='unet1d'",
                "model.conditioning_mode='global'",
                "model.action_cnn_kernel_size=5",
            ]
        )
        model = make_model(cfg)

        self.assertIsInstance(model.action_transformer, ActionUNet1D)

        feat = torch.randn(2, 32)
        prop = torch.randn(2, 9)
        actions = torch.randn(2, 8, 7)
        loss = model(feat, prop, actions)
        sample = model.sample(feat[:1], prop[:1], n_steps=2)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(sample.shape), (1, 8, 7))

    def test_minipi0_fm_checkpoint_metadata_saves_action_backbone(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=32",
                "model.action_backbone='unet1d'",
                "model.action_cnn_kernel_size=3",
                "model.dropout=0.2",
            ]
        )
        model = make_model(cfg)

        payload = build_checkpoint_payload(model=model, cfg=cfg, epoch=0, loss=1.23)

        self.assertEqual(payload["model_config"]["action_backbone"], "unet1d")
        self.assertEqual(payload["model_config"]["action_cnn_kernel_size"], 3)
        self.assertEqual(payload["model_config"]["dropout"], 0.2)

    def test_minipi0_fm_resnet_checkpoint_metadata_leaves_vision_model_name_null(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=32",
                "model.vision_backbone='resnet18'",
            ]
        )
        model = make_model(cfg)

        payload = build_checkpoint_payload(model=model, cfg=cfg, epoch=0, loss=1.23)

        self.assertEqual(payload["model_config"]["vision_backbone"], "resnet18")
        self.assertIsNone(payload["model_config"]["vision_model_name"])
        self.assertTrue(payload["model_config"]["vision_pretrained"])

    def test_minipi0_fm_can_finetune_resnet_backbone(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='image'",
                "model.freeze_vision_backbone=false",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nlayers=1",
            ]
        )
        model = make_model(cfg)

        assert model.obs_encoder.img_backbone is not None
        trainable = [p.requires_grad for p in model.obs_encoder.img_backbone.parameters()]

        self.assertTrue(trainable)
        self.assertTrue(all(trainable))

    def test_minipi0_fm_freezes_resnet_backbone_by_default(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='image'",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nlayers=1",
            ]
        )
        model = make_model(cfg)

        assert model.obs_encoder.img_backbone is not None
        trainable = [p.requires_grad for p in model.obs_encoder.img_backbone.parameters()]

        self.assertTrue(trainable)
        self.assertFalse(any(trainable))

    def test_minipi0_fm_defaults_to_cross_attention_transformer(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=32",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=6",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nhead=4",
                "model.nlayers=2",
                "model.action_backbone='transformer'",
            ]
        )
        model = make_model(cfg)

        self.assertIsInstance(model.action_transformer, CrossAttentionActionTransformer)

        feat = torch.randn(2, 32)
        prop = torch.randn(2, 9)
        actions = torch.randn(2, 6, 7)
        loss = model(feat, prop, actions)
        sample_euler = model.sample(feat[:1], prop[:1], n_steps=2, solver="euler")
        sample_heun = model.sample(feat[:1], prop[:1], n_steps=2, solver="heun")

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(sample_euler.shape), (1, 6, 7))
        self.assertEqual(tuple(sample_heun.shape), (1, 6, 7))

    def test_minipi0_fm_cross_attention_supports_observation_history(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='feature'",
                "model.vision_dim=16",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=5",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nhead=4",
                "model.nlayers=1",
                "model.obs_horizon=2",
            ]
        )
        model = make_model(cfg)

        feat = torch.randn(2, 2, 16)
        prop = torch.randn(2, 2, 9)
        actions = torch.randn(2, 5, 7)
        loss = model.compute_loss(feat, prop, actions, smoothness_weight=0.1, jerk_weight=0.1)
        sample = model.sample(feat[:1], prop[:1], n_steps=2)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(sample.shape), (1, 5, 7))

    def test_token_to_film_pooler_shape_and_gradients(self):
        pooler = AttentionFiLMPooler(action_dim=7, cond_dim=16, d_model=32, nhead=4)
        tokens = torch.randn(2, 5, 16, requires_grad=True)
        actions = torch.randn(2, 6, 7)
        tau = torch.rand(2)

        out = pooler(tokens, actions, tau)
        loss = out.square().mean()
        loss.backward()

        self.assertEqual(tuple(out.shape), (2, 32))
        self.assertIsNotNone(tokens.grad)

    def test_resnet_spatial_tokens_have_expected_count(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.obs_mode='image'",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=4",
                "model.cond_dim=16",
                "model.d_model=16",
                "model.nhead=4",
                "model.nlayers=1",
                "model.vision_token_grid_size=2",
                "model.freeze_vision_backbone=true",
            ]
        )
        model = make_model(cfg)

        img = torch.rand(1, 2, 3, 32, 32)
        prop = torch.randn(1, 9)
        tokens = model.obs_encoder.forward_tokens(img, prop)

        # Two cameras * 2x2 spatial tokens + one proprio token.
        self.assertEqual(tuple(tokens.shape), (1, 9, 16))


if __name__ == "__main__":
    unittest.main()
