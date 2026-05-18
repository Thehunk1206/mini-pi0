import unittest

import numpy as np
import torch

from mini_pi0.config.io import load_config
from mini_pi0.dataset.episodes import EpisodeData
from mini_pi0.deploy.sim_runner import _resolve_deploy_rollout_controls
from mini_pi0.eval.core import _resolve_eval_rollout_controls
from mini_pi0.models.registry import make_model
from mini_pi0.train.optim import ExponentialMovingAverage
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.train.augmentation import GpuBatchProcessor
from mini_pi0.train.runner import _augment_actions, _augment_image_batch, _build_dataloaders, _build_optimizer, _curate_episodes
from mini_pi0.train.samplers import BlockShuffleSampler, dataset_prefers_locality_sampler, locality_order_for_dataset


def _make_episode(length: int, action_scale: float, object_delta: float) -> EpisodeData:
    actions = (np.random.randn(length, 7) * action_scale).astype(np.float32)
    obs = []
    for t in range(length):
        frac = float(t) / max(1.0, float(length - 1))
        obj = np.array([object_delta * frac, 0.0, 0.0], dtype=np.float32)
        obs.append(
            {
                "observation.state.object": obj,
                "observation.state.eef_pos": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "observation.images.right_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        )
    return EpisodeData(obs=obs, actions=actions)


class _LocalityTensorDataset(torch.utils.data.TensorDataset):
    prefers_locality_sampler = True


class TrainingStabilityControlTests(unittest.TestCase):
    def test_episode_curation_filters_low_quality_episodes(self):
        good = _make_episode(length=40, action_scale=0.5, object_delta=0.15)
        short = _make_episode(length=8, action_scale=0.5, object_delta=0.15)
        no_progress = _make_episode(length=40, action_scale=1e-4, object_delta=1e-4)

        cfg = load_config(
            overrides=[
                "data.filter_min_episode_length=16",
                "data.filter_min_action_std=0.01",
                "data.filter_min_state_delta=0.02",
                "data.filter_state_delta_key='observation.state.object'",
                "data.filter_drop_nan=true",
            ]
        )
        kept, summary = _curate_episodes([good, short, no_progress], cfg)
        self.assertEqual(len(kept), 1)
        self.assertEqual(summary["before_episodes"], 3)
        self.assertEqual(summary["after_episodes"], 1)
        self.assertEqual(summary["removed_episodes"], 2)
        self.assertTrue(summary["reasons"])

    def test_eval_warmup_rollout_controls(self):
        cfg = load_config(
            overrides=[
                "eval.execute_steps=4",
                "eval.n_flow_steps=10",
                "eval.action_smoothing_alpha=0.0",
                "eval.stability_warmup_steps=20",
                "eval.stability_warmup_execute_steps=2",
                "eval.stability_warmup_n_flow_steps=16",
                "eval.stability_warmup_action_smoothing_alpha=0.2",
            ]
        )
        warm = _resolve_eval_rollout_controls(cfg, env_steps_done=5)
        steady = _resolve_eval_rollout_controls(cfg, env_steps_done=25)
        self.assertEqual(warm, (2, 16, 0.2))
        self.assertEqual(steady, (4, 10, 0.0))

    def test_deploy_warmup_rollout_controls(self):
        cfg = load_config(
            overrides=[
                "deploy.execute_steps=4",
                "deploy.n_flow_steps=10",
                "deploy.action_smoothing_alpha=0.0",
                "deploy.stability_warmup_steps=30",
                "deploy.stability_warmup_execute_steps=1",
                "deploy.stability_warmup_n_flow_steps=20",
                "deploy.stability_warmup_action_smoothing_alpha=0.3",
            ]
        )
        warm = _resolve_deploy_rollout_controls(cfg, env_steps_done=0)
        steady = _resolve_deploy_rollout_controls(cfg, env_steps_done=35)
        self.assertEqual(warm, (1, 20, 0.3))
        self.assertEqual(steady, (4, 10, 0.0))

    def test_minipi0_fm_optimizer_uses_backbone_and_expert_lrs(self):
        cfg = load_config(
            overrides=[
                "model.name='mini_pi0_fm'",
                "model.action_dim=7",
                "model.prop_dim=9",
                "model.chunk_size=4",
                "model.cond_dim=32",
                "model.d_model=32",
                "model.nhead=4",
                "model.nlayers=2",
                "model.action_backbone='cnn1d'",
                "train.lr=1e-4",
                "train.lr_backbone=2e-5",
                "train.lr_expert=8e-5",
            ]
        )
        model = make_model(cfg)

        optimizer, lr_summary = _build_optimizer(model, cfg)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertAlmostEqual(lr_summary["backbone_lr"], 2e-5, places=10)
        self.assertAlmostEqual(lr_summary["expert_lr"], 8e-5, places=10)
        groups_by_name = {str(g.get("name")): float(g["lr"]) for g in optimizer.param_groups}
        self.assertEqual(groups_by_name["backbone"], 2e-5)
        self.assertEqual(groups_by_name["expert"], 8e-5)

    def test_training_augmentations_image_and_action(self):
        cfg = load_config(
            overrides=[
                "train.image_aug_enable=true",
                "train.image_aug_crop_scale=0.9",
                "train.image_aug_brightness=0.2",
                "train.image_aug_contrast=0.2",
                "train.image_aug_saturation=0.2",
                "train.action_noise_std=0.05",
                "train.action_noise_clip=1.0",
            ]
        )
        torch.manual_seed(0)
        img = torch.rand(4, 3, 32, 32)
        actions = torch.zeros(4, 8, 7)

        img_aug = _augment_image_batch(img, cfg)
        actions_aug = _augment_actions(actions, cfg)

        self.assertEqual(tuple(img_aug.shape), (4, 3, 32, 32))
        self.assertTrue(torch.all(img_aug >= 0.0).item())
        self.assertTrue(torch.all(img_aug <= 1.0).item())
        self.assertFalse(torch.allclose(img_aug, img))
        self.assertEqual(tuple(actions_aug.shape), (4, 8, 7))
        self.assertFalse(torch.allclose(actions_aug, actions))
        self.assertTrue(torch.all(actions_aug.abs() <= 1.0 + 1e-6).item())

    def test_gpu_batch_processor_normalizes_raw_actions_on_device(self):
        cfg = load_config(overrides=["train.image_aug_enable=false", "train.action_noise_std=0.0"])
        stats = ActionStats(mean=np.array([1.0, 2.0], dtype=np.float32), std=np.array([2.0, 4.0], dtype=np.float32))
        processor = GpuBatchProcessor(
            cfg=cfg,
            device=torch.device("cpu"),
            action_stats=stats,
            normalize_actions=True,
        )
        img = torch.full((1, 3, 8, 8), 255, dtype=torch.uint8)
        prop = torch.ones(1, 3)
        actions = torch.tensor([[[3.0, 10.0]]])

        img_out, prop_out, actions_out = processor.train_batch(img, prop, actions)

        self.assertEqual(img_out.device.type, "cpu")
        self.assertTrue(torch.allclose(img_out.max(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(prop_out, prop.float()))
        self.assertTrue(torch.allclose(actions_out, torch.tensor([[[1.0, 2.0]]])))

    def test_dataloader_uses_prefetch_factor_with_workers(self):
        cfg = load_config(
            overrides=[
                "train.batch_size=2",
                "train.num_workers=2",
                "train.persistent_workers=true",
                "train.prefetch_factor=3",
                "train.dataloader_in_order=false",
            ]
        )
        dataset = torch.utils.data.TensorDataset(torch.arange(8), torch.arange(8), torch.arange(8))

        loader, _, num_workers, persistent, prefetch = _build_dataloaders(
            dataset,
            None,
            cfg,
            torch.device("cpu"),
        )

        self.assertEqual(num_workers, 2)
        self.assertTrue(persistent)
        self.assertEqual(prefetch, 3)
        self.assertEqual(loader.prefetch_factor, 3)
        if hasattr(loader, "in_order"):
            self.assertFalse(loader.in_order)

    def test_block_shuffle_sampler_yields_all_indices_once(self):
        sampler = BlockShuffleSampler(range(10), block_size=4, seed=0)

        indices = list(sampler)

        self.assertEqual(sorted(indices), list(range(10)))
        self.assertEqual(len(indices), 10)

    def test_locality_order_sorts_subset_by_source_index(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(8))
        subset = torch.utils.data.Subset(dataset, [5, 2, 7, 1])

        order = locality_order_for_dataset(subset)

        self.assertEqual(order, (3, 1, 0, 2))

    def test_dataloader_auto_uses_block_shuffle_for_locality_dataset(self):
        cfg = load_config(
            overrides=[
                "train.batch_size=2",
                "train.num_workers=0",
                "train.sample_order='auto'",
                "train.block_shuffle_size=4",
                "train.block_shuffle_within_block=false",
            ]
        )
        dataset = _LocalityTensorDataset(torch.arange(8), torch.arange(8), torch.arange(8))

        loader, _, _, _, _ = _build_dataloaders(dataset, None, cfg, torch.device("cpu"))

        self.assertTrue(dataset_prefers_locality_sampler(dataset))
        self.assertIsInstance(loader.sampler, BlockShuffleSampler)
        self.assertEqual(loader.sampler.block_size, 4)
        self.assertFalse(loader.sampler.shuffle_within_block)

    def test_ema_update_aligns_shadow_dtype_after_resume(self):
        model = torch.nn.Linear(2, 2).float()
        ema = ExponentialMovingAverage(model, decay=0.5)
        state = ema.state_dict()
        model = model.double()
        ema.load_state_dict(state)

        ema.update(model)

        for name, value in model.state_dict().items():
            self.assertEqual(ema.shadow[name].dtype, value.dtype)
            self.assertEqual(ema.shadow[name].device, value.device)

    def test_validation_ema_is_disabled_by_default(self):
        cfg = load_config(overrides=["train.ema_decay=0.999", "train.checkpoint_use_ema=true"])

        self.assertFalse(cfg.train.val_use_ema)
        self.assertTrue(cfg.train.checkpoint_use_ema)


if __name__ == "__main__":
    unittest.main()
