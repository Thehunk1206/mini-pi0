from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mini_pi0.dataset.obs_processor import ObsProcessor
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.eval.core import _blend_with_previous_tail
import mini_pi0.models.fm as fm_module
from mini_pi0.models.fm import MiniPi0FlowMatching


@pytest.fixture()
def action_stats_path(tmp_path):
    """Create a small action stats file for processor tests."""

    path = tmp_path / "action_stats.json"
    ActionStats(mean=np.zeros(7, dtype=np.float32), std=np.ones(7, dtype=np.float32)).save(str(path))
    return path


def _obs(value: int) -> dict[str, np.ndarray]:
    """Build one synthetic canonical observation."""

    image = np.full((8, 8, 3), value, dtype=np.uint8)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 1,
        "robot0_eef_pos": np.full(3, value, dtype=np.float32),
        "robot0_eef_quat": np.full(4, value, dtype=np.float32),
        "robot0_gripper_qpos": np.full(2, value, dtype=np.float32),
    }


def test_obs_processor_history_repeat_pads_and_preserves_camera_axis(action_stats_path) -> None:
    processor = ObsProcessor(
        action_stats_path=str(action_stats_path),
        image_key="agentview_image",
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        obs_horizon=2,
        preserve_camera_dim=True,
    )

    img0, prop0 = processor.obs_to_tensors(_obs(2))
    img1, prop1 = processor.obs_to_tensors(_obs(4))

    assert tuple(img0.shape) == (1, 2, 2, 3, 8, 8)
    assert tuple(prop0.shape) == (1, 2, 9)
    assert torch.allclose(prop0[:, 0], prop0[:, 1])
    assert tuple(img1.shape) == (1, 2, 2, 3, 8, 8)
    assert torch.all(prop1[:, 0] == 2)
    assert torch.all(prop1[:, 1] == 4)


def test_blend_with_previous_tail_uses_configured_prefix() -> None:
    chunk = np.ones((4, 2), dtype=np.float32)
    tail = np.zeros((2, 2), dtype=np.float32)

    out = _blend_with_previous_tail(chunk, tail, blend=0.25)

    assert np.allclose(out[:2], 0.75)
    assert np.allclose(out[2:], 1.0)


class _ConstantVelocityDenoiser(nn.Module):
    """Tiny denoiser used to verify FM reconstruction algebra."""

    chunk_size = 2
    action_dim = 2

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return a constant velocity field with the same shape as input actions."""

        return torch.ones_like(noisy_actions)


def test_flow_loss_components_reconstructs_clean_estimate_from_noisy_state(monkeypatch) -> None:
    model = MiniPi0FlowMatching(
        action_dim=2,
        prop_dim=3,
        obs_mode="feature",
        vision_dim=4,
        chunk_size=2,
        cond_dim=8,
        d_model=8,
        nhead=2,
        nlayers=1,
        conditioning_mode="global",
    )
    model.action_transformer = _ConstantVelocityDenoiser()
    monkeypatch.setattr(
        fm_module,
        "sample_tau_beta",
        lambda batch_size, device, dtype=torch.float32, **_: torch.full((batch_size,), 0.25, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    clean_actions = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    img = torch.zeros(1, 4)
    prop = torch.zeros(1, 3)

    _loss, clean_pred = model._flow_loss_components(img, prop, clean_actions)

    expected_noisy = 0.25 * clean_actions
    expected_clean_pred = expected_noisy + 0.75 * torch.ones_like(clean_actions)
    assert torch.allclose(clean_pred, expected_clean_pred)


def test_cross_attention_transformer_uses_configured_dropout() -> None:
    model = MiniPi0FlowMatching(
        action_dim=7,
        prop_dim=9,
        obs_mode="feature",
        vision_dim=16,
        chunk_size=6,
        cond_dim=32,
        d_model=32,
        nhead=4,
        nlayers=2,
        action_backbone="transformer",
        conditioning_mode="cross_attention",
        dropout=0.25,
    )
    layer = model.action_transformer.layers[0]

    assert layer.self_attn.dropout == pytest.approx(0.25)
    assert layer.cross_attn.dropout == pytest.approx(0.25)


def test_cnn1d_and_unet_token_film_forward_sample() -> None:
    for backbone in ("cnn1d", "unet1d"):
        model = MiniPi0FlowMatching(
            action_dim=7,
            prop_dim=9,
            obs_mode="feature",
            vision_dim=16,
            chunk_size=6,
            cond_dim=32,
            d_model=32,
            nhead=4,
            nlayers=2,
            action_backbone=backbone,
            conditioning_mode="cross_attention",
        )
        feat = torch.randn(2, 2, 16)
        prop = torch.randn(2, 2, 9)
        actions = torch.randn(2, 6, 7)

        loss = model(feat, prop, actions)
        sample = model.sample(feat[:1], prop[:1], n_steps=2)

        assert torch.isfinite(loss)
        assert tuple(sample.shape) == (1, 6, 7)


def test_timm_token_backbone_returns_tokens_when_available() -> None:
    pytest.importorskip("timm")
    try:
        model = MiniPi0FlowMatching(
            action_dim=7,
            prop_dim=9,
            obs_mode="image",
            chunk_size=4,
            cond_dim=16,
            d_model=16,
            nhead=4,
            nlayers=1,
            vision_backbone="timm",
            vision_model_name="vit_small_patch16_224",
            vision_pretrained=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"timm token model unavailable in this environment: {exc}")
    img = torch.rand(1, 3, 256, 256)
    prop = torch.randn(1, 9)

    tokens = model.obs_encoder.forward_tokens(img, prop)

    assert tokens.ndim == 3
    assert tokens.shape[0] == 1
    assert tuple(model.obs_encoder.img_backbone.image_mean.shape) == (1, 3, 1, 1)
    assert tuple(model.obs_encoder.img_backbone.image_std.shape) == (1, 3, 1, 1)


def test_resnet18_backbone_registers_imagenet_preprocessing_buffers() -> None:
    model = MiniPi0FlowMatching(
        action_dim=7,
        prop_dim=9,
        obs_mode="image",
        chunk_size=4,
        cond_dim=16,
        d_model=16,
        nhead=4,
        nlayers=1,
        vision_backbone="resnet18",
    )
    backbone = model.obs_encoder.img_backbone

    assert tuple(backbone.image_mean.shape) == (1, 3, 1, 1)
    assert tuple(backbone.image_std.shape) == (1, 3, 1, 1)
    assert torch.allclose(backbone.image_mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.allclose(backbone.image_std.flatten(), torch.tensor([0.229, 0.224, 0.225]))


def test_timm_backbone_requires_model_name() -> None:
    pytest.importorskip("timm")

    with pytest.raises(ValueError, match="vision_model_name is required"):
        MiniPi0FlowMatching(
            action_dim=7,
            prop_dim=9,
            obs_mode="image",
            chunk_size=4,
            cond_dim=16,
            d_model=16,
            nhead=4,
            nlayers=1,
            vision_backbone="timm",
            vision_model_name=None,
            vision_pretrained=False,
        )
