"""Tests for ReinFlow policy sampling and deterministic evaluation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mini_pi0.config.io import load_config
from mini_pi0.rl.flow_policy import ReinFlowActorCritic


class _TinyActionTransformer(nn.Module):
    def __init__(self, action_dim: int, chunk_size: int, cond_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_proj = nn.Linear(cond_dim, action_dim)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return 0.1 * actions + self.cond_proj(cond).unsqueeze(1) + tau.view(-1, 1, 1) * 0.01


class _TinyFlowPolicy(nn.Module):
    def __init__(self, action_dim: int, chunk_size: int, cond_dim: int, prop_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(prop_dim, cond_dim)
        self.action_transformer = _TinyActionTransformer(action_dim, chunk_size, cond_dim)

    def _encode_conditioning(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        del img
        return self.encoder(prop)


def _cfg(*extra: str):
    return load_config(
        overrides=[
            "model.action_dim=2",
            "model.chunk_size=3",
            "model.prop_dim=5",
            "model.cond_dim=4",
            "model.d_model=8",
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.flow_steps=2",
            "rl.execution_horizon=1",
            "rl.noise_std_min=0.05",
            "rl.noise_std_max=0.12",
            "rl.noise_std_init=0.08",
            "rl.freeze_vision_during_rl=False",
            *extra,
        ]
    )


def _policy(cfg) -> ReinFlowActorCritic:
    base = _TinyFlowPolicy(
        action_dim=cfg.model.action_dim,
        chunk_size=cfg.model.chunk_size,
        cond_dim=cfg.model.cond_dim,
        prop_dim=cfg.model.prop_dim,
    )
    return ReinFlowActorCritic(cfg, policy=base)


def test_sample_path_returns_expected_shapes_and_bounded_noise() -> None:
    cfg = _cfg()
    policy = _policy(cfg)
    image = torch.zeros(4, 3, 8, 8)
    proprio = torch.randn(4, cfg.model.prop_dim)

    output = policy.sample_path(image, proprio)
    cond = policy.actor.encode_conditioning(image, proprio)
    std = policy.actor.kernel.noise(output.path[:, 0], torch.zeros(4), cond)

    assert output.path.shape == (4, 3, 3, 2)
    assert output.action_chunk.shape == (4, 3, 2)
    assert torch.isfinite(output.log_prob).all()
    assert torch.allclose(std, torch.full_like(std, 0.08), atol=1e-6)


def test_evaluate_path_recomputes_sampled_logprob() -> None:
    cfg = _cfg()
    policy = _policy(cfg)
    image = torch.zeros(2, 3, 8, 8)
    proprio = torch.randn(2, cfg.model.prop_dim)
    bounds = (torch.full((1, 1, 2), -1.0), torch.full((1, 1, 2), 1.0))

    output = policy.sample_path(image, proprio, bounds=bounds)
    evaluated = policy.evaluate_path(image, proprio, output.path, bounds=bounds)

    assert torch.allclose(output.log_prob, evaluated.log_prob, atol=1e-5)


def test_deterministic_sample_matches_manual_euler() -> None:
    cfg = _cfg("rl.clip_denoised_actions=False")
    policy = _policy(cfg)
    image = torch.zeros(1, 3, 8, 8)
    proprio = torch.randn(1, cfg.model.prop_dim)
    initial = torch.randn(1, cfg.model.chunk_size, cfg.model.action_dim)

    actual = policy.deterministic_sample(image, proprio, initial_noise=initial.clone())
    cond = policy.actor.encode_conditioning(image, proprio)
    expected = initial.clone()
    for tau_value in (0.0, 0.5):
        tau = torch.tensor([tau_value], dtype=expected.dtype)
        expected = expected + 0.5 * policy.policy.action_transformer(expected, tau, cond)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_real_transformer_dropout_does_not_change_stored_path_logprob() -> None:
    cfg = load_config(
        overrides=[
            "model.action_dim=2",
            "model.chunk_size=3",
            "model.prop_dim=5",
            "model.cond_dim=16",
            "model.d_model=16",
            "model.nhead=4",
            "model.nlayers=2",
            "model.vision_pretrained=False",
            "model.dropout=0.1",
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.flow_steps=2",
        ]
    )
    policy = ReinFlowActorCritic(cfg)
    policy.train()
    image = torch.rand(2, 3, 64, 64)
    proprio = torch.rand(2, 5)

    with torch.no_grad():
        output = policy.sample_path(image, proprio)
        evaluated = policy.evaluate_path(image, proprio, output.path)

    assert torch.allclose(output.log_prob, evaluated.log_prob, atol=1e-5)
    assert not policy.actor.training
