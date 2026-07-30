"""Round-trip tests for complete ReinFlow checkpoints."""

from __future__ import annotations

# ruff: noqa: E402 - keep torch optional at collection time.

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mini_pi0.config.io import load_config
from mini_pi0.rl.checkpointing import (
    load_reinflow_checkpoint,
    restore_reinflow_checkpoint,
    save_reinflow_checkpoint,
)
from mini_pi0.rl.flow_policy import ReinFlowActorCritic
from mini_pi0.rl.flow_ppo import ReinFlowPPOUpdater


class _TinyActionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(4, 2)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return 0.1 * actions + self.cond_proj(cond).unsqueeze(1) + 0.01 * tau[:, None, None]


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.action_transformer = _TinyActionTransformer()

    def _encode_conditioning(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        del image
        return self.encoder(proprio)


def _config():
    return load_config(
        overrides=[
            "model.action_dim=2",
            "model.chunk_size=3",
            "model.prop_dim=3",
            "model.cond_dim=4",
            "model.d_model=8",
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.action_normalization='env_bounds'",
            "rl.flow_steps=2",
            "rl.noise_std_min=0.04",
            "rl.noise_std_init=0.07",
            "rl.noise_std_max=0.10",
            "rl.freeze_vision_during_rl=False",
        ]
    )


def _policy(cfg) -> ReinFlowActorCritic:
    return ReinFlowActorCritic(cfg, policy=_TinyPolicy())


def _initialize_optimizers(policy: ReinFlowActorCritic, updater: ReinFlowPPOUpdater) -> None:
    updater.actor_optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in policy.actor.parameters()).backward()
    updater.actor_optimizer.step()
    updater.actor_scheduler.step()

    updater.critic_optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in policy.critic.parameters()).backward()
    updater.critic_optimizer.step()
    updater.critic_scheduler.step()


def _assert_optimizer_states_equal(
    first: torch.optim.Optimizer,
    second: torch.optim.Optimizer,
) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state["param_groups"] == second_state["param_groups"]
    assert first_state["state"].keys() == second_state["state"].keys()
    for key, values in first_state["state"].items():
        for name, value in values.items():
            other = second_state["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_checkpoint_resume_reproduces_next_action_and_optimizer_state(tmp_path: Path) -> None:
    cfg = _config()
    original = _policy(cfg)
    original_updater = ReinFlowPPOUpdater(policy=original, reference=None, cfg=cfg)
    _initialize_optimizers(original, original_updater)
    checkpoint_path = tmp_path / "latest.pt"
    image = torch.zeros(1, 3, 8, 8)
    proprio = torch.ones(1, 3)

    torch.manual_seed(1234)
    save_reinflow_checkpoint(
        checkpoint_path,
        cfg=cfg,
        policy=original,
        updater=original_updater,
        next_update=7,
        primitive_steps=321,
        best_eval_success=0.75,
        metrics={"update": 7},
        environment_manifest={"backend": "fake"},
    )
    expected_action = original.deterministic_sample(image, proprio)

    resumed = _policy(cfg)
    resumed_updater = ReinFlowPPOUpdater(policy=resumed, reference=None, cfg=cfg)
    payload = load_reinflow_checkpoint(checkpoint_path)
    state = restore_reinflow_checkpoint(payload, policy=resumed, updater=resumed_updater)
    actual_action = resumed.deterministic_sample(image, proprio)

    assert torch.equal(actual_action, expected_action)
    assert state.next_update == 7
    assert state.primitive_steps == 321
    assert state.best_eval_success == pytest.approx(0.75)
    _assert_optimizer_states_equal(original_updater.actor_optimizer, resumed_updater.actor_optimizer)
    _assert_optimizer_states_equal(original_updater.critic_optimizer, resumed_updater.critic_optimizer)
    assert not checkpoint_path.with_name(".latest.pt.tmp").exists()
