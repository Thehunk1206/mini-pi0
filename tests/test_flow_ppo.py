"""Behavior tests for macro-decision storage and ReinFlow PPO updates."""

from __future__ import annotations

# ruff: noqa: E402 - keep the optional torch dependency skippable at collection.

import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mini_pi0.config.io import load_config
from mini_pi0.rl.buffers import ReinFlowRolloutBuffer
from mini_pi0.rl.exceptions import ReinFlowNumericalError
from mini_pi0.rl.flow_policy import ReinFlowActorCritic
from mini_pi0.rl.flow_ppo import ReinFlowPPOProgress, ReinFlowPPOUpdater, _require_matching_old_policy


class _TinyActionTransformer(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, action_dim)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return 0.1 * actions + self.cond_proj(cond).unsqueeze(1) + tau[:, None, None] * 0.01


class _TinyFlowPolicy(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, prop_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(prop_dim, cond_dim)
        self.action_transformer = _TinyActionTransformer(action_dim, cond_dim)

    def _encode_conditioning(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        del image
        return self.encoder(proprio)


def _config(*, critic_warmup_updates: int = 0):
    return load_config(
        overrides=[
            "model.action_dim=2",
            "model.chunk_size=3",
            "model.prop_dim=5",
            "model.cond_dim=4",
            "model.d_model=8",
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='checkpoint'",
            "rl.action_normalization='dataset_stats'",
            "rl.checkpoint='unused.pt'",
            "rl.action_stats_path='unused.json'",
            "rl.use_reference_policy=True",
            "rl.flow_steps=2",
            "rl.execution_horizon=1",
            "rl.noise_std_min=0.04",
            "rl.noise_std_init=0.07",
            "rl.noise_std_max=0.10",
            "rl.freeze_vision_during_rl=False",
            "rl.num_envs=2",
            "rl.minibatch_size=4",
            "rl.epochs_per_update=1",
            "rl.target_kl=0.01",
            "rl.reference_w2_coef=0.1",
            "rl.reference_transition_kl_coef=0.1",
            f"rl.critic_warmup_updates={critic_warmup_updates}",
        ]
    )


def _policy(cfg) -> ReinFlowActorCritic:
    return ReinFlowActorCritic(
        cfg,
        policy=_TinyFlowPolicy(
            action_dim=cfg.model.action_dim,
            cond_dim=cfg.model.cond_dim,
            prop_dim=cfg.model.prop_dim,
        ),
    )


def _filled_buffer(policy: ReinFlowActorCritic, *, rewards: list[torch.Tensor]) -> ReinFlowRolloutBuffer:
    num_envs = int(rewards[0].shape[0])
    image = torch.zeros(num_envs, 3, 8, 8)
    proprio = torch.randn(num_envs, policy.cfg.model.prop_dim)
    buffer = ReinFlowRolloutBuffer(
        capacity=len(rewards),
        num_envs=num_envs,
        storage_device=torch.device("cpu"),
    )
    for reward in rewards:
        sample = policy.sample_path(image, proprio)
        buffer.add(
            image=image,
            proprio=proprio,
            path=sample.path,
            log_prob=sample.log_prob,
            reward=reward,
            value=sample.value,
            next_value=torch.zeros(num_envs),
            bootstrap_discount=torch.full((num_envs,), 0.99),
            trace_continue=torch.ones(num_envs),
            duration=torch.ones(num_envs, dtype=torch.long),
        )
    buffer.compute_returns_and_advantages(gae_lambda=0.95)
    return buffer


def _parameters_changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters(), strict=True))


def test_macro_gae_bootstraps_truncation_without_crossing_reset() -> None:
    buffer = ReinFlowRolloutBuffer(capacity=2, num_envs=1, storage_device=torch.device("cpu"))
    common = {
        "image": torch.zeros(1, 3, 4, 4),
        "proprio": torch.zeros(1, 2),
        "path": torch.zeros(1, 3, 2, 1),
        "log_prob": torch.zeros(1),
        "duration": torch.ones(1, dtype=torch.long),
    }
    buffer.add(
        **common,
        reward=torch.tensor([1.0]),
        value=torch.tensor([0.5]),
        next_value=torch.tensor([0.4]),
        bootstrap_discount=torch.tensor([0.9]),
        trace_continue=torch.tensor([1.0]),
    )
    buffer.add(
        **common,
        reward=torch.tensor([2.0]),
        value=torch.tensor([0.4]),
        next_value=torch.tensor([0.7]),
        bootstrap_discount=torch.tensor([0.9]),
        trace_continue=torch.tensor([0.0]),
    )

    buffer.compute_returns_and_advantages(gae_lambda=0.8)

    assert buffer.returns is not None
    assert torch.allclose(buffer.returns[:, 0], torch.tensor([2.9656, 2.63]), atol=1e-5)


def test_reinflow_update_changes_actor_and_keeps_reference_frozen() -> None:
    cfg = _config()
    policy = _policy(cfg)
    reference = copy.deepcopy(policy)
    updater = ReinFlowPPOUpdater(policy=policy, reference=reference, cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.4])])
    actor_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    reference_before = [parameter.detach().clone() for parameter in reference.parameters()]

    stats = updater.update(buffer, device=torch.device("cpu"), update_index=0, bounds=None)

    assert stats.actor_updated
    assert _parameters_changed(actor_before, policy.actor)
    assert all(
        torch.equal(old, new.detach())
        for old, new in zip(reference_before, reference.parameters(), strict=True)
    )
    assert stats.reference_w2 == pytest.approx(0.0, abs=1e-7)
    assert stats.transition_kl == pytest.approx(0.0, abs=1e-7)
    assert stats.approx_kl >= 0.0
    assert stats.rollout_log_prob_correction_max == pytest.approx(0.0, abs=1e-7)


def test_reinflow_update_rebases_rollout_log_probs_before_actor_step() -> None:
    cfg = _config()
    policy = _policy(cfg)
    updater = ReinFlowPPOUpdater(policy=policy, reference=copy.deepcopy(policy), cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.4])])
    buffer._storage["log_probs"].add_(2.0)  # noqa: SLF001 - emulate batch-shape precision drift.

    stats = updater.update(buffer, device=torch.device("cpu"), update_index=0, bounds=None)

    assert stats.rollout_log_prob_correction_max == pytest.approx(2.0, abs=1e-6)
    assert stats.approx_kl == pytest.approx(0.0, abs=1e-7)
    assert stats.clip_fraction == pytest.approx(0.0, abs=1e-7)
    assert stats.ratio_min == pytest.approx(1.0, abs=1e-7)
    assert stats.ratio_max == pytest.approx(1.0, abs=1e-7)


def test_reinflow_update_reports_rebase_and_optimizer_minibatches() -> None:
    cfg = _config()
    policy = _policy(cfg)
    updater = ReinFlowPPOUpdater(policy=policy, reference=copy.deepcopy(policy), cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.4])])
    events: list[ReinFlowPPOProgress] = []

    updater.update(
        buffer,
        device=torch.device("cpu"),
        update_index=0,
        bounds=None,
        progress=events.append,
    )

    assert [event.stage for event in events] == ["rebase", "optimize"]
    assert all(event.completed == event.total == 1 for event in events)
    assert events[-1].value_loss > 0.0
    assert events[-1].policy_loss is not None


def test_old_policy_invariant_rejects_preupdate_likelihood_drift() -> None:
    with pytest.raises(ReinFlowNumericalError, match="No actor update was applied"):
        _require_matching_old_policy(torch.tensor([0.0, 2e-5]))


def test_actor_step_is_skipped_when_current_policy_exceeds_target_kl() -> None:
    cfg = _config()
    policy = _policy(cfg)
    updater = ReinFlowPPOUpdater(policy=policy, reference=copy.deepcopy(policy), cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.4])])
    batch = next(buffer.minibatches(4, torch.device("cpu")))
    shifted = replace(batch, old_log_probs=batch.old_log_probs + 1.0)
    actor_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]

    metrics, step_applied = updater._step_actor(  # noqa: SLF001 - verify KL guard behavior.
        shifted,
        bounds=None,
        require_old_policy_match=False,
    )

    assert not step_applied
    assert metrics["approx_kl"] > cfg.rl.target_kl
    assert not _parameters_changed(actor_before, policy.actor)


def test_critic_warmup_does_not_change_actor() -> None:
    cfg = _config(critic_warmup_updates=1)
    policy = _policy(cfg)
    updater = ReinFlowPPOUpdater(policy=policy, reference=copy.deepcopy(policy), cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([1.0, 0.2]), torch.tensor([0.1, 1.4])])
    actor_before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in policy.critic.parameters()]

    stats = updater.update(buffer, device=torch.device("cpu"), update_index=0, bounds=None)

    assert not stats.actor_updated
    assert not _parameters_changed(actor_before, policy.actor)
    assert _parameters_changed(critic_before, policy.critic)


def test_update_rejects_non_finite_rollout() -> None:
    cfg = _config()
    policy = _policy(cfg)
    updater = ReinFlowPPOUpdater(policy=policy, reference=copy.deepcopy(policy), cfg=cfg)
    buffer = _filled_buffer(policy, rewards=[torch.tensor([float("nan"), 0.0])])

    with pytest.raises(ReinFlowNumericalError, match="value_loss"):
        updater.update(buffer, device=torch.device("cpu"), update_index=0, bounds=None)
