"""CLI runners for ReinFlow PPO and its isolated Gaussian baseline."""

from __future__ import annotations

import copy
import json
import random
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.rl.config import validate_rl_config
from mini_pi0.sim.batched import BatchedSimulatorAdapter, Observation, SerialBatchAdapter
from mini_pi0.sim.registry import make_sim_adapter


@dataclass
class _RolloutState:
    """Persistent environment and observation state across PPO updates."""

    observations: list[Observation]
    images: torch.Tensor
    proprio: torch.Tensor
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_successes: np.ndarray
    next_reset_seed: int
    primitive_steps: int = 0


@dataclass(frozen=True)
class _MacroStep:
    """Results from executing one action chunk without reconditioning."""

    observations: list[Observation]
    training_rewards: torch.Tensor
    episode_rewards: np.ndarray
    durations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    successes: np.ndarray
    clipped_actions: int
    action_count: int


def run_isaac_smoke(cfg: RootConfig) -> dict[str, Any]:
    """Run a minimal Isaac Lab reset/step smoke test."""

    sim_cfg = copy.deepcopy(cfg)
    sim_cfg.simulator.backend = "isaaclab"
    adapter = make_sim_adapter(sim_cfg)
    try:
        obs = adapter.reset(seed=int(cfg.experiment.seed))
        low, _high = adapter.action_spec()
        action = np.zeros_like(low, dtype=np.float32)
        step = adapter.step(action)
        summary = {
            "backend": adapter.backend_name,
            "task": sim_cfg.simulator.task,
            "action_dim": int(low.shape[0]),
            "obs_keys": sorted(obs.keys()),
            "reward": float(step.reward),
            "done": bool(step.done),
            "success": bool(adapter.check_success(step.info, step.obs)),
        }
    finally:
        adapter.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def run_rl_smoke(cfg: RootConfig) -> dict[str, Any]:
    """Run one simulator-agnostic RL reset/sample/step smoke test."""

    from mini_pi0.models.registry import load_checkpoint
    from mini_pi0.rl.checkpointing import load_reinflow_checkpoint, materialize_embedded_action_stats
    from mini_pi0.rl.flow_policy import ReinFlowActorCritic
    from mini_pi0.utils.device import resolve_device

    validate_rl_config(cfg, require_files=True)
    if _algorithm(cfg) != "reinflow_ppo":
        raise ValueError("rl-smoke currently supports rl.algorithm='reinflow_ppo'.")
    smoke_cfg = copy.deepcopy(cfg)
    resume_payload = load_reinflow_checkpoint(smoke_cfg.rl.resume_from) if smoke_cfg.rl.resume_from else None
    checkpoint = None if resume_payload is not None else _load_checkpoint_if_needed(smoke_cfg, load_checkpoint)
    source = resume_payload if resume_payload is not None else checkpoint
    if source is not None:
        _inject_model_config_from_checkpoint(smoke_cfg, source)
    with tempfile.TemporaryDirectory(prefix="mini-pi0-rl-smoke-") as temp_dir:
        if resume_payload is not None:
            stats_path = materialize_embedded_action_stats(resume_payload, Path(temp_dir) / "action_stats.json")
            if stats_path is not None:
                smoke_cfg.rl.action_stats_path = str(stats_path)
        device = resolve_device(smoke_cfg.rl.device)
        actor = ReinFlowActorCritic(smoke_cfg).to(device)
        if resume_payload is not None:
            actor.load_state_dict(resume_payload["actor_critic"], strict=True)
        elif checkpoint is not None:
            actor.policy.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
        adapter = SerialBatchAdapter([make_sim_adapter(smoke_cfg)])
        try:
            observations = adapter.reset([int(smoke_cfg.experiment.seed)])
            low, high = adapter.action_spec()
            processor = _make_obs_processor(smoke_cfg, device=str(device))
            processor.reset_batch_history(observations)
            img, prop = processor.obs_batch_to_tensors(observations)
            bounds = _policy_action_bounds(smoke_cfg, processor, low, high, device)
            with torch.no_grad():
                sample = actor.sample_path(img, prop, bounds=bounds)
                actions, _clip_mask = _policy_actions_to_env(
                    sample.action_chunk[:, :1],
                    cfg=smoke_cfg,
                    processor=processor,
                    low=low,
                    high=high,
                    device=device,
                )
            step = adapter.step(actions[:, 0], np.ones(1, dtype=bool))
            summary = {
                "algorithm": _algorithm(smoke_cfg),
                "backend": adapter.backend_name,
                "task": smoke_cfg.simulator.task,
                "init_mode": _init_mode(smoke_cfg),
                "action_dim": int(low.shape[0]),
                "flow_steps": int(smoke_cfg.rl.flow_steps),
                "path_shape": list(sample.path.shape),
                "reward": float(step.rewards[0]),
                "done": bool(step.terminated[0] or step.truncated[0]),
                "success": bool(step.successes[0]),
            }
        finally:
            adapter.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def run_rl_eval(cfg: RootConfig) -> dict[str, Any]:
    """Evaluate a BC or ReinFlow checkpoint with deterministic ODE actions."""

    from mini_pi0.models.registry import load_checkpoint
    from mini_pi0.rl.checkpointing import load_reinflow_checkpoint, materialize_embedded_action_stats
    from mini_pi0.rl.flow_policy import ReinFlowActorCritic
    from mini_pi0.utils.device import resolve_device

    validate_rl_config(cfg, require_files=True)
    eval_cfg = copy.deepcopy(cfg)
    resume_payload = load_reinflow_checkpoint(eval_cfg.rl.resume_from) if eval_cfg.rl.resume_from else None
    checkpoint = None if resume_payload is not None else _load_checkpoint_if_needed(eval_cfg, load_checkpoint)
    source = resume_payload if resume_payload is not None else checkpoint
    if source is not None:
        _inject_model_config_from_checkpoint(eval_cfg, source)
    with tempfile.TemporaryDirectory(prefix="mini-pi0-rl-eval-") as temp_dir:
        if resume_payload is not None:
            stats_path = materialize_embedded_action_stats(
                resume_payload,
                Path(temp_dir) / "action_stats.json",
            )
            if stats_path is not None:
                eval_cfg.rl.action_stats_path = str(stats_path)
        device = resolve_device(eval_cfg.rl.device)
        policy = ReinFlowActorCritic(eval_cfg).to(device)
        if resume_payload is not None:
            policy.load_state_dict(resume_payload["actor_critic"], strict=True)
        elif checkpoint is not None:
            policy.policy.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
        summary = _evaluate_actor(eval_cfg, policy, device)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def run_rl_train(cfg: RootConfig) -> dict[str, Any]:
    """Dispatch RL training by configured algorithm."""

    algorithm = _algorithm(cfg)
    if algorithm == "reinflow_ppo":
        return _run_reinflow_train(cfg)
    if algorithm == "gaussian_ppo_baseline":
        return _run_gaussian_ppo_baseline_train(cfg)
    validate_rl_config(cfg, require_files=False)
    raise ValueError(f"Unsupported rl.algorithm: {cfg.rl.algorithm}")


def _run_reinflow_train(cfg: RootConfig) -> dict[str, Any]:
    """Train or fine-tune a stochastic flow policy with ReinFlow PPO."""

    from mini_pi0.models.registry import load_checkpoint
    from mini_pi0.rl.checkpointing import (
        load_reinflow_checkpoint,
        materialize_embedded_action_stats,
        restore_reinflow_checkpoint,
    )
    from mini_pi0.rl.flow_policy import ReinFlowActorCritic
    from mini_pi0.rl.flow_ppo import ReinFlowPPOUpdater
    from mini_pi0.train.data import seed_everything
    from mini_pi0.utils.device import resolve_device
    from mini_pi0.utils.runs import create_run_dir

    validate_rl_config(cfg, require_files=True)
    seed_everything(int(cfg.experiment.seed))
    resume_payload = load_reinflow_checkpoint(cfg.rl.resume_from) if cfg.rl.resume_from else None
    ckpt = None if resume_payload is not None else _load_checkpoint_if_needed(cfg, load_checkpoint)
    if resume_payload is not None:
        _inject_model_config_from_checkpoint(cfg, resume_payload)
    if ckpt is not None:
        _inject_model_config_from_checkpoint(cfg, ckpt)
    run_dir = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-reinflow")
    if resume_payload is not None:
        stats_path = materialize_embedded_action_stats(
            resume_payload,
            run_dir / "artifacts" / "action_stats.json",
        )
        if stats_path is not None:
            cfg.rl.action_stats_path = str(stats_path)
    dump_config(run_dir / "config_resolved.yaml", cfg)

    device = resolve_device(cfg.rl.device)
    actor = ReinFlowActorCritic(cfg).to(device)
    if ckpt is not None:
        model_state = _checkpoint_model_state(ckpt)
        actor.policy.load_state_dict(model_state, strict=True)
    reference = copy.deepcopy(actor).to(device) if bool(cfg.rl.use_reference_policy) else None
    updater = ReinFlowPPOUpdater(policy=actor, reference=reference, cfg=cfg)
    start_update = 0
    primitive_steps = 0
    best_eval_success = float("-inf")
    latest_metrics: dict[str, object] = {}
    if resume_payload is not None:
        restored = restore_reinflow_checkpoint(resume_payload, policy=actor, updater=updater)
        start_update = restored.next_update
        primitive_steps = restored.primitive_steps
        best_eval_success = restored.best_eval_success
        latest_metrics = restored.latest_metrics

    adapter = _make_batched_adapter(cfg)
    try:
        summary = _run_reinflow_loop(
            cfg=cfg,
            run_dir=run_dir,
            actor=actor,
            updater=updater,
            adapter=adapter,
            device=device,
            start_update=start_update,
            primitive_steps=primitive_steps,
            best_eval_success=best_eval_success,
            latest_metrics=latest_metrics,
        )
    finally:
        adapter.close()
    return summary


def _run_gaussian_ppo_baseline_train(cfg: RootConfig) -> dict[str, Any]:
    """Run the older Gaussian surrogate PPO baseline."""

    from mini_pi0.models.registry import load_checkpoint
    from mini_pi0.rl.ppo import FlowMatchingActorCritic, PPOUpdater
    from mini_pi0.train.data import seed_everything
    from mini_pi0.utils.device import resolve_device
    from mini_pi0.utils.runs import create_run_dir

    validate_rl_config(cfg, require_files=True)
    seed_everything(int(cfg.experiment.seed))
    run_dir = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-gaussian-ppo")
    dump_config(run_dir / "config_resolved.yaml", cfg)

    ckpt = load_checkpoint(cfg.rl.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("rl.checkpoint must load a dict checkpoint payload.")
    _inject_model_config_from_checkpoint(cfg, ckpt)

    device = resolve_device(cfg.rl.device)
    actor = FlowMatchingActorCritic(cfg, log_std_init=float(cfg.rl.log_std_init)).to(device)
    reference = FlowMatchingActorCritic(cfg, log_std_init=float(cfg.rl.log_std_init)).to(device)
    model_state = _checkpoint_model_state(ckpt)
    actor.policy.load_state_dict(model_state, strict=True)
    reference.policy.load_state_dict(model_state, strict=True)
    reference.load_state_dict(actor.state_dict(), strict=True)
    updater = PPOUpdater(actor=actor, reference=reference, cfg=cfg)

    adapters = _make_adapters(cfg)
    try:
        summary = _run_gaussian_ppo_loop(
            cfg=cfg,
            run_dir=run_dir,
            actor=actor,
            updater=updater,
            adapters=adapters,
            device=device,
        )
    finally:
        for adapter in adapters:
            adapter.close()
    return summary


def _run_reinflow_loop(
    *,
    cfg: RootConfig,
    run_dir: Path,
    actor: Any,
    updater: Any,
    adapter: BatchedSimulatorAdapter,
    device: torch.device,
    start_update: int = 0,
    primitive_steps: int = 0,
    best_eval_success: float = float("-inf"),
    latest_metrics: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Collect macro transitions and optimize joint denoising-path likelihoods."""

    from mini_pi0.rl.buffers import ReinFlowRolloutBuffer
    from mini_pi0.rl.checkpointing import save_reinflow_checkpoint
    from mini_pi0.rl.rewards import make_reward_strategy
    from mini_pi0.utils.runs import append_jsonl

    low, high = adapter.action_spec()
    _validate_action_dimensions(cfg, low, high)
    processor = _make_obs_processor(cfg, device=str(device))
    seeds = [int(cfg.experiment.seed) + index for index in range(adapter.num_envs)]
    observations = adapter.reset(seeds)
    processor.reset_batch_history(observations)
    images, proprio = processor.obs_batch_to_tensors(observations)
    state = _RolloutState(
        observations=observations,
        images=images,
        proprio=proprio,
        episode_returns=np.zeros(adapter.num_envs, dtype=np.float64),
        episode_lengths=np.zeros(adapter.num_envs, dtype=np.int64),
        episode_successes=np.zeros(adapter.num_envs, dtype=bool),
        next_reset_seed=int(cfg.experiment.seed) + adapter.num_envs,
        primitive_steps=int(primitive_steps),
    )
    reward_strategy = make_reward_strategy(
        cfg.rl.reward_strategy,
        grasp_weight=float(cfg.rl.peg_potential_grasp_weight),
        alignment_weight=float(cfg.rl.peg_potential_alignment_weight),
        insertion_weight=float(cfg.rl.peg_potential_insertion_weight),
    )
    policy_bounds = _policy_action_bounds(cfg, processor, low, high, device)
    storage_device = _rollout_storage_device(cfg, device)
    latest_summary: dict[str, Any] = dict(latest_metrics or {})
    environment_manifest: dict[str, object] = {
        "backend": adapter.backend_name,
        "task": str(cfg.simulator.task),
        "num_envs": adapter.num_envs,
        "action_low": low.tolist(),
        "action_high": high.tolist(),
    }

    for update_index in range(int(start_update), int(cfg.rl.total_updates)):
        actor.actor.kernel.noise.set_training_progress(_update_progress(update_index, int(cfg.rl.total_updates)))
        buffer = ReinFlowRolloutBuffer(
            capacity=int(cfg.rl.rollout_decisions_per_update),
            num_envs=adapter.num_envs,
            storage_device=storage_device,
        )
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_successes: list[bool] = []
        clipped_actions = 0
        action_count = 0

        for _decision in range(int(cfg.rl.rollout_decisions_per_update)):
            actor.actor.likelihood_mode()
            with torch.no_grad(), _autocast(cfg, device):
                sample = actor.sample_path(state.images, state.proprio, bounds=policy_bounds)
            macro = _execute_macro_action(
                cfg=cfg,
                adapter=adapter,
                action_chunk=sample.action_chunk,
                observations=state.observations,
                processor=processor,
                reward_strategy=reward_strategy,
                low=low,
                high=high,
                device=device,
            )
            next_images, next_proprio = processor.obs_batch_to_tensors(macro.observations)
            with torch.no_grad(), _autocast(cfg, device):
                next_values = actor.value(next_images, next_proprio).float()
            bootstrap = torch.pow(
                torch.full_like(macro.durations, float(cfg.rl.gamma), dtype=torch.float32),
                macro.durations.float(),
            )
            bootstrap = torch.where(macro.terminated, torch.zeros_like(bootstrap), bootstrap)
            trace_continue = ~(macro.terminated | macro.truncated)
            buffer.add(
                image=state.images,
                proprio=state.proprio,
                path=sample.path,
                log_prob=sample.log_prob,
                reward=macro.training_rewards,
                value=sample.value.float(),
                next_value=next_values,
                bootstrap_discount=bootstrap,
                trace_continue=trace_continue.float(),
                duration=macro.durations,
            )

            state.observations = macro.observations
            state.images = next_images
            state.proprio = next_proprio
            state.episode_returns += macro.episode_rewards
            state.episode_lengths += macro.durations.cpu().numpy()
            state.episode_successes |= macro.successes
            state.primitive_steps += int(macro.durations.sum().item())
            clipped_actions += macro.clipped_actions
            action_count += macro.action_count
            completed = np.flatnonzero((macro.terminated | macro.truncated).cpu().numpy())
            for env_index in completed:
                completed_returns.append(float(state.episode_returns[env_index]))
                completed_lengths.append(int(state.episode_lengths[env_index]))
                completed_successes.append(bool(state.episode_successes[env_index]))
                state.episode_returns[env_index] = 0.0
                state.episode_lengths[env_index] = 0
                state.episode_successes[env_index] = False
            _reset_completed_environments(state, adapter, processor, completed)

        buffer.compute_returns_and_advantages(gae_lambda=float(cfg.rl.gae_lambda))
        stats = updater.update(
            buffer,
            device=device,
            update_index=update_index,
            bounds=policy_bounds,
        )
        reward_mean = float(np.mean(completed_returns)) if completed_returns else 0.0
        success_rate = float(np.mean(completed_successes)) if completed_successes else 0.0
        row = {
            "update": update_index + 1,
            "primitive_steps": state.primitive_steps,
            "algorithm": _algorithm(cfg),
            "init_mode": _init_mode(cfg),
            "backend": adapter.backend_name,
            "reward_mean": reward_mean,
            "episode_length_mean": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "success_rate": success_rate,
            "completed_episodes": len(completed_returns),
            "action_clip_fraction": clipped_actions / max(1, action_count),
            "noise_std_upper_bound": float(actor.actor.kernel.noise.current_std_max.item()),
            **vars(stats),
        }
        eval_interval = int(cfg.rl.eval_every_updates)
        if eval_interval > 0 and (update_index + 1) % eval_interval == 0:
            with _preserve_rng_state(device):
                evaluation = _evaluate_actor(cfg, actor, device)
            row.update(
                {
                    "eval_success_rate": evaluation["success_rate"],
                    "eval_return_mean": evaluation["return_mean"],
                    "eval_episode_length_mean": evaluation["episode_length_mean"],
                }
            )
            eval_success = float(evaluation["success_rate"])
            if eval_success > best_eval_success:
                best_eval_success = eval_success
                save_reinflow_checkpoint(
                    run_dir / "checkpoints" / "best_rl.pt",
                    cfg=cfg,
                    policy=actor,
                    updater=updater,
                    next_update=update_index + 1,
                    primitive_steps=state.primitive_steps,
                    best_eval_success=best_eval_success,
                    metrics=row,
                    environment_manifest=environment_manifest,
                )
        append_jsonl(run_dir / "metrics" / "rl_metrics.jsonl", row)
        latest_summary = row
        save_reinflow_checkpoint(
            run_dir / "checkpoints" / "latest_rl.pt",
            cfg=cfg,
            policy=actor,
            updater=updater,
            next_update=update_index + 1,
            primitive_steps=state.primitive_steps,
            best_eval_success=best_eval_success,
            metrics=row,
            environment_manifest=environment_manifest,
        )
        print(
            f"[reinflow] update={update_index + 1:04d}/{cfg.rl.total_updates:04d} "
            f"reward={reward_mean:.3f} success={success_rate:.3f} "
            f"policy={stats.policy_loss:.4f} value={stats.value_loss:.4f} "
            f"kl={stats.approx_kl:.5f}",
            flush=True,
        )

    return _write_summary(
        run_dir,
        best_eval_success,
        latest_summary,
        metric_name="best_eval_success",
    )


def _evaluate_actor(cfg: RootConfig, actor: Any, device: torch.device) -> dict[str, Any]:
    """Run fixed-seed deterministic ODE evaluation episodes."""

    from mini_pi0.rl.rewards import NativeReward

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.rl.num_envs = 1
    adapter = _make_batched_adapter(eval_cfg)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    try:
        low, high = adapter.action_spec()
        _validate_action_dimensions(eval_cfg, low, high)
        processor = _make_obs_processor(eval_cfg, device=str(device))
        bounds = _policy_action_bounds(eval_cfg, processor, low, high, device)
        for episode_index in range(int(eval_cfg.rl.eval_episodes)):
            seed = int(eval_cfg.rl.eval_seed_start) + episode_index
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            observations = adapter.reset([seed])
            processor.reset_batch_history(observations)
            images, proprio = processor.obs_batch_to_tensors(observations)
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            while episode_length < int(eval_cfg.simulator.horizon):
                actor.actor.likelihood_mode()
                with torch.no_grad(), _autocast(eval_cfg, device):
                    action_chunk = actor.deterministic_sample(images, proprio, bounds=bounds)
                macro = _execute_macro_action(
                    cfg=eval_cfg,
                    adapter=adapter,
                    action_chunk=action_chunk,
                    observations=observations,
                    processor=processor,
                    reward_strategy=NativeReward(),
                    low=low,
                    high=high,
                    device=device,
                )
                observations = macro.observations
                episode_return += float(macro.episode_rewards[0])
                episode_length += int(macro.durations[0])
                episode_success = bool(episode_success or macro.successes[0])
                if bool(macro.terminated[0] or macro.truncated[0]):
                    break
                images, proprio = processor.obs_batch_to_tensors(observations)
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)
    finally:
        adapter.close()
    return {
        "episodes": len(returns),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
    }


def _execute_macro_action(
    *,
    cfg: RootConfig,
    adapter: BatchedSimulatorAdapter,
    action_chunk: torch.Tensor,
    observations: list[Observation],
    processor: Any,
    reward_strategy: Any,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
) -> _MacroStep:
    """Execute a fixed policy chunk until its horizon or an episode boundary."""

    env_actions, clip_mask = _policy_actions_to_env(
        action_chunk[:, : int(cfg.rl.execution_horizon)],
        cfg=cfg,
        processor=processor,
        low=low,
        high=high,
        device=device,
    )
    num_envs = adapter.num_envs
    active = np.ones(num_envs, dtype=bool)
    final_observations = list(observations)
    discounted_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    durations = np.zeros(num_envs, dtype=np.int64)
    terminated = np.zeros(num_envs, dtype=bool)
    truncated = np.zeros(num_envs, dtype=bool)
    successes = np.zeros(num_envs, dtype=bool)
    clipped_actions = 0
    action_count = 0

    for primitive_index in range(env_actions.shape[1]):
        if not active.any():
            break
        stepped = active.copy()
        output = adapter.step(env_actions[:, primitive_index], stepped)
        final_observations = output.observations
        discounts = np.power(float(cfg.rl.gamma), durations, dtype=np.float64)
        discounted_rewards[stepped] += (discounts[stepped] * output.rewards[stepped]).astype(np.float32)
        episode_rewards[stepped] += output.rewards[stepped]
        durations[stepped] += 1
        successes |= output.successes & stepped
        terminated |= output.terminated & stepped
        truncated |= output.truncated & stepped
        if bool(cfg.rl.terminate_on_success):
            terminated |= output.successes & stepped
        active &= ~(terminated | truncated)
        clipped_actions += int(clip_mask[stepped, primitive_index].sum())
        action_count += int(clip_mask[stepped, primitive_index].size)

    training_rewards = [
        reward_strategy.macro_reward(
            float(discounted_rewards[index]),
            observations[index],
            final_observations[index],
            int(durations[index]),
            float(cfg.rl.gamma),
        )
        for index in range(num_envs)
    ]
    return _MacroStep(
        observations=final_observations,
        training_rewards=torch.tensor(training_rewards, dtype=torch.float32),
        episode_rewards=episode_rewards,
        durations=torch.from_numpy(durations),
        terminated=torch.from_numpy(terminated),
        truncated=torch.from_numpy(truncated),
        successes=successes,
        clipped_actions=clipped_actions,
        action_count=action_count,
    )


def _reset_completed_environments(
    state: _RolloutState,
    adapter: BatchedSimulatorAdapter,
    processor: Any,
    completed: np.ndarray,
) -> None:
    """Reset completed environments without disturbing other observation histories."""

    indices = [int(index) for index in completed]
    if not indices:
        return
    seeds = list(range(state.next_reset_seed, state.next_reset_seed + len(indices)))
    state.next_reset_seed += len(indices)
    reset_observations = adapter.reset_at(indices, seeds)
    for index, observation in reset_observations.items():
        state.observations[index] = observation
    processor.reset_batch_history_at(reset_observations)
    ordered = [reset_observations[index] for index in indices]
    reset_images, reset_proprio = processor.obs_batch_to_tensors(ordered, env_indices=indices)
    state.images[indices] = reset_images
    state.proprio[indices] = reset_proprio


def _run_gaussian_ppo_loop(
    *,
    cfg: RootConfig,
    run_dir: Path,
    actor: Any,
    updater: Any,
    adapters: list[Any],
    device: Any,
) -> dict[str, Any]:
    """Collect rollouts and optimize the Gaussian PPO baseline."""

    import torch

    from mini_pi0.rl.buffers import PPORolloutBuffer
    from mini_pi0.rl.rewards import shaped_reward
    from mini_pi0.utils.runs import append_jsonl

    obs_batch = [adapter.reset(seed=int(cfg.experiment.seed) + idx) for idx, adapter in enumerate(adapters)]
    low, high = adapters[0].action_spec()
    processor = _make_obs_processor(cfg, device=str(device))
    processor.reset_batch_history(obs_batch)
    best_reward = float("-inf")
    latest_summary: dict[str, Any] = {}

    for update_idx in range(int(cfg.rl.total_updates)):
        buffer = PPORolloutBuffer()
        episode_rewards = np.zeros((len(adapters),), dtype=np.float32)
        completed_rewards: list[float] = []
        successes = 0
        completions = 0

        for _step_idx in range(int(cfg.rl.rollout_steps)):
            img, prop = processor.obs_batch_to_tensors(obs_batch)
            with torch.no_grad():
                actions_norm, log_prob, value = actor.act(img, prop)
                _dist_ref, _value_ref = updater.reference(img, prop)
                ref_log_prob = _dist_ref.log_prob(actions_norm).sum(dim=-1)
                env_actions = _normalized_actions_to_env(
                    actions_norm,
                    cfg=cfg,
                    processor=processor,
                    low=low,
                    high=high,
                    device=device,
                )

            next_obs: list[dict[str, np.ndarray]] = []
            rewards: list[float] = []
            dones: list[float] = []
            for env_idx, (adapter, action) in enumerate(zip(adapters, env_actions, strict=True)):
                step = adapter.step(action.detach().cpu().numpy().astype(np.float32))
                reward = shaped_reward(step.reward, step.info)
                episode_rewards[env_idx] += float(reward)
                done = bool(step.done)
                if adapter.check_success(step.info, step.obs):
                    successes += 1
                    done = True
                if done:
                    completed_rewards.append(float(episode_rewards[env_idx]))
                    episode_rewards[env_idx] = 0.0
                    completions += 1
                    next_obs.append(adapter.reset(seed=int(cfg.experiment.seed) + update_idx + env_idx + 1))
                else:
                    next_obs.append(step.obs)
                rewards.append(float(reward))
                dones.append(float(done))

            buffer.add(
                image=img.detach().cpu(),
                proprio=prop.detach().cpu(),
                action=actions_norm.detach().cpu(),
                log_prob=log_prob.detach().cpu(),
                reward=torch.tensor(rewards, dtype=torch.float32),
                done=torch.tensor(dones, dtype=torch.float32),
                value=value.detach().cpu(),
                ref_log_prob=ref_log_prob.detach().cpu(),
            )
            obs_batch = next_obs
            if any(dones):
                processor.reset_batch_history(obs_batch)

        img_last, prop_last = processor.obs_batch_to_tensors(obs_batch)
        with torch.no_grad():
            _dist, last_value = actor(img_last, prop_last)
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=float(cfg.rl.gamma),
            gae_lambda=float(cfg.rl.gae_lambda),
        )
        stats = updater.update(buffer, device=device)
        reward_mean = float(np.mean(completed_rewards)) if completed_rewards else 0.0
        success_rate = float(successes / max(1, completions))
        row = {
            "update": int(update_idx + 1),
            "algorithm": _algorithm(cfg),
            "init_mode": _init_mode(cfg),
            "backend": str(cfg.simulator.backend),
            "reward_mean": reward_mean,
            "success_rate": success_rate,
            "completed_episodes": int(completions),
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "reference_kl": stats.reference_kl,
            "total_loss": stats.total_loss,
        }
        append_jsonl(run_dir / "metrics" / "rl_metrics.jsonl", row)
        latest_summary = row
        if reward_mean > best_reward:
            best_reward = reward_mean
            _save_rl_checkpoint(run_dir / "checkpoints" / "best_rl.pt", cfg=cfg, actor=actor, updater=updater, row=row)
        _save_rl_checkpoint(run_dir / "checkpoints" / "latest_rl.pt", cfg=cfg, actor=actor, updater=updater, row=row)
        print(
            f"[gaussian-ppo] update={update_idx + 1:04d}/{cfg.rl.total_updates:04d} "
            f"backend={cfg.simulator.backend} reward_mean={reward_mean:.3f} "
            f"success_rate={success_rate:.3f} policy_loss={stats.policy_loss:.4f} "
            f"value_loss={stats.value_loss:.4f} kl={stats.approx_kl:.5f}",
            flush=True,
        )

    return _write_summary(run_dir, best_reward, latest_summary)


def _make_adapters(cfg: RootConfig) -> list[Any]:
    """Create one adapter per requested RL environment."""

    adapters = []
    for idx in range(int(cfg.rl.num_envs)):
        sim_cfg = copy.deepcopy(cfg)
        sim_cfg.experiment.seed = int(cfg.experiment.seed) + idx
        adapters.append(make_sim_adapter(sim_cfg))
    return adapters


def _make_batched_adapter(cfg: RootConfig) -> BatchedSimulatorAdapter:
    """Prefer native vectors while retaining serial adapter compatibility."""

    if str(cfg.simulator.backend).strip().lower() == "maniskill3" and int(cfg.rl.num_envs) > 1:
        from mini_pi0.sim.maniskill3_batched import ManiSkill3BatchedAdapter

        return ManiSkill3BatchedAdapter(cfg)
    if str(cfg.simulator.backend).strip().lower() == "isaaclab" and int(cfg.rl.num_envs) > 1:
        from mini_pi0.sim.isaaclab_batched import IsaacLabBatchedAdapter

        return IsaacLabBatchedAdapter(cfg)
    return SerialBatchAdapter(_make_adapters(cfg))


def _make_obs_processor(cfg: RootConfig, *, device: str):
    """Create an observation tensorizer with optional action stats."""

    from mini_pi0.dataset.obs_processor import ObsProcessor

    stats_path = cfg.rl.action_stats_path if _action_normalization(cfg) == "dataset_stats" else None
    return ObsProcessor(
        action_stats_path=stats_path,
        image_key=cfg.robot.image_key,
        image_keys=effective_image_keys(cfg.robot),
        proprio_keys=effective_state_keys(cfg.robot),
        device=device,
        obs_horizon=int(getattr(cfg.model, "obs_horizon", 1)),
        preserve_camera_dim=str(getattr(cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention",
    )


def _normalized_actions_to_env(
    actions: Any,
    *,
    cfg: RootConfig,
    processor: Any,
    low: np.ndarray,
    high: np.ndarray,
    device: Any,
):
    """Convert normalized policy actions to clipped simulator actions."""

    import torch

    low_t = torch.as_tensor(np.asarray(low, dtype=np.float32), dtype=torch.float32, device=device).reshape(1, -1)
    high_t = torch.as_tensor(np.asarray(high, dtype=np.float32), dtype=torch.float32, device=device).reshape(1, -1)
    if actions.shape[-1] != low_t.shape[-1]:
        raise ValueError(
            f"Policy action_dim={actions.shape[-1]} does not match simulator action_dim={low_t.shape[-1]}. "
            "Update model.action_dim/robot.action_dim or the simulator control mode."
        )
    if _action_normalization(cfg) == "dataset_stats":
        env_actions = processor.denormalize(actions)
    else:
        center = 0.5 * (high_t + low_t)
        radius = 0.5 * (high_t - low_t)
        env_actions = center + torch.tanh(actions) * radius
    return torch.clamp(env_actions, low_t, high_t)


def _policy_action_bounds(
    cfg: RootConfig,
    processor: Any,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map simulator action limits into the actor's normalized action space."""

    lower = torch.as_tensor(low, dtype=torch.float32, device=device)
    upper = torch.as_tensor(high, dtype=torch.float32, device=device)
    if _action_normalization(cfg) == "dataset_stats":
        if processor.action_mean is None or processor.action_std is None:
            raise RuntimeError("Dataset action statistics are required for checkpoint fine-tuning.")
        lower = (lower - processor.action_mean) / processor.action_std
        upper = (upper - processor.action_mean) / processor.action_std
    else:
        limit = float(cfg.rl.scratch_policy_clip)
        lower = torch.full_like(lower, -limit)
        upper = torch.full_like(upper, limit)
    return lower, upper


def _policy_actions_to_env(
    actions: torch.Tensor,
    *,
    cfg: RootConfig,
    processor: Any,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert policy-space action chunks and report elementwise clipping."""

    lower = torch.as_tensor(low, dtype=torch.float32, device=device)
    upper = torch.as_tensor(high, dtype=torch.float32, device=device)
    if actions.shape[-1] != lower.numel():
        raise ValueError(
            f"Policy action_dim={actions.shape[-1]} does not match simulator action_dim={lower.numel()}."
        )
    if _action_normalization(cfg) == "dataset_stats":
        raw_actions = processor.denormalize(actions.float())
    else:
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)
        raw_actions = center + torch.tanh(actions.float()) * radius
    clip_mask = (raw_actions < lower) | (raw_actions > upper)
    clipped = torch.maximum(torch.minimum(raw_actions, upper), lower)
    return (
        clipped.detach().cpu().numpy().astype(np.float32),
        clip_mask.detach().cpu().numpy(),
    )


def _validate_action_dimensions(cfg: RootConfig, low: np.ndarray, high: np.ndarray) -> None:
    """Fail before rollout when policy, robot, and simulator actions disagree."""

    action_dim = int(np.asarray(low).size)
    if np.asarray(high).size != action_dim:
        raise ValueError("Simulator action lower and upper bounds have different dimensions.")
    configured = {"model.action_dim": int(cfg.model.action_dim), "robot.action_dim": int(cfg.robot.action_dim)}
    mismatched = [f"{name}={value}" for name, value in configured.items() if value != action_dim]
    if mismatched:
        details = ", ".join(mismatched)
        raise ValueError(f"Simulator action_dim={action_dim} does not match {details}.")


def _rollout_storage_device(cfg: RootConfig, policy_device: torch.device) -> torch.device:
    """Resolve rollout storage without moving serial simulator data to CUDA."""

    requested = str(cfg.rl.rollout_storage_device).strip().lower()
    if requested == "auto" or requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("rl.rollout_storage_device='cuda' requires CUDA.")
    return policy_device if policy_device.type == "cuda" else torch.device("cuda")


def _autocast(cfg: RootConfig, device: torch.device):
    """Use bf16 only for CUDA neural-network forward passes."""

    enabled = str(cfg.rl.dtype).strip().lower() == "bf16" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@contextmanager
def _preserve_rng_state(device: torch.device):
    """Keep periodic evaluation from changing the training random stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _update_progress(update_index: int, total_updates: int) -> float:
    """Return normalized training progress including both endpoints."""

    if total_updates <= 1:
        return 1.0
    return float(update_index) / float(total_updates - 1)


def _load_checkpoint_if_needed(cfg: RootConfig, loader: Any) -> dict[str, Any] | None:
    """Load checkpoint payload when required by config."""

    if cfg.rl.resume_from or (_init_mode(cfg) != "checkpoint" and not bool(cfg.rl.use_reference_policy)):
        return None
    ckpt = loader(cfg.rl.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("rl.checkpoint must load a dict checkpoint payload.")
    return ckpt


def _inject_model_config_from_checkpoint(cfg: RootConfig, ckpt: dict[str, Any]) -> None:
    """Backfill model config fields from checkpoint metadata."""

    model_name = ckpt.get("model_name")
    if isinstance(model_name, str) and model_name:
        cfg.model.name = model_name
    model_cfg = ckpt.get("model_config")
    resolved = ckpt.get("resolved_config")
    if model_cfg is None and isinstance(resolved, dict):
        model_cfg = resolved.get("model")
    if isinstance(model_cfg, dict):
        for key, value in model_cfg.items():
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)


def _checkpoint_model_state(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Return model state dict from checkpoint payload."""

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def _save_rl_checkpoint(
    path: Path,
    *,
    cfg: RootConfig,
    actor: Any,
    updater: Any,
    row: dict[str, Any],
) -> None:
    """Save RL checkpoint with FM policy and optimizer state."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": actor.policy.state_dict(),
            "actor_critic": actor.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "model_name": cfg.model.name,
            "model_config": {
                "action_dim": cfg.model.action_dim,
                "prop_dim": cfg.model.prop_dim,
                "chunk_size": cfg.model.chunk_size,
                "cond_dim": cfg.model.cond_dim,
                "d_model": cfg.model.d_model,
                "nhead": cfg.model.nhead,
                "nlayers": cfg.model.nlayers,
                "action_backbone": cfg.model.action_backbone,
                "conditioning_mode": cfg.model.conditioning_mode,
                "obs_horizon": cfg.model.obs_horizon,
                "vision_backbone": cfg.model.vision_backbone,
                "vision_model_name": cfg.model.vision_model_name,
                "vision_pretrained": cfg.model.vision_pretrained,
                "action_cnn_kernel_size": cfg.model.action_cnn_kernel_size,
                "freeze_vision_backbone": cfg.model.freeze_vision_backbone,
                "dropout": cfg.model.dropout,
            },
            "rl_algorithm": cfg.rl.algorithm,
            "rl_init_mode": cfg.rl.init_mode,
            "rl_config": cfg.rl,
            "metrics": row,
        },
        path,
    )


def _write_summary(
    run_dir: Path,
    best_metric: float,
    latest_summary: dict[str, Any],
    *,
    metric_name: str = "best_reward_mean",
) -> dict[str, Any]:
    """Write and return the final RL run summary."""

    best_path = run_dir / "checkpoints" / "best_rl.pt"
    summary = {
        "run_dir": str(run_dir),
        metric_name: float(best_metric),
        "latest": latest_summary,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "latest_checkpoint": str(run_dir / "checkpoints" / "latest_rl.pt"),
    }
    with (run_dir / "metrics" / "rl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _algorithm(cfg: RootConfig) -> str:
    """Return normalized RL algorithm key."""

    return str(cfg.rl.algorithm or "").strip().lower()


def _init_mode(cfg: RootConfig) -> str:
    """Return normalized RL initialization mode."""

    return str(cfg.rl.init_mode or "").strip().lower()


def _action_normalization(cfg: RootConfig) -> str:
    """Return normalized action normalization key."""

    if _algorithm(cfg) == "gaussian_ppo_baseline":
        return "dataset_stats"
    return str(cfg.rl.action_normalization or "").strip().lower()
