from __future__ import annotations

"""CLI runners for Isaac smoke tests and PPO warm-start fine-tuning."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.rl.config import validate_rl_config
from mini_pi0.sim.registry import make_sim_adapter


def run_isaac_smoke(cfg: RootConfig) -> dict[str, Any]:
    """Run a minimal Isaac Lab reset/step smoke test.

    Args:
        cfg: Resolved root configuration.

    Returns:
        JSON-serializable smoke-test summary.
    """

    sim_cfg = copy.deepcopy(cfg)
    sim_cfg.simulator.backend = "isaaclab"
    adapter = make_sim_adapter(sim_cfg)
    try:
        obs = adapter.reset(seed=int(cfg.experiment.seed))
        low, high = adapter.action_spec()
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


def run_rl_train(cfg: RootConfig) -> dict[str, Any]:
    """Fine-tune an FM checkpoint with PPO in simulation."""

    from mini_pi0.models.registry import load_checkpoint
    from mini_pi0.rl.ppo import FlowMatchingActorCritic, PPOUpdater
    from mini_pi0.train.data import seed_everything
    from mini_pi0.utils.device import resolve_device
    from mini_pi0.utils.runs import create_run_dir

    validate_rl_config(cfg, require_files=True)
    seed_everything(int(cfg.experiment.seed))
    run_dir = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-rl")
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
        summary = _run_ppo_loop(
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


def _run_ppo_loop(
    *,
    cfg: RootConfig,
    run_dir: Path,
    actor: Any,
    updater: Any,
    adapters: list[Any],
    device: Any,
) -> dict[str, Any]:
    """Collect rollouts and optimize PPO."""

    import torch

    from mini_pi0.dataset.obs_processor import ObsProcessor
    from mini_pi0.rl.buffers import PPORolloutBuffer
    from mini_pi0.rl.rewards import shaped_reward
    from mini_pi0.utils.runs import append_jsonl

    obs_batch = [adapter.reset(seed=int(cfg.experiment.seed) + idx) for idx, adapter in enumerate(adapters)]
    processor = ObsProcessor(
        action_stats_path=cfg.rl.action_stats_path,
        image_key=cfg.robot.image_key,
        image_keys=effective_image_keys(cfg.robot),
        proprio_keys=effective_state_keys(cfg.robot),
        device=str(device),
        obs_horizon=int(getattr(cfg.model, "obs_horizon", 1)),
        preserve_camera_dim=str(getattr(cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention",
    )
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
                raw_actions = processor.denormalize(actions_norm).detach().cpu().numpy()

            next_obs: list[dict[str, np.ndarray]] = []
            rewards: list[float] = []
            dones: list[float] = []
            for env_idx, (adapter, action) in enumerate(zip(adapters, raw_actions, strict=True)):
                step = adapter.step(np.asarray(action, dtype=np.float32))
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
            f"[rl] update={update_idx + 1:04d}/{cfg.rl.total_updates:04d} "
            f"reward_mean={reward_mean:.3f} success_rate={success_rate:.3f} "
            f"policy_loss={stats.policy_loss:.4f} value_loss={stats.value_loss:.4f} "
            f"kl={stats.approx_kl:.5f} ref_kl={stats.reference_kl:.5f}",
            flush=True,
        )

    summary = {
        "run_dir": str(run_dir),
        "best_reward_mean": float(best_reward),
        "latest": latest_summary,
        "best_checkpoint": str(run_dir / "checkpoints" / "best_rl.pt"),
        "latest_checkpoint": str(run_dir / "checkpoints" / "latest_rl.pt"),
    }
    with (run_dir / "metrics" / "rl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _make_adapters(cfg: RootConfig) -> list[Any]:
    """Create one adapter per requested RL environment."""

    adapters = []
    for idx in range(int(cfg.rl.num_envs)):
        sim_cfg = copy.deepcopy(cfg)
        sim_cfg.simulator.backend = "isaaclab"
        sim_cfg.experiment.seed = int(cfg.experiment.seed) + idx
        adapters.append(make_sim_adapter(sim_cfg))
    return adapters


def _inject_model_config_from_checkpoint(cfg: RootConfig, ckpt: dict[str, Any]) -> None:
    """Backfill model config fields from checkpoint metadata."""

    model_name = ckpt.get("model_name")
    if isinstance(model_name, str) and model_name:
        cfg.model.name = model_name
    model_cfg = ckpt.get("model_config")
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
    """Save PPO checkpoint with FM policy and optimizer state."""

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
            "rl_config": cfg.rl,
            "metrics": row,
        },
        path,
    )
