"""Validation helpers for ReinFlow and Gaussian PPO configurations."""

from __future__ import annotations

import json
import math
from pathlib import Path

from mini_pi0.config.schema import RootConfig
from mini_pi0.rl.exceptions import ReinFlowConfigError

REINFLOW_ALGORITHM = "reinflow_ppo"
GAUSSIAN_BASELINE_ALGORITHM = "gaussian_ppo_baseline"


def validate_rl_config(cfg: RootConfig, *, require_files: bool = True) -> None:
    """Validate the complete RL contract and referenced artifacts.

    Args:
        cfg: Root configuration containing the ``rl`` section.
        require_files: Validate artifact existence and action-stat contents.

    Raises:
        ReinFlowConfigError: If values or mode combinations are invalid.
        FileNotFoundError: If a required artifact is missing.
    """

    algorithm = _key(cfg.rl.algorithm)
    if algorithm == "flow_ppo":
        raise ReinFlowConfigError(
            "rl.algorithm='flow_ppo' was replaced by 'reinflow_ppo'. "
            "Update the config to use the paper-aligned implementation."
        )
    if algorithm == "ppo":
        raise ReinFlowConfigError(
            "rl.algorithm='ppo' is ambiguous. Use 'reinflow_ppo' or "
            "'gaussian_ppo_baseline'."
        )
    if algorithm not in {REINFLOW_ALGORITHM, GAUSSIAN_BASELINE_ALGORITHM}:
        raise ReinFlowConfigError(
            "rl.algorithm must be 'reinflow_ppo' or 'gaussian_ppo_baseline'."
        )

    _validate_modes(cfg, algorithm)
    _validate_common_scalars(cfg)
    if algorithm == REINFLOW_ALGORITHM:
        _validate_reinflow(cfg)
    if require_files:
        _validate_artifacts(cfg, algorithm)


def _validate_modes(cfg: RootConfig, algorithm: str) -> None:
    """Validate mutually dependent algorithm modes."""

    init_mode = _key(cfg.rl.init_mode)
    action_norm = _key(cfg.rl.action_normalization)
    if init_mode not in {"scratch", "checkpoint"}:
        raise ReinFlowConfigError("rl.init_mode must be 'scratch' or 'checkpoint'.")
    if action_norm not in {"env_bounds", "dataset_stats"}:
        raise ReinFlowConfigError("rl.action_normalization must be 'env_bounds' or 'dataset_stats'.")
    if algorithm == GAUSSIAN_BASELINE_ALGORITHM and init_mode != "checkpoint":
        raise ReinFlowConfigError("gaussian_ppo_baseline only supports checkpoint initialization.")
    if init_mode == "checkpoint" and action_norm != "dataset_stats":
        raise ReinFlowConfigError("Checkpoint fine-tuning requires rl.action_normalization='dataset_stats'.")
    if init_mode == "scratch" and action_norm != "env_bounds":
        raise ReinFlowConfigError("Scratch training requires rl.action_normalization='env_bounds'.")
    if cfg.rl.resume_from and cfg.rl.checkpoint:
        raise ReinFlowConfigError("rl.resume_from is mutually exclusive with rl.checkpoint.")


def _validate_common_scalars(cfg: RootConfig) -> None:
    """Validate PPO scalars shared by both algorithm paths."""

    positive_ints = {
        "rl.total_updates": cfg.rl.total_updates,
        "rl.num_envs": cfg.rl.num_envs,
        "rl.minibatch_size": cfg.rl.minibatch_size,
        "rl.epochs_per_update": cfg.rl.epochs_per_update,
        "rl.eval_episodes": cfg.rl.eval_episodes,
    }
    for name, value in positive_ints.items():
        if int(value) < 1:
            raise ReinFlowConfigError(f"{name} must be >= 1.")
    _validate_fraction("rl.gamma", float(cfg.rl.gamma), allow_one=True)
    _validate_fraction("rl.gae_lambda", float(cfg.rl.gae_lambda), allow_one=True)
    _validate_fraction("rl.clip_ratio", float(cfg.rl.clip_ratio), allow_one=False)
    if float(cfg.rl.max_grad_norm) < 0.0:
        raise ReinFlowConfigError("rl.max_grad_norm must be >= 0.")
    if float(cfg.rl.value_coef) < 0.0 or float(cfg.rl.entropy_coef) < 0.0:
        raise ReinFlowConfigError("RL loss coefficients must be non-negative.")
    if cfg.rl.target_kl is not None and float(cfg.rl.target_kl) <= 0.0:
        raise ReinFlowConfigError("rl.target_kl must be > 0 when set.")


def _validate_reinflow(cfg: RootConfig) -> None:
    """Validate ReinFlow-specific policy, optimizer, and rollout settings."""

    positive_ints = {
        "rl.rollout_decisions_per_update": cfg.rl.rollout_decisions_per_update,
        "rl.flow_steps": cfg.rl.flow_steps,
        "rl.execution_horizon": cfg.rl.execution_horizon,
        "rl.critic_warmup_epochs": cfg.rl.critic_warmup_epochs,
    }
    for name, value in positive_ints.items():
        if int(value) < 1:
            raise ReinFlowConfigError(f"{name} must be >= 1.")
    if int(cfg.rl.execution_horizon) > int(cfg.model.chunk_size):
        raise ReinFlowConfigError("rl.execution_horizon cannot exceed model.chunk_size.")
    if _key(cfg.rl.flow_solver) != "euler":
        raise ReinFlowConfigError("reinflow_ppo currently requires rl.flow_solver='euler'.")
    if _key(cfg.rl.noise_mode) != "learned_diagonal":
        raise ReinFlowConfigError("reinflow_ppo requires rl.noise_mode='learned_diagonal'.")
    if _key(cfg.rl.lr_scheduler) not in {"constant", "cosine"}:
        raise ReinFlowConfigError("rl.lr_scheduler must be 'constant' or 'cosine'.")
    if _key(cfg.rl.dtype) not in {"fp32", "bf16"}:
        raise ReinFlowConfigError("rl.dtype must be 'fp32' or 'bf16'.")
    if _key(cfg.rl.rollout_storage_device) not in {"auto", "cpu", "cuda"}:
        raise ReinFlowConfigError("rl.rollout_storage_device must be 'auto', 'cpu', or 'cuda'.")
    if _key(cfg.rl.reward_strategy) not in {"native", "peg_potential"}:
        raise ReinFlowConfigError("rl.reward_strategy must be 'native' or 'peg_potential'.")
    if float(cfg.rl.actor_lr) <= 0.0 or float(cfg.rl.critic_lr) <= 0.0:
        raise ReinFlowConfigError("rl.actor_lr and rl.critic_lr must be > 0.")
    for name in (
        "actor_weight_decay",
        "critic_weight_decay",
        "reference_w2_coef",
        "reference_transition_kl_coef",
        "velocity_anchor_coef",
    ):
        if float(getattr(cfg.rl, name)) < 0.0:
            raise ReinFlowConfigError(f"rl.{name} must be >= 0.")
    for name in ("critic_warmup_updates", "actor_lr_warmup_updates", "eval_every_updates"):
        if int(getattr(cfg.rl, name)) < 0:
            raise ReinFlowConfigError(f"rl.{name} must be >= 0.")
    _validate_noise(cfg)
    has_reference_loss = any(
        float(value) > 0.0
        for value in (
            cfg.rl.reference_w2_coef,
            cfg.rl.reference_transition_kl_coef,
            cfg.rl.velocity_anchor_coef,
        )
    )
    if _key(cfg.rl.init_mode) == "scratch" and (has_reference_loss or cfg.rl.use_reference_policy):
        raise ReinFlowConfigError("Scratch ReinFlow must not use a checkpoint reference policy.")
    if has_reference_loss and not bool(cfg.rl.use_reference_policy):
        raise ReinFlowConfigError("Reference regularization requires rl.use_reference_policy=true.")


def _validate_noise(cfg: RootConfig) -> None:
    """Validate smooth bounded transition-noise parameters."""

    lower = float(cfg.rl.noise_std_min)
    upper = float(cfg.rl.noise_std_max)
    initial = float(cfg.rl.noise_std_init)
    final_upper = upper if cfg.rl.noise_std_final_max is None else float(cfg.rl.noise_std_final_max)
    if not (0.0 < lower <= initial <= upper):
        raise ReinFlowConfigError("Noise bounds must satisfy 0 < min <= init <= max.")
    if not (lower <= final_upper <= upper):
        raise ReinFlowConfigError("rl.noise_std_final_max must lie within [noise_std_min, noise_std_max].")
    _validate_fraction("rl.noise_schedule_hold_fraction", float(cfg.rl.noise_schedule_hold_fraction), allow_one=True)
    if float(cfg.rl.scratch_policy_clip) <= 0.0:
        raise ReinFlowConfigError("rl.scratch_policy_clip must be > 0.")


def _validate_artifacts(cfg: RootConfig, algorithm: str) -> None:
    """Validate required checkpoint, resume, and action-stat artifacts."""

    if cfg.rl.resume_from:
        _require_file("rl.resume_from", cfg.rl.resume_from)
        return
    needs_checkpoint = _key(cfg.rl.init_mode) == "checkpoint" or algorithm == GAUSSIAN_BASELINE_ALGORITHM
    if needs_checkpoint:
        _require_file("rl.checkpoint", cfg.rl.checkpoint)
        stats_path = _require_file("rl.action_stats_path", cfg.rl.action_stats_path)
        _validate_action_stats(stats_path, int(cfg.model.action_dim))


def _validate_action_stats(path: Path, action_dim: int) -> None:
    """Validate action-stat shapes, finiteness, and positive scales."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mean = [float(value) for value in payload["mean"]]
        std = [float(value) for value in payload["std"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReinFlowConfigError(f"Invalid action statistics at {path}: {exc}") from exc
    if len(mean) != action_dim or len(std) != action_dim:
        raise ReinFlowConfigError(
            f"Action stats dimension mismatch: expected {action_dim}, got mean={len(mean)}, std={len(std)}."
        )
    if not all(math.isfinite(value) for value in (*mean, *std)) or not all(value > 0.0 for value in std):
        raise ReinFlowConfigError("Action statistics must be finite and every std must be positive.")


def _key(value: object) -> str:
    """Normalize an enum-like config value."""

    return str(value or "").strip().lower()


def _validate_fraction(name: str, value: float, *, allow_one: bool) -> None:
    """Validate a scalar in ``[0, 1]`` or ``[0, 1)``."""

    valid = 0.0 <= value <= 1.0 if allow_one else 0.0 <= value < 1.0
    if not valid:
        operator = "<=" if allow_one else "<"
        raise ReinFlowConfigError(f"{name} must satisfy 0 <= value {operator} 1.")


def _require_file(name: str, value: str | None) -> Path:
    """Return a required existing file path."""

    if value is None or not str(value).strip():
        raise FileNotFoundError(f"{name} is required by this RL configuration.")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {value}")
    return path
