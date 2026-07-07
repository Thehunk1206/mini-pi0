from __future__ import annotations

"""Validation helpers for RL fine-tuning configuration."""

from pathlib import Path

from mini_pi0.config.schema import RootConfig


def validate_rl_config(cfg: RootConfig, *, require_files: bool = True) -> None:
    """Validate PPO warm-start configuration.

    Args:
        cfg: Root configuration containing the ``rl`` section.
        require_files: Check checkpoint/action-stat paths exist when true.

    Raises:
        ValueError: If a scalar hyperparameter is invalid.
        FileNotFoundError: If required checkpoint/stat files are absent.
    """

    if str(cfg.rl.algorithm).strip().lower() != "ppo":
        raise ValueError("rl.algorithm must be 'ppo'.")
    if int(cfg.rl.total_updates) < 1:
        raise ValueError("rl.total_updates must be >= 1.")
    if int(cfg.rl.rollout_steps) < 1:
        raise ValueError("rl.rollout_steps must be >= 1.")
    if int(cfg.rl.num_envs) < 1:
        raise ValueError("rl.num_envs must be >= 1.")
    if int(cfg.rl.minibatch_size) < 1:
        raise ValueError("rl.minibatch_size must be >= 1.")
    if int(cfg.rl.epochs_per_update) < 1:
        raise ValueError("rl.epochs_per_update must be >= 1.")
    _validate_fraction("rl.gamma", float(cfg.rl.gamma), lower=0.0, upper=1.0, allow_one=True)
    _validate_fraction("rl.gae_lambda", float(cfg.rl.gae_lambda), lower=0.0, upper=1.0, allow_one=True)
    _validate_fraction("rl.clip_ratio", float(cfg.rl.clip_ratio), lower=0.0, upper=1.0, allow_one=False)
    if float(cfg.rl.lr) <= 0.0:
        raise ValueError("rl.lr must be > 0.")
    if float(cfg.rl.kl_coef) < 0.0:
        raise ValueError("rl.kl_coef must be >= 0.")
    if float(cfg.rl.value_coef) < 0.0:
        raise ValueError("rl.value_coef must be >= 0.")
    if float(cfg.rl.entropy_coef) < 0.0:
        raise ValueError("rl.entropy_coef must be >= 0.")
    if float(cfg.rl.max_grad_norm) < 0.0:
        raise ValueError("rl.max_grad_norm must be >= 0.")
    if cfg.rl.target_kl is not None and float(cfg.rl.target_kl) <= 0.0:
        raise ValueError("rl.target_kl must be > 0 when set.")
    if require_files:
        _require_file("rl.checkpoint", cfg.rl.checkpoint)
        _require_file("rl.action_stats_path", cfg.rl.action_stats_path)


def _validate_fraction(name: str, value: float, *, lower: float, upper: float, allow_one: bool) -> None:
    """Validate a scalar fraction."""

    upper_ok = value <= upper if allow_one else value < upper
    if value < lower or not upper_ok:
        hi = "<=" if allow_one else "<"
        raise ValueError(f"{name} must satisfy {lower} <= value {hi} {upper}.")


def _require_file(name: str, path: str) -> None:
    """Require a path to exist."""

    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
